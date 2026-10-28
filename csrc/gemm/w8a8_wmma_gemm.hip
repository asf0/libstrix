// SPDX-License-Identifier: Apache-2.0
// W8A8 iu8-WMMA GEMM for larger M: uint8b128 weights, activations quantized
// per 128-element group inside the op, fp32 group scales.
// Moved from the asf0/strix-vllm fork (csrc/rocm/skinny_gemms_w8a8.cu), itself derived
// from vLLM's csrc/rocm/skinny_gemms.cu (Apache-2.0, vLLM contributors).
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W8A8 grouped skinny GEMM for gfx115x wave32 RDNA: uint8b128 weights, int8
// activations quantized on the fly, native iu8 16x16x16 WMMA.
//
// Own TU: the WMMA kernels are kept apart from the dot2/fdot2 kernels in
// skinny_gemms_int8.cu (see the in-tree miscompile notes on mixing WMMA with
// the legacy dot kernels in one TU).
//
// Math, per output element (m, n):
//   out[m,n] = sum_g s_a[m][g] * s_w[n][g] * (sum_{k in g} q[m,k] * w[n,k])
// with
//   q[m,k]    = clamp(rint(x[m,k] / s_a[m][g]), -127, 127)
//   s_a[m][g] = max_{k in g}|x[m,:]| / 127 (1 if the group is all zero)
//   w[n,k]    = raw uint8b128 byte - 128
//   s_w[n][g] = weight scale of K-group g (g = 128 K elements)
//
// The activation scale is per (row, K-group): a lone outlier only coarsens its
// own 128-element slice instead of the whole row (the naive per-token scheme
// diluted a 10x spike over all 40 groups and measured 3.3x the quantization
// error of this one on outlier-heavy rows; at model level the per-token scheme
// cost 2.6 points of teacher-forced top-1 agreement and never shipped).
//
// The k-loop keeps the WMMA i32 accumulator for the current group and folds it
// into an fp32 accumulator at every group boundary (a 128-byte LDS chunk is
// exactly one group):
//   accf += (f32)accg * (s_w[n][g] * s_a[m][g]);   accg = 0
//
// The weights stay raw uint8b128 in global memory - every W8 path reads the
// same buffer, so no path carries a converted copy - and the byte-center shift
// is folded into the LDS commit: the 16 staged bytes of a block are XORed with
// 0x80 on the way into shared memory, which turns the excess-128 byte into its
// two's-complement value. The WMMA then reads the true signed weight
// (bSign=true) and the fold needs no rowsum correction. Applying the XOR at
// the commit point instead of at the load keeps the staging loads in flight
// across the whole chunk compute, where their latency is hidden; the cost is
// one v_xor_b32 per 4 bytes staged.
//
// Weight scales are read in group-major order [G][N] (per-channel scales are
// the degenerate single group [N]), so the per-group scale load is one
// coalesced 16-lane read: the checkpoint's [N][G] layout gathers 16 sectors
// per warp and measured 29-44% slower at decode M. Runtime strides keep [N][G]
// working too. Scales are fp32; the caller keeps one fp32 [G][N] copy per
// layer (0.7 GiB at 27B), far cheaper than a converted weight copy.
//
// The chip reports 20 "compute units" (WGPs); the GPU has 2 CUs per WGP.
// Split-K uses whole 128-element groups only, and its partials are fp32
// [k_split, M, N].

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <climits>
#include <string>

namespace strix {

// The iu8 WMMA builtin below is gfx11-family wave32; the tuning was done on
// gfx115x (Strix Halo/Krackan). The host entry point rejects other devices;
// non-gfx115x offload targets compile a trapping stub for the kernel bodies
// instead. The macro only exists in device passes, so the *bodies* - never the
// kernel definitions or the host-side launchers - carry the guard, and it is
// defined (again) inside the extracted section below.
// ====================== BEGIN w8a8 device code ======================
// Everything between the markers is self-contained HIP device/host-launch code
// (no torch types); the standalone Phase-B harness extracts this section
// verbatim so the kernels validated there are the ones that ship here. The
// target macro is part of the section for the same reason.
#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || \
    defined(__gfx1153__)
  #define __HIP__GFX115X__ 1
#endif

namespace w8a8 {

using v4i32 = int __attribute__((ext_vector_type(4)));
using v8i32 = int __attribute__((ext_vector_type(8)));
using v8f32 = float __attribute__((ext_vector_type(8)));

// K elements per weight-scale group; the fold below is hard-wired to one group
// per 128-byte LDS chunk, so grouped scales must be g=128.
constexpr int kGroupK = 128;

template <typename scalar_t>
__device__ __forceinline__ float w8a8_to_float(scalar_t v);

template <>
__device__ __forceinline__ float w8a8_to_float(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float w8a8_to_float(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t w8a8_from_float(float v);

template <>
__device__ __forceinline__ half w8a8_from_float<half>(float v) {
  return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __hip_bfloat16
w8a8_from_float<__hip_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// Group-scale W8A8 GEMM. CTA tile BM = 16*BMF*WM rows x BN = 16*BNF*WN cols
// (WN = WAVES / WM): the waves form a WM x WN grid and each owns BMF x BNF
// fragments. WM = 1 is the original layout (every wave spans the full BM);
// WM > 1 trades A-fragment LDS reads for B ones - per k-tile a wave loads
// BMF + BNF fragments for BMF * BNF WMMAs. KBT=128 bytes of K are
// staged per row per chunk (one group). LDS = 2*KBT*(BM+BN) bytes, double
// buffered, so BM+BN <= 256 at KBT=128 (64 KiB).
// W4 = true reads int4 weights instead: [N][B_row_stride] bytes holding K/2
// bytes per row in vLLM's ExLlama-shuffled int32 words (k order 0 2 4 6 1 3 5 7
// per nibble), symmetric uint4b8. Each 16-element k-tile is 8 bytes, expanded
// at the LDS commit to the uint8b128 bytes the W8 path uses (nibble + 120),
// so everything after the commit is shared.
// WG is the weight-scale group: 128 (one per LDS chunk, folded at the chunk
// end) or 32 (scales [K/32][N]; four folds per chunk, every two k-tiles, each
// with its own weight scale and the chunk's activation scale).
// SWZ > 0 rasterizes blocks in groups of SWZ m-blocks (all n-blocks in a
// group, m fastest, Triton's grouped ordering): one wave of CTAs then covers a
// small rectangle of the output, so the A panel and the weight tile it reads
// come from L2 instead of being re-read once per block. Without it each
// operand is read once per output block, which is what the tile sweep measured.
// STAGGER starts every CTA at a different chunk of its own K slice and wraps
// around at the slice end, as mxfp4_wmma_gemm does (see the comment there).
// It is a bench-only variant here (cfg 12-16) and the reason is measured: the
// trick pays 7-21% when the weight rows are a whole number of KiB apart
// (5120, 6144, 17408 B - every 27B INT8 shape as a contiguous [N, K]), but the
// plugin pads each of those rows by 128 B off the 1 KiB cliff before the op
// ever sees them (_pack_w8_rows), which is worth 5-20% by itself and leaves
// the rotation 0-5% (1-2% in situ). Production keeps the padding, and the
// rotation's own stride is not a knob: 1, 3, 7, 11 and 19 chunks between
// neighbouring n-blocks all measured within 1%.
template <typename scalar_t, int BMF, int BNF, int WAVES, bool W4, int WG,
          int SWZ, bool STAGGER, int WM = 1>
__global__ void __launch_bounds__(WAVES * 32, 1) w8a8_gemm_group_kernel(
    const signed char* __restrict__ A,    // [M][K] int8 activations
    const unsigned char* __restrict__ B,  // [N][B_row_stride] uint8b128 raw
    const float* __restrict__ sa_pg,      // [G][t16][16] per-(row,group) scale
    const float* __restrict__ s_w,        // [G][N] fp32 ([N] per-channel)
    scalar_t* __restrict__ Cout,          // [M][N]
    float* __restrict__ partials,         // [k_split][M][N] fp32, or null
    const int M, const int N, const int K, const int kper, const int G,
    const int t16, const int B_row_stride, const int sw_nstride,
    const int sw_gstride) {
#if defined(__HIP__GFX115X__)
  constexpr int KBT = kGroupK;  // one LDS chunk == one weight group
  constexpr int NT = KBT / 16;
  static_assert(WAVES % WM == 0, "wave grid");
  constexpr int WN = WAVES / WM;
  constexpr int BM = 16 * BMF * WM;
  constexpr int BN = 16 * BNF * WN;
  constexpr int NBA = NT * BMF * WM;
  constexpr int NBB = NT * BNF * WN;
  constexpr int RA = (NBA + 2 * WAVES - 1) / (2 * WAVES);
  constexpr int RB = (NBB + 2 * WAVES - 1) / (2 * WAVES);
  constexpr bool kTA = (NBA % (2 * WAVES) == 0);
  constexpr bool kTB = (NBB % (2 * WAVES) == 0);

  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int lo = lane & 15, hi = lane >> 4;
  const int wm = wave / WN, wn = wave % WN;  // this wave's cell of the grid
  int bid_m = blockIdx.y, bid_n = blockIdx.x;
  if constexpr (SWZ > 0) {
    const int pid = blockIdx.x + blockIdx.y * (int)gridDim.x;
    const int per_group = SWZ * (int)gridDim.x;
    const int group = pid / per_group;
    const int first_m = group * SWZ;
    const int gm = min((int)gridDim.y - first_m, SWZ);
    bid_m = first_m + pid % gm;
    bid_n = (pid % per_group) / gm;
  }
  const int m_base = bid_m * BM;
  const int n_base = bid_n * BN;
  const int t_base = m_base >> 4;  // first 16-row tile of this CTA
  const int g_begin = (int)blockIdx.z * (kper / kGroupK);
  const int ngroups = kper / kGroupK;
  const int nchunks = kper / KBT;
  const int k_begin = (int)blockIdx.z * kper;
  // Rotated K start: chunk c of the loop is chunk ck of this CTA's own slice
  // [k_begin, k_begin + kper), and the wrap is inside that slice, so split-K
  // partials still cover exactly their own K range once and the per-group fold
  // still runs once per group (it follows ck, not the loop order). With
  // STAGGER false ck == c, ck1 == c + 1 and ck2 == c + 2, i.e. the linear loop
  // this kernel shipped with.
  const int stagger =
      STAGGER ? (bid_n * 7 + bid_m * 3 + (int)blockIdx.z * 5) % nchunks : 0;
  int ck = stagger;
  int ck1 = ck + 1 < nchunks ? ck + 1 : 0;
  int ck2 = ck1 + 1 < nchunks ? ck1 + 1 : 0;

  // 256-byte fragment blocks: [chunk buffer][k-tile][m-fragment][row lane *
  // 16].
  __shared__ alignas(16) signed char a_lds[2][NT][BMF * WM][256];
  __shared__ alignas(16) unsigned char b_lds[2][NT][BNF * WN][256];

  v8i32 accg[BMF][BNF];
  v8f32 accf[BMF][BNF];
  #pragma unroll
  for (int f = 0; f < BMF; ++f)
  #pragma unroll
    for (int g = 0; g < BNF; ++g) {
      accg[f][g] = v8i32{0, 0, 0, 0, 0, 0, 0, 0};
      accf[f][g] = v8f32{0, 0, 0, 0, 0, 0, 0, 0};
    }

  // Register-staged pipeline: chunk c+1 is loaded into registers before the
  // compute of chunk c and committed to LDS between two barriers after it.
  static_assert(NBA <= 2 * WAVES * RA && NBB <= 2 * WAVES * RB, "staging");
  v4i32 sgA[RA], sgB[RB];
  auto load_stage = [&](int k0) {
  #pragma unroll
    for (int r = 0; r < RA; ++r) {
      const int blk = r * 2 * WAVES + wave * 2 + hi;
      v4i32 v = {0, 0, 0, 0};
      if (kTA || blk < NBA) {
        const int f = blk / NT, t = blk % NT;
        const int row = m_base + 16 * f + lo;
        if (row < M)
          __builtin_memcpy(&v, A + (size_t)row * K + k0 + 16 * t, 16);
      }
      sgA[r] = v;
    }
  #pragma unroll
    for (int r = 0; r < RB; ++r) {
      const int blk = r * 2 * WAVES + wave * 2 + hi;
      v4i32 v = {0, 0, 0, 0};
      if (kTB || blk < NBB) {
        const int g = blk / NT, t = blk % NT;
        const int row = n_base + 16 * g + lo;
        if (row < N) {
          if constexpr (W4)
            __builtin_memcpy(&v, B + (size_t)row * B_row_stride + (k0 + 16 * t) / 2,
                             8);
          else
            __builtin_memcpy(&v, B + (size_t)row * B_row_stride + k0 + 16 * t,
                             16);
        }
      }
      // Raw uint8b128 bytes straight from global; the byte-center shift is
      // folded into the LDS commit below.
      sgB[r] = v;
    }
  };
  auto store_stage = [&](int buf) {
  #pragma unroll
    for (int r = 0; r < RA; ++r) {
      const int blk = r * 2 * WAVES + wave * 2 + hi;
      if (kTA || blk < NBA)
        __builtin_memcpy(&a_lds[buf][blk % NT][blk / NT][lo * 16], &sgA[r], 16);
    }
  #pragma unroll
    for (int r = 0; r < RB; ++r) {
      const int blk = r * 2 * WAVES + wave * 2 + hi;
      if (kTB || blk < NBB) {
        // Byte-center shift at the LDS commit: raw uint8b128 bytes XOR 0x80
        // decode to (byte - 128) as *signed* int8, so the WMMA below reads the
        // true signed weight with no global prep copy and no fold correction.
        // Doing it here (instead of at the load) keeps the staging loads in
        // flight across the whole chunk, where their latency is hidden.
        if constexpr (W4) {
          // ExLlama word -> k0..k3 / k4..k7 bytes: low nibbles hold
          // k0 k4 k1 k5, high nibbles k2 k6 k3 k7.
          uint32_t o[4];
  #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t w = (uint32_t)sgB[r][h];
            const uint32_t lo = w & 0x0F0F0F0Fu, hn = (w >> 4) & 0x0F0F0F0Fu;
            o[2 * h] = __builtin_amdgcn_perm(hn, lo, 0x06040200u) + 0x78787878u;
            o[2 * h + 1] = __builtin_amdgcn_perm(hn, lo, 0x07050301u) + 0x78787878u;
          }
  #pragma unroll
          for (int q = 0; q < 4; ++q) sgB[r][q] = (int)o[q];
        }
  #pragma unroll
        for (int q = 0; q < 4; ++q) sgB[r][q] ^= (int)0x80808080;
        __builtin_memcpy(&b_lds[buf][blk % NT][blk / NT][lo * 16], &sgB[r], 16);
      }
    }
  };

  load_stage(k_begin + ck * KBT);
  store_stage(0);
  __syncthreads();
  const v8i32 Z{0, 0, 0, 0, 0, 0, 0, 0};
  // Group scales are prefetched two groups ahead of the fold that consumes them
  // (three live buffers, rotated by BN moves at each fold).
  float swfA[BNF], swfB[BNF], swfC[BNF];
  auto load_sw = [&](float (&dst)[BNF], int gj) {
    const int gc = g_begin + gj;
  #pragma unroll
    for (int g = 0; g < BNF; ++g) {
      const int n = n_base + 16 * (wn * BNF + g) + lo;
      dst[g] = (n < N) ? s_w[(size_t)n * sw_nstride + (size_t)gc * sw_gstride]
                       : 0.0f;
    }
  };
  static_assert(WG == kGroupK || WG == 32, "weight group must be 128 or 32");
  constexpr int WSUB = kGroupK / WG;  // weight groups per chunk
  constexpr int TPG = NT / WSUB;      // k-tiles per weight group
  if constexpr (WG == kGroupK) {
    load_sw(swfA, ck);
    if (nchunks > 1) load_sw(swfB, ck1);
  }
  int cur = 0;
  for (int c = 0; c < nchunks; ++c) {
    if (c + 1 < nchunks) load_stage(k_begin + ck1 * KBT);
    // Group boundaries follow the chunk's own K position, not the loop order.
    const bool gend = ((ck + 1) % (kGroupK / KBT) == 0);
    const int gn = ck + 1;  // groups finished at gend
    float sws[WSUB][BNF];
    // The chunk's activation scales are reloaded at each fold (cached): kept
    // live for all fragments, BMF = 8 spilled ~1 KB per lane.
    const float* sag_c = sa_pg + (size_t)(g_begin + ck) * t16 * 16;
    if constexpr (WG == kGroupK) {
      // The prefetch is two loop steps ahead, i.e. the group of chunk ck2.
      if (gend && c + 2 < nchunks) load_sw(swfC, ck2);
    } else {
  #pragma unroll
      for (int q = 0; q < WSUB; ++q)
  #pragma unroll
        for (int g = 0; g < BNF; ++g) {
          const int n = n_base + 16 * (wn * BNF + g) + lo;
          const size_t gc = (size_t)(g_begin + ck) * WSUB + q;
          sws[q][g] = (n < N) ? s_w[(size_t)n * sw_nstride + gc * sw_gstride] : 0.0f;
        }
    }
  #pragma unroll
    for (int t = 0; t < NT; ++t) {
      v4i32 af[BMF];
  #pragma unroll
      for (int f = 0; f < BMF; ++f)
        __builtin_memcpy(&af[f], &a_lds[cur][t][wm * BMF + f][lo * 16], 16);
  #pragma unroll
      for (int g = 0; g < BNF; ++g) {
        v4i32 bf;
        __builtin_memcpy(&bf, &b_lds[cur][t][wn * BNF + g][lo * 16], 16);
  #pragma unroll
        for (int f = 0; f < BMF; ++f) {
          // The first k-tile of a group initializes the i32 accumulator from a
          // constant zero instead of a copy of the previous group's result.
          const v8i32 cin = (t % TPG == 0) ? Z : accg[f][g];
          accg[f][g] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, af[f], true, bf, cin, true);
        }
      }
      if constexpr (WG != kGroupK) {
        if (t % TPG == TPG - 1) {
  #pragma unroll
          for (int f = 0; f < BMF; ++f) {
            v8f32 sa8;
            __builtin_memcpy(&sa8, sag_c + (t_base + wm * BMF + f) * 16 + hi * 8, 32);
  #pragma unroll
            for (int g = 0; g < BNF; ++g) {
              const float sw = sws[t / TPG][g];
  #pragma unroll
              for (int i = 0; i < 8; ++i)
                accf[f][g][i] += (float)accg[f][g][i] * (sw * sa8[i]);
            }
          }
        }
      }
    }
    // Group boundary: fold the raw i32 accumulator into fp32 with the product
    // of the weight scale and the per-(row, group) activation scale.
    if (WG == kGroupK && gend) {
      const float* sag = sa_pg + (size_t)(g_begin + gn - 1) * t16 * 16;
  #pragma unroll
      for (int f = 0; f < BMF; ++f) {
        // The lane's 8 C rows of this m-fragment are 16*f + 2*i + hi, so its
        // per-group scales are one contiguous 32 B chunk (tile order hi*8 + i).
        v8f32 sa8;
        __builtin_memcpy(&sa8, sag + (t_base + wm * BMF + f) * 16 + hi * 8, 32);
  #pragma unroll
        for (int g = 0; g < BNF; ++g) {
          const float sw = swfA[g];
  #pragma unroll
          for (int i = 0; i < 8; ++i)
            accf[f][g][i] += (float)accg[f][g][i] * (sw * sa8[i]);
        }
      }
      if (c + 1 < nchunks)
  #pragma unroll
        for (int g = 0; g < BNF; ++g) {
          swfA[g] = swfB[g];
          swfB[g] = swfC[g];
        }
    }
    __syncthreads();
    if (c + 1 < nchunks) store_stage(cur ^ 1);
    __syncthreads();
    cur ^= 1;
    ck = ck1;
    ck1 = ck2;
    ck2 = ck2 + 1 < nchunks ? ck2 + 1 : 0;
  }

  // ---- epilogue ----------------------------------------------------------
  // The per-group activation scales are already in accf; split-K writes the
  // scaled fp32 partials and the reduce kernel only sums them.
  #pragma unroll
  for (int f = 0; f < BMF; ++f) {
    int mm[8];
  #pragma unroll
    for (int i = 0; i < 8; ++i) mm[i] = m_base + 16 * (wm * BMF + f) + 2 * i + hi;
  #pragma unroll
    for (int g = 0; g < BNF; ++g) {
      const int n = n_base + 16 * (wn * BNF + g) + lo;
      if (n < N) {
  #pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (mm[i] < M) {
            const float v = accf[f][g][i];
            if (partials)
              partials[((size_t)blockIdx.z * M + mm[i]) * N + n] = v;
            else
              Cout[(size_t)mm[i] * N + n] = w8a8_from_float<scalar_t>(v);
          }
        }
      }
    }
  }
#else
  __builtin_trap();
#endif
}

// ------------------------------- act-quant -------------------------------
// One CTA per row, T threads, CPT K-elements per thread-chunk (CPT/8 x 16 B
// vectors, 16B aligned). Two passes over the row: chunk maxima (staged in
// dynamic shared memory) then quantize with the per-group scale. Convention,
// for the K-group g of an element:
//   q = clamp(rint(x / s), -127, 127),  s = max_{k in g}|x| / 127  (1 if 0).
// The per-group scales go to the GEMM fold's permuted layout,
//   sa_pg[(g * t16 + m/16) * 16 + hi * 8 + i],  hi = (m%16)&1, i = (m%16)>>1,
// i.e. the 16 rows of a tile in C-fragment lane order, so a fold lane reads
// one 32 B chunk per (m-fragment, group). t16 is the tile stride; slots for
// rows >= M (and tiles past t16) must be zero - the caller zero-fills.
template <typename scalar_t, int T, int CPT>
__global__ void __launch_bounds__(T)
    act_quant_kernel(const scalar_t* __restrict__ x,
                     signed char* __restrict__ q, float* __restrict__ sa_pg,
                     const int M, const int K, const int nchunk,
                     const int ngroups, const int t16) {
#if defined(__HIP__GFX115X__)
  static_assert(kGroupK % CPT == 0, "a group is a whole number of chunks");
  constexpr int CPG = kGroupK / CPT;  // chunks per group
  extern __shared__ float smem[];
  float* chunk_max = smem;          // [nchunk]
  float* inv_l = smem + nchunk;     // [ngroups]
  const int m = blockIdx.x;
  const int t = threadIdx.x;
  const scalar_t* xr = x + (size_t)m * K;

  for (int c = t; c < nchunk; c += T) {
    const int base = c * CPT;
    float mx = 0.0f;
  #pragma unroll
    for (int v = 0; v < CPT / 8; ++v) {
      const int o = base + v * 8;
      if (o + 8 <= K) {
        const uint4 raw = *reinterpret_cast<const uint4*>(xr + o);
        const scalar_t* h = reinterpret_cast<const scalar_t*>(&raw);
  #pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float f = fabsf(w8a8_to_float(h[i]));
          mx = f > mx ? f : mx;
        }
      } else {
        for (int i = 0; i < 8 && o + i < K; ++i) {
          const float f = fabsf(w8a8_to_float(xr[o + i]));
          mx = f > mx ? f : mx;
        }
      }
    }
    chunk_max[c] = mx;
  }
  __syncthreads();

  // One thread per group: the group scale is the max of its CPG chunk maxima.
  for (int g = t; g < ngroups; g += T) {
    float mx = 0.0f;
  #pragma unroll
    for (int i = 0; i < CPG; ++i) {
      const int c = g * CPG + i;
      if (c < nchunk && c * CPT < K) {
        const float v = chunk_max[c];
        mx = v > mx ? v : mx;
      }
    }
    const float sc = (mx > 0.0f) ? mx / 127.0f : 1.0f;
    inv_l[g] = 1.0f / sc;
    sa_pg[((size_t)g * t16 + (m >> 4)) * 16 + ((m & 15) & 1) * 8 +
          ((m & 15) >> 1)] = sc;
  }
  __syncthreads();

  for (int c = t; c < nchunk; c += T) {
    const int base = c * CPT;
    const float inv = inv_l[c / CPG];
  #pragma unroll
    for (int v = 0; v < CPT / 8; ++v) {
      const int o = base + v * 8;
      signed char out[8];
      if (o + 8 <= K) {
        const uint4 raw = *reinterpret_cast<const uint4*>(xr + o);
        const scalar_t* h = reinterpret_cast<const scalar_t*>(&raw);
  #pragma unroll
        for (int i = 0; i < 8; ++i) {
          int qq = (int)rintf(w8a8_to_float(h[i]) * inv);
          qq = qq > 127 ? 127 : (qq < -127 ? -127 : qq);
          out[i] = (signed char)qq;
        }
      } else {
        for (int i = 0; i < 8; ++i) {
          const int idx = o + i;
          int qq = 0;
          if (idx < K) {
            qq = (int)rintf(w8a8_to_float(xr[idx]) * inv);
            qq = qq > 127 ? 127 : (qq < -127 ? -127 : qq);
          }
          out[i] = (signed char)qq;
        }
      }
      __builtin_memcpy(q + (size_t)m * K + o, out, 8);
    }
  }
#else
  __builtin_trap();
#endif
}

// ------------------------------- split-K reduce
// ------------------------------- Partials carry both the weight and the
// per-(row, group) activation scales from the GEMM fold, so this only sums the
// k-slices.
template <typename scalar_t>
__global__ void w8a8_reduce_splitk_kernel(const float* __restrict__ partials,
                                          scalar_t* __restrict__ Cout,
                                          const int M, const int N,
                                          const int k_split) {
#if defined(__HIP__GFX115X__)
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)M * N) return;
  const int m = (int)(idx / N);
  const int n = (int)(idx - (long long)m * N);
  float acc = 0.0f;
  for (int z = 0; z < k_split; ++z) {
    acc += partials[((long long)z * M + m) * N + n];
  }
  Cout[idx] = w8a8_from_float<scalar_t>(acc);
#else
  __builtin_trap();
#endif
}

// ------------------------------- launchers -------------------------------
template <typename scalar_t, int BMF, int BNF, int WAVES, bool W4, int WG,
          int SWZ, bool STAGGER = false, int WM = 1>
void launch_gemm(const signed char* A, const unsigned char* B,
                 const float* sa_pg, const float* s_w, scalar_t* Cout,
                 float* partials, const int M, const int N, const int K,
                 const int k_split, const int t16, const int B_row_stride,
                 const int sw_nstride, const int sw_gstride, hipStream_t st) {
  constexpr int BM = 16 * BMF * WM, BN = 16 * BNF * (WAVES / WM);
  dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((M + BM - 1) / BM),
            (unsigned)k_split);
  const int kper = K / k_split;
  const int G = K / kGroupK;
  w8a8_gemm_group_kernel<scalar_t, BMF, BNF, WAVES, W4, WG, SWZ, STAGGER, WM>
      <<<grid, WAVES * 32, 0, st>>>(A, B, sa_pg, s_w, Cout, partials, M, N, K,
                                    kper, G, t16, B_row_stride, sw_nstride,
                                    sw_gstride);
}

// Tile by batch: the smallest tile that still covers M in ONE m-block, because
// a second m-block re-reads the whole weight matrix (bench/kernels/w8a16_decode.py:
// batch 40 streams 68 GB/s at BM 32 against 172 GB/s at BM 64). Tiles stay
// within 16 * BMF fragment rows and BM + BN <= 256 for the 64 KB staging, and
// the 32-element-group path is capped at BM 64 (BMF 8 spills its four folds).
constexpr int kBM16Boundary = 16;
constexpr int kBM32Boundary = 32;
constexpr int kBM64Boundary = 64;

// Bench-only tile overrides, one per (BMF, BNF, WAVES) under test; -1 keeps
// the production choice below. All of them keep BM * BN and the accumulator
// count within the register and LDS budget: fragments are BMF * BNF * 16
// VGPRs (128 for every config here) and the staging is 2 * 128 * (BM + BN)
// bytes, so BM + BN <= 256.
#define STRIX_W8A8_TILE(BMF, BNF, WAVES, SWZ)                                \
  launch_gemm<scalar_t, BMF, BNF, WAVES, W4, WG, SWZ>(                        \
      A, B, sa_pg, s_w, Cout, partials, M, N, K, k_split, t16, B_row_stride,  \
      sw_nstride, sw_gstride, st)
// A WM x (WAVES / WM) wave grid (see WM on the kernel).
#define STRIX_W8A8_TILE_G(BMF, BNF, WAVES, SWZ, WM)                          \
  launch_gemm<scalar_t, BMF, BNF, WAVES, W4, WG, SWZ, false, WM>(             \
      A, B, sa_pg, s_w, Cout, partials, M, N, K, k_split, t16, B_row_stride,  \
      sw_nstride, sw_gstride, st)
// Same tile with the rotated K start (see STAGGER on the kernel).
#define STRIX_W8A8_TILE_S(BMF, BNF, WAVES, SWZ)                              \
  launch_gemm<scalar_t, BMF, BNF, WAVES, W4, WG, SWZ, true>(                  \
      A, B, sa_pg, s_w, Cout, partials, M, N, K, k_split, t16, B_row_stride,  \
      sw_nstride, sw_gstride, st)
template <typename scalar_t, bool W4, int WG>
void launch_gemm_sized(const signed char* A, const unsigned char* B,
                       const float* sa_pg, const float* s_w, scalar_t* Cout,
                       float* partials, const int M, const int N, const int K,
                       const int k_split, const int t16, const int B_row_stride,
                       const int sw_nstride, const int sw_gstride,
                       hipStream_t st, const int cfg) {
  if (cfg >= 0) {
    switch (cfg) {
      case 0: STRIX_W8A8_TILE(8, 1, 8, 0); return;  // production, M > 32
      case 1: STRIX_W8A8_TILE(4, 2, 4, 0); return;  // BM 64, BN 128, 48 KB LDS
      case 2: STRIX_W8A8_TILE(4, 1, 8, 0); return;  // BM 64, BN 128 (g32 shape)
      case 3: STRIX_W8A8_TILE(2, 2, 4, 0); return;  // BM 32, BN 128
      case 4: STRIX_W8A8_TILE(2, 1, 8, 0); return;  // production, M <= 32
      case 5: STRIX_W8A8_TILE(8, 1, 8, 8); return;  // + 8 x n grouped raster
      case 6: STRIX_W8A8_TILE(8, 1, 8, 4); return;  // + 4 x n grouped raster
      case 7: STRIX_W8A8_TILE(4, 2, 4, 8); return;  // smaller tile, swizzled
      // Decode tiles: 16 accumulator VGPRs instead of 128, so far more CTAs
      // fit per WGP and the weight stream has more loads in flight (batch
      // 16..64 runs at 153 GB/s of the 213 GB/s peak).
      case 8: STRIX_W8A8_TILE(1, 1, 8, 0); return;   // BM 16, BN 128, 36 KB
      case 9: STRIX_W8A8_TILE(1, 1, 4, 0); return;   // BM 16, BN 64, 20 KB
      case 10: STRIX_W8A8_TILE(2, 1, 4, 0); return;  // BM 32, BN 64, 24 KB
      // One m-block for a batch of 33..64: two m-blocks read the whole weight
      // matrix twice, which is what halves the rate at those batches.
      case 11: STRIX_W8A8_TILE(4, 1, 8, 0); return;  // BM 64, BN 128, 48 KB
      // The same decode tiles with the rotated K start (STAGGER): 12 and 15
      // are the production shape at M 33..64 with and without the grouped
      // raster (identical at decode, where there is one m-block).
      case 12: STRIX_W8A8_TILE_S(4, 1, 8, 0); return;  // BM 64, K staggered
      case 13: STRIX_W8A8_TILE_S(2, 1, 8, 0); return;  // BM 32, K staggered
      case 14: STRIX_W8A8_TILE_S(1, 1, 8, 0); return;  // BM 16, K staggered
      case 15: STRIX_W8A8_TILE_S(4, 1, 8, 8); return;  // BM 64, swizzled + staggered
      case 16: STRIX_W8A8_TILE_S(8, 1, 8, 8); return;  // prefill tile, staggered
      // The prefill tile as a 2 x 4 / 4 x 2 wave grid (same 128 x 128 block,
      // same accumulators): 6 fragment loads per 8 WMMAs instead of 9.
      case 17: STRIX_W8A8_TILE_G(4, 2, 8, 8, 2); return;
      case 18: STRIX_W8A8_TILE_G(2, 4, 8, 8, 4); return;
      default: break;
    }
  }
  // No STAGGER: the caller's 128 B row pad already breaks the pitch aliasing
  // the rotation would fix (0-3% left, under the run-to-run noise end to end).
  if (M <= kBM16Boundary)
    launch_gemm<scalar_t, 1, 1, 8, W4, WG, 0>(A, B, sa_pg, s_w, Cout, partials, M, N, K,
                                   k_split, t16, B_row_stride, sw_nstride,
                                   sw_gstride, st);
  else if (M <= kBM32Boundary)
    launch_gemm<scalar_t, 2, 1, 8, W4, WG, 0>(A, B, sa_pg, s_w, Cout, partials, M, N, K,
                                   k_split, t16, B_row_stride, sw_nstride,
                                   sw_gstride, st);
  else if constexpr (WG != kGroupK)
    // Four folds per chunk at BMF = 8 spill ~1 KB per lane; 64-row tiles fit.
    launch_gemm<scalar_t, 4, 1, 8, W4, WG, 0>(A, B, sa_pg, s_w, Cout, partials, M, N, K,
                                   k_split, t16, B_row_stride, sw_nstride,
                                   sw_gstride, st);
  else if (M <= kBM64Boundary)
    launch_gemm<scalar_t, 4, 1, 8, W4, WG, 8>(A, B, sa_pg, s_w, Cout, partials, M, N, K,
                                   k_split, t16, B_row_stride, sw_nstride,
                                   sw_gstride, st);
  else
    // 8 x n grouped raster: +6% on qkvz/gate_up/down, -1.5% on o_proj (which
    // has 40 n-blocks where the others have 112 / 272). Measured in
    // bench/kernels/w8a8_gemm.py, cfg 0 vs 5. The 2 x 4 wave grid (cfg 17)
    // over the same 128 x 128 block loads 6 fragments per 8 WMMAs instead of
    // 9: +4-9 % on the four 27B shapes (37.4 -> 38.9 TOPS on qkvz).
    launch_gemm<scalar_t, 4, 2, 8, W4, WG, 8, false, 2>(A, B, sa_pg, s_w, Cout, partials,
                                                M, N, K, k_split, t16, B_row_stride,
                                                sw_nstride, sw_gstride, st);
}
#undef STRIX_W8A8_TILE

constexpr int kActQThreads = 256;
constexpr int kActQChunk = 32;

template <typename scalar_t>
void launch_actq(const scalar_t* x, signed char* q, float* sa_pg, const int M,
                 const int K, const int t16, hipStream_t st) {
  const int nchunk = (K + kActQChunk - 1) / kActQChunk;
  const int ngroups = (K + kGroupK - 1) / kGroupK;
  const int smem = (nchunk + ngroups) * (int)sizeof(float);
  act_quant_kernel<scalar_t, kActQThreads, kActQChunk>
      <<<M, kActQThreads, smem, st>>>(x, q, sa_pg, M, K, nchunk, ngroups, t16);
}

template <typename scalar_t>
void launch_reduce(const float* partials, scalar_t* Cout, const int M,
                   const int N, const int k_split, hipStream_t st) {
  const long long total = (long long)M * N;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  w8a8_reduce_splitk_kernel<scalar_t><<<(unsigned)blocks, threads, 0, st>>>(
      partials, Cout, M, N, k_split);
}

// CTA tile of the config that launch_gemm_sized picks for this M.
inline void w8a8_cfg_tile(const int M, int& bm, int& bn, const bool g32 = false) {
  if (M <= kBM16Boundary)
    bm = 16;
  else if (M <= kBM32Boundary)
    bm = 32;
  else
    bm = (g32 || M <= kBM64Boundary) ? 64 : 128;
  bn = 128;
}

// 16-row tiles the GEMM of this M reads (grid m-blocks x BMF): the per-group
// scale buffer's tile stride. Rows and tiles past M are pad slots and must
// stay zero.
inline int w8a8_t16_tiles(const int M, const bool g32 = false) {
  int bm = 0, bn = 0;
  w8a8_cfg_tile(M, bm, bn, g32);
  return ((M + bm - 1) / bm) * (bm / 16);
}

// Split-K rule (ported from the prototype, tightened): double k_split until the
// grid has at least ~2*WGP CTAs (the device reports WGPs, 2 CUs each). Slices
// are whole groups, so k_split must divide K/128 - the prototype's rule could
// land on 16 or 32 for K=5120 (40 groups), which leaves a partial group per
// slice and drops K elements.
inline int w8a8_pick_ksplit(const int M, const int N, const int K,
                            const int target_ctas, const bool g32 = false) {
  int bm = 0, bn = 0;
  w8a8_cfg_tile(M, bm, bn, g32);
  const long long nctas = (long long)((N + bn - 1) / bn) * ((M + bm - 1) / bm);
  const int groups = K / kGroupK;
  int ks = 1;
  while (nctas * ks < target_ctas && ks * 2 <= groups && groups % (ks * 2) == 0)
    ks *= 2;
  return ks;
}

}  // namespace w8a8
// ======================= END w8a8 device code =======================

// ----------------------------- host wrapper -----------------------------
static bool on_gfx115x_w8a8() {
  static const bool result = [] {
    const std::string arch(at::cuda::getCurrentDeviceProperties()->gcnArchName);
    return arch.find("gfx115") != std::string::npos;
  }();
  return result;
}

// Launch the actq, GEMM and split-K reduce for one activation dtype; the
// template keeps every dispatch branch a compile-time instantiation of the
// device code.
template <typename scalar_t, bool W4, int WG>
static void w8a8_launch_all(const at::Tensor& in_act, const at::Tensor& out,
                            const float* swptr, signed char* aptr,
                            const unsigned char* wptr, float* sapgptr,
                            float* ppart, const int Mi, const int Ni,
                            const int Ki, const int k_split, const int t16,
                            const int b_stride, const int nstride,
                            const int gstride, const hipStream_t stream,
                            const int cfg) {
  const scalar_t* xptr = reinterpret_cast<const scalar_t*>(in_act.data_ptr());
  scalar_t* cptr = reinterpret_cast<scalar_t*>(out.data_ptr());
  w8a8::launch_actq<scalar_t>(xptr, aptr, sapgptr, Mi, Ki, t16, stream);
  w8a8::launch_gemm_sized<scalar_t, W4, WG>(aptr, wptr, sapgptr, swptr, cptr, ppart,
                                    Mi, Ni, Ki, k_split, t16, b_stride, nstride,
                                    gstride, stream, cfg);
  if (k_split > 1)
    w8a8::launch_reduce<scalar_t>(ppart, cptr, Mi, Ni, k_split, stream);
}

// W8A8 grouped skinny GEMM: int8 weights (one raw uint8b128 byte per element;
// the kernel flips the byte-center bit on the way into LDS, so the WMMA reads
// the true signed weight) with symmetric per-group (g=128) or per-channel
// scales, and int8 activations quantized per (row, K-group) inside the op. No
// weight-sized copy is prepared: the op reads the raw checkpoint bytes, and
// the only per-layer copy is the fp32 scale transpose.
// in_w8:    [N, K] uint8; rows may be padded past K (row stride >= K).
// in_act:   [M, K] fp16/bf16, contiguous.
// in_scale: fp32 group scales as [G, N] (G = K/128, group-major, the fast
//           layout: one coalesced load per group) or [N, G], or [N]
//           per-channel scales.
// returns:  [M, N] in the activation dtype.
static torch::Tensor wmma_gemm_impl(const at::Tensor& in_w8,
                                    const at::Tensor& in_act,
                                    const at::Tensor& in_scale, const bool w4,
                                    const int cfg = -1) {
  TORCH_CHECK(on_gfx115x_w8a8(),
              "w8a8_wmma_gemm targets gfx115x (wave32 iu8 WMMA) GPUs only");

  TORCH_CHECK(in_w8.dim() == 2, "Weights must be [N, K], got ", in_w8.dim(),
              " dims");
  const int64_t N = in_w8.size(0);
  const int64_t K = w4 ? 2 * in_w8.size(1) : in_w8.size(1);
  TORCH_CHECK(in_act.dim() == 2, "Activation must be [M, K], got ",
              in_act.dim(), " dims");
  const int64_t M = in_act.size(0);
  TORCH_CHECK(in_act.size(1) == K, "Activation K (", in_act.size(1),
              ") must match weight K (", K, ")");
  TORCH_CHECK(in_w8.scalar_type() == (w4 ? at::kChar : at::kByte),
              "Weights must be uint8 (uint8b128 layout, one byte per element),"
              " got ",
              in_w8.scalar_type());
  const auto act_dtype = in_act.scalar_type();
  TORCH_CHECK(act_dtype == at::kHalf || act_dtype == at::kBFloat16,
              "Activation must be float16 or bfloat16, got ", act_dtype);
  const auto scale_dtype = in_scale.scalar_type();
  TORCH_CHECK(scale_dtype == at::kFloat, "Scale must be float32, got ",
              scale_dtype);
  TORCH_CHECK(K % w8a8::kGroupK == 0,
              "K must be divisible by the weight group ", w8a8::kGroupK,
              ", got K=", K);
  TORCH_CHECK(in_act.is_contiguous(), "Activation must be contiguous");
  TORCH_CHECK(in_w8.stride(1) == 1, "Weights must be contiguous in dim 1");
  const int64_t B_row_stride = in_w8.stride(0);
  TORCH_CHECK(B_row_stride >= (w4 ? K / 2 : K), "Weight row stride (",
              B_row_stride, " bytes) is shorter than a row");
  const int64_t num_groups = K / w8a8::kGroupK;
  // [K/32, N] scales select the 32-element weight groups (int4 only).
  const bool g32 = w4 && in_scale.dim() == 2 && in_scale.size(0) == K / 32 &&
                   in_scale.size(1) == N && K / 32 != num_groups;

  // Grouped scales: [G, N] (group-major, coalesced loads) or [N, G] (checkpoint
  // layout); per-channel scales ([N]) are the degenerate single group with a
  // group stride of 0.
  int64_t sw_nstride = 1, sw_gstride = 0;
  if (in_scale.dim() == 1) {
    TORCH_CHECK(in_scale.numel() == N && in_scale.stride(0) == 1,
                "Per-channel scale must be [N=", N, "] contiguous, got shape ",
                in_scale.sizes());
  } else {
    TORCH_CHECK(in_scale.dim() == 2, "Scale must be [G, N], [N, G] or [N]");
    TORCH_CHECK(in_scale.stride(1) == 1, "Scale must be contiguous in dim 1");
    if (g32) {
      sw_nstride = 1;
      sw_gstride = in_scale.stride(0);
    } else if (in_scale.size(0) == num_groups && in_scale.size(1) == N) {
      sw_nstride = 1;
      sw_gstride = in_scale.stride(0);
    } else {
      TORCH_CHECK(in_scale.size(0) == N && in_scale.size(1) == num_groups,
                  "Scale must be [G, N] = [", num_groups, ", ", N,
                  "], [N, G] or [N=", N, "] but got [", in_scale.size(0), ", ",
                  in_scale.size(1), "]");
      sw_nstride = in_scale.stride(0);
      sw_gstride = 1;
    }
  }
  TORCH_CHECK(M <= INT_MAX && N <= INT_MAX && K <= INT_MAX &&
                  B_row_stride <= INT_MAX && sw_nstride * N <= INT_MAX &&
                  sw_gstride * num_groups <= INT_MAX,
              "Tensor dimensions overflow the kernel's int indexing");

  auto out = torch::empty({M, N}, in_act.options().dtype(in_act.scalar_type()));
  if (M == 0 || N == 0) return out;

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(in_act));
  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

  // The device reports WGPs (20 on gfx1151); the prototype's split-K target of
  // ~40 CTAs is 2 CTA slots per WGP.
  const int target_ctas =
      2 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int k_split =
      w8a8::w8a8_pick_ksplit((int)M, (int)N, (int)K, target_ctas, g32);

  const int t16 = w8a8::w8a8_t16_tiles((int)M, g32);
  auto q = torch::empty({M, K}, in_act.options().dtype(at::kChar));
  // Zero-filled so the rows past M (and the tiles past t16) hold a zero scale;
  // a pad row's A operand is zero, so 0 * 0 keeps the fold at zero.
  auto sa_pg =
      torch::zeros({num_groups, t16, 16}, in_act.options().dtype(at::kFloat));
  auto partials =
      (k_split > 1)
          ? torch::empty({k_split, M, N}, in_act.options().dtype(at::kFloat))
          : at::Tensor();

  signed char* aptr = reinterpret_cast<signed char*>(q.data_ptr());
  const unsigned char* wptr =
      reinterpret_cast<const unsigned char*>(in_w8.data_ptr());
  float* sapgptr = sa_pg.data_ptr<float>();
  const float* swptr = in_scale.data_ptr<float>();
  float* ppart = partials.defined() ? partials.data_ptr<float>() : nullptr;
  const int Mi = (int)M, Ni = (int)N, Ki = (int)K;
  const int b_stride = (int)B_row_stride;
  const int nstride = (int)sw_nstride, gstride = (int)sw_gstride;

#define STRIX_W8A8_LAUNCH(T, W4, WG)                                         \
  w8a8_launch_all<T, W4, WG>(in_act, out, swptr, aptr, wptr, sapgptr, ppart, Mi, \
                         Ni, Ki, k_split, t16, b_stride, nstride, gstride,   \
                         stream, cfg)
  if (act_dtype == at::kHalf) {
    if (g32) STRIX_W8A8_LAUNCH(half, true, 32);
    else if (w4) STRIX_W8A8_LAUNCH(half, true, 128);
    else STRIX_W8A8_LAUNCH(half, false, 128);
  } else {
    if (g32) STRIX_W8A8_LAUNCH(__hip_bfloat16, true, 32);
    else if (w4) STRIX_W8A8_LAUNCH(__hip_bfloat16, true, 128);
    else STRIX_W8A8_LAUNCH(__hip_bfloat16, false, 128);
  }
#undef STRIX_W8A8_LAUNCH
  return out;
}

torch::Tensor w8a8_wmma_gemm(const at::Tensor& in_w8, const at::Tensor& in_act,
                              const at::Tensor& in_scale) {
  return wmma_gemm_impl(in_w8, in_act, in_scale, false);
}

// Tile sweep entry point (bench/kernels/w8a8_gemm.py): same math and same
// activation quantize, one of the fixed (BMF, BNF, WAVES) configs in
// launch_gemm_sized. M must be a multiple of 128 so every config's 16-row tile
// stride matches the shared scale buffer.
torch::Tensor w8a8_wmma_gemm_cfg(const at::Tensor& in_w8,
                                 const at::Tensor& in_act,
                                 const at::Tensor& in_scale, int64_t cfg) {
  return wmma_gemm_impl(in_w8, in_act, in_scale, false, (int)cfg);
}

// W4A8 variant of the above for vLLM's skinny int4 layout (RDNAHybridW4A16:
// int8 [N, K/2] view of ExLlama-shuffled int32 words, symmetric uint4b8,
// rows may be padded): the same op with the weights unpacked on their way
// into LDS. Group scales: [K/128, N] / [N, K/128], [N], or [K/32, N] for
// 32-element groups.
torch::Tensor w4a8_wmma_gemm(const at::Tensor& in_w4, const at::Tensor& in_act,
                              const at::Tensor& in_scale) {
  return wmma_gemm_impl(in_w4, in_act, in_scale, true);
}

}  // namespace strix
