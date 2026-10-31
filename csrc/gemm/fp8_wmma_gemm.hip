// SPDX-License-Identifier: MIT
// FP8 W8A16 WMMA GEMM: C[M, N] = A[M, K] (bf16) x W[N, K]^T, W float8_e4m3fn
// with 128x128 block scales (fp32 [ceil(N/128), K/128]) as block-quantized
// checkpoints store them. For the batches between the skinny decode kernel
// (<= 5 rows per weight pass) and hipBLASLt on a dequantized copy (which
// moves five bytes per weight): the fp8 bytes are decoded to bf16 as they are
// staged into LDS (clone of mxfp4_wmma_gemm.hip, 128-wide scale blocks) and
// the main loop is the plain bf16 WMMA GEMM. gfx1151 has no fp8 WMMA.
//
// Decode per byte pair: the e4m3 byte as an fp16 holds value / 2^8 (shifts and
// masks, no convert; see w8a16_skinny_gemm.hip), widened to fp32 and scaled by
// block_scale * 256 before the bf16 round. e4m3 -> bf16 is exact, so the only
// rounding is the scale multiply.
//
// RAW variant (fp8_wmma_gemm_raw): the e4m3 bytes become fp16 (value / 2^8) with
// shifts and masks only, the activations arrive as fp16, the fp16 WMMA runs on the
// raw values and each K step is folded into the fp32 accumulator with its
// block scale. The weights stay exact and the per-element decode is ~3x
// cheaper (see moe_gemm_w8a16.hip, where this was measured on the experts).
//
// Per-channel scales (CH, compressed-tensors "channel" checkpoints): scales is
// fp32 [N] and is applied once per output in the epilogue, so the main loop
// runs on the exact decoded values (times 256 for the bf16 kernel's staging,
// plain value / 2^8 for RAW with the 2^8 restored in the epilogue). K only
// needs to be a multiple of 64 there (BK = 64 tiles when K % 128 != 0).
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kBlock = 128;  // weight elements per scale (both dims)
constexpr int kPad = 8;      // LDS row pitch BK + kPad
constexpr int kBK = 128;     // K granularity the op requires

__device__ __forceinline__ uint32_t fp8_pair_half_bits(uint32_t q) {
  return ((q & 0x7Fu) << 7) | ((q & 0x80u) << 8) | ((q & 0x7F00u) << 15) |
         ((q & 0x8000u) << 16);
}

// Four e4m3 bytes -> two bf16 pairs (k even in the low half), times `s`
// (block scale * 256).
__device__ __forceinline__ void fp8x4_to_bf16(uint32_t q, float s, uint32_t& o0,
                                              uint32_t& o1) {
  const uint32_t lo = fp8_pair_half_bits(q), hi = fp8_pair_half_bits(q >> 16);
  const half2 a = *reinterpret_cast<const half2*>(&lo);
  const half2 b = *reinterpret_cast<const half2*>(&hi);
  o0 = pack_bf16x2(__half2float(a.x) * s, __half2float(a.y) * s);
  o1 = pack_bf16x2(__half2float(b.x) * s, __half2float(b.y) * s);
}

template <int BM, int BN, int WAVES_M, int WAVES_N, int BK, bool STAGGER,
          bool RAW, bool CH>
__global__ __launch_bounds__(WAVES_M* WAVES_N* kWave) void fp8_gemm_kernel(
    uint16_t* __restrict__ c, const uint16_t* __restrict__ a,
    const uint8_t* __restrict__ w, const float* __restrict__ scales, int M,
    int N, int K, int64_t lda, int64_t ldw, int64_t lds_n, int64_t ldc) {
  constexpr int kPitch = BK + kPad;
  constexpr int kThreads = WAVES_M * WAVES_N * kWave;
  constexpr int TM = BM / WAVES_M, TN = BN / WAVES_N;
  constexpr int FM = TM / 16, FN = TN / 16;
  constexpr int kChunksA = BM * BK / 8;   // 16-byte bf16 chunks of A
  constexpr int kChunksB = BN * BK / 16;  // 16-byte fp8 chunks (16 k)
  constexpr int kLoadsA = kChunksA / kThreads, kLoadsB = kChunksB / kThreads;
  static_assert(kChunksA % kThreads == 0 && kChunksB % kThreads == 0);
  static_assert(BK % 16 == 0 && kBlock % BK == 0 || BK % kBlock == 0);

  __shared__ __attribute__((aligned(16))) uint16_t s_a[BM * kPitch];
  __shared__ __attribute__((aligned(16))) uint16_t s_b[BN * kPitch];

  // N tiles fastest: blocks running together share the A rows of one M tile.
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int wm = wave % WAVES_M, wn = wave / WAVES_M;

  uint4 ra[kLoadsA];
  uint4 rb[kLoadsB];
  float rs[kLoadsB];  // block scale * 256 of the staged weight chunks

  auto load = [&](int k0) {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads, r = ch / (BK / 8), c8 = (ch % (BK / 8)) * 8;
      uint4 v = {};
      if (m0 + r < M) v = *reinterpret_cast<const uint4*>(a + int64_t(m0 + r) * lda + k0 + c8);
      ra[i] = v;
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads, r = ch / (BK / 16);
      const int k16 = (ch % (BK / 16)) * 16;
      uint4 v = {};
      float s = 0.f;
      if (n0 + r < N) {
        v = *reinterpret_cast<const uint4*>(w + int64_t(n0 + r) * ldw + k0 + k16);
        if constexpr (CH)
          s = 256.0f;
        else if constexpr (!RAW)
          s = scales[int64_t((n0 + r) / kBlock) * lds_n + (k0 + k16) / kBlock] * 256.0f;
      }
      rb[i] = v;
      rs[i] = s;
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads, r = ch / (BK / 8), c8 = (ch % (BK / 8)) * 8;
      *reinterpret_cast<uint4*>(s_a + r * kPitch + c8) = ra[i];
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads, r = ch / (BK / 16);
      const int k16 = (ch % (BK / 16)) * 16;
      if constexpr (RAW) {
        uint32_t* dstr = reinterpret_cast<uint32_t*>(s_b + r * kPitch + k16);
        *reinterpret_cast<uint4*>(dstr) = make_uint4(
            fp8_pair_half_bits(rb[i].x), fp8_pair_half_bits(rb[i].x >> 16),
            fp8_pair_half_bits(rb[i].y), fp8_pair_half_bits(rb[i].y >> 16));
        *reinterpret_cast<uint4*>(dstr + 4) = make_uint4(
            fp8_pair_half_bits(rb[i].z), fp8_pair_half_bits(rb[i].z >> 16),
            fp8_pair_half_bits(rb[i].w), fp8_pair_half_bits(rb[i].w >> 16));
        continue;
      }
      uint32_t o[8];
      fp8x4_to_bf16(rb[i].x, rs[i], o[0], o[1]);
      fp8x4_to_bf16(rb[i].y, rs[i], o[2], o[3]);
      fp8x4_to_bf16(rb[i].z, rs[i], o[4], o[5]);
      fp8x4_to_bf16(rb[i].w, rs[i], o[6], o[7]);
      // 16 values = 2 uint4 stores; the row pitch keeps them 16-byte aligned.
      uint32_t* dst = reinterpret_cast<uint32_t*>(s_b + r * kPitch + k16);
      *reinterpret_cast<uint4*>(dst) = make_uint4(o[0], o[1], o[2], o[3]);
      *reinterpret_cast<uint4*>(dst + 4) = make_uint4(o[4], o[5], o[6], o[7]);
    }
  };

  f32x8 acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) acc[i][j] = f32x8{};
  // RAW: one K step accumulates in `part`, then folds into `acc` with its scale.
  constexpr bool FOLD = RAW && !CH;
  f32x8 part[FOLD ? FM : 1][FOLD ? FN : 1];
  // This wave's TN rows lie in one 128-row scale block; clamp for waves past N.
  const float* sblk = nullptr;
  if constexpr (FOLD) {
    const int row0 = n0 + wn * TN;
    const int blk = min(row0 / kBlock, (N + kBlock - 1) / kBlock - 1);
    sblk = scales + int64_t(blk) * lds_n;
  }

  // STAGGER starts every CTA at a different K chunk and wraps around: the sum
  // over K does not care about the order, and rows ldw bytes apart that is a
  // power of two (K = 8192 -> 4096 B, K = 16384 -> 8192 B) otherwise camp on
  // the same DRAM channel for the whole loop - measured 21 GB/s at x1 against
  // 138 GB/s for the non-aliasing K = 5376 (bench/kernels/mxfp4_gemm.py).
  const int nch = K / BK;
  const int stagger = STAGGER ? (blockIdx.x * 7 + blockIdx.y * 3) % nch : 0;
  int k0 = stagger * BK;
  int knext = k0 + BK < K ? k0 + BK : 0;
  load(k0);
  store();
  __syncthreads();
  for (int done = 0; done < nch; ++done) {
    const bool more = done + 1 < nch;
    if (more) load(knext);
    if constexpr (FOLD) {
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) part[i][j] = f32x8{};
    }
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      bf16x16 fb[FN];
#pragma unroll
      for (int j = 0; j < FN; ++j)
        fb[j] = load_frag_bf16(s_b + (wn * TN + j * 16) * kPitch + kk, kPitch, lane);
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        const bf16x16 fa = load_frag_bf16(s_a + (wm * TM + i * 16) * kPitch + kk, kPitch, lane);
#pragma unroll
        for (int j = 0; j < FN; ++j) {
          if constexpr (FOLD)
            part[i][j] = wmma_f16(fa, fb[j], part[i][j]);
          else if constexpr (RAW)
            acc[i][j] = wmma_f16(fa, fb[j], acc[i][j]);
          else
            acc[i][j] = wmma_bf16(fa, fb[j], acc[i][j]);
        }
      }
    }
    if constexpr (FOLD) {
      // k0 is this step's K offset, whatever the stagger made it.
      const float sc = sblk[k0 / kBlock] * 256.0f;
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][j][e] += part[i][j][e] * sc;
    }
    __syncthreads();
    if (more) store();
    __syncthreads();
    k0 = knext;
    knext = k0 + BK < K ? k0 + BK : 0;
  }

  // acc[i][j][e] = C[m wm*TM + i*16 + acc_row(e)][n wn*TN + j*16 + col]
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = m0 + wm * TM + i * 16 + acc_row(e, lane);
      if (m >= M) continue;
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        const int n = n0 + wn * TN + j * 16 + acc_col(lane);
        if (n >= N) continue;
        float v = acc[i][j][e];
        if constexpr (CH) v *= scales[n] * (RAW ? 256.0f : 1.0f);
        c[int64_t(m) * ldc + n] = static_cast<uint16_t>(pack_bf16x2(v, 0.f) & 0xffffu);
      }
    }
}

}  // namespace

template <int BM, int BN, int WAVES_M, int WAVES_N, int BK, bool STAGGER,
          bool RAW, bool CH>
static void fp8_launch(const at::Tensor& x, const at::Tensor& w,
                       const at::Tensor& scales, at::Tensor& c) {
  const int m = (int)x.size(0), n = (int)w.size(0), k = (int)x.size(1);
  fp8_gemm_kernel<BM, BN, WAVES_M, WAVES_N, BK, STAGGER, RAW, CH>
      <<<dim3((unsigned)((n + BN - 1) / BN), (unsigned)((m + BM - 1) / BM)),
         WAVES_M * WAVES_N * kWave, 0, at::hip::getCurrentHIPStream()>>>(
          reinterpret_cast<uint16_t*>(c.data_ptr()),
          reinterpret_cast<const uint16_t*>(x.data_ptr()),
          reinterpret_cast<const uint8_t*>(w.data_ptr()),
          scales.data_ptr<float>(), m, n, k, x.stride(0), w.stride(0),
          CH ? 0 : scales.stride(0), c.stride(0));
}

// x: [M, K] bf16 (contiguous rows); w: float8_e4m3fn [N, K] (rows may be
// padded, 16-byte aligned); scales: fp32 [ceil(N/128), K/128], or [N] per
// output channel. Returns [M, N] bf16. cfg < 0 picks the tile by M; the fixed
// ids are for the bench.
static torch::Tensor fp8_gemm_impl(const torch::Tensor& w, const torch::Tensor& x,
                                   const torch::Tensor& scales, int64_t cfg,
                                   bool raw) {
  TORCH_CHECK(x.scalar_type() == (raw ? at::kHalf : at::kBFloat16) && x.dim() == 2 &&
                  x.stride(1) == 1 && x.stride(0) % 8 == 0,
              "x must be ", raw ? "fp16" : "bf16", " [M, K] with 16-byte aligned rows");
  TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn && w.dim() == 2 &&
                  w.stride(1) == 1 && w.stride(0) % 16 == 0,
              "w must be float8_e4m3fn [N, K] with 16-byte aligned rows");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(w.size(1) == k, "K mismatch: w is [N, ", w.size(1), "], x is [M, ", k, "]");
  const bool ch = scales.dim() == 1;
  TORCH_CHECK(scales.scalar_type() == at::kFloat, "scales must be float32");
  if (ch) {
    TORCH_CHECK(scales.size(0) == n && scales.stride(0) == 1,
                "per-channel scale must be contiguous float32 [N]");
    TORCH_CHECK(k % 64 == 0, "K must be a multiple of 64 (got ", k, ")");
  } else {
    TORCH_CHECK(k % kBK == 0, "K must be a multiple of ", kBK, " (got ", k, ")");
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == (n + kBlock - 1) / kBlock &&
                    scales.size(1) == k / kBlock && scales.stride(1) == 1,
                "scale must be float32 [ceil(N/128), K/128] or [N]");
  }
  const bool bk64 = k % 128 != 0;
  auto c = torch::empty({m, n}, x.options().dtype(at::kBFloat16));
  if (m == 0 || n == 0) return c;

#define STRIX_F8_CH(BM, BN, WM, WN, BK)                                       \
  do {                                                                        \
    if (raw)                                                                  \
      fp8_launch<BM, BN, WM, WN, BK, true, true, true>(x, w, scales, c);      \
    else                                                                      \
      fp8_launch<BM, BN, WM, WN, BK, true, false, true>(x, w, scales, c);     \
  } while (0)
#define STRIX_F8(BM, BN, WM, WN, BK)                                          \
  do {                                                                        \
    if (ch && bk64)                                                           \
      STRIX_F8_CH(BM, BN, WM, WN, 64);                                        \
    else if (ch)                                                              \
      STRIX_F8_CH(BM, BN, WM, WN, BK);                                        \
    else if (raw)                                                             \
      fp8_launch<BM, BN, WM, WN, BK, true, true, false>(x, w, scales, c);     \
    else                                                                      \
      fp8_launch<BM, BN, WM, WN, BK, true, false, false>(x, w, scales, c);    \
  } while (0)
  switch (cfg) {
    case 0: STRIX_F8(16, 32, 1, 2, 128); return c;
    case 1: STRIX_F8(16, 64, 1, 4, 128); return c;
    case 2: STRIX_F8(32, 64, 1, 4, 128); return c;
    case 3: STRIX_F8(32, 128, 2, 4, 128); return c;
    case 4: STRIX_F8(64, 64, 1, 4, 128); return c;
    case 5: STRIX_F8(64, 128, 2, 4, 128); return c;
    case 6: STRIX_F8(128, 128, 2, 4, 64); return c;
    case 7: STRIX_F8(128, 128, 4, 2, 64); return c;
    case 8: STRIX_F8(128, 256, 4, 4, 64); return c;
    case 9: STRIX_F8(16, 128, 1, 4, 128); return c;
    default: break;
  }
  // The smallest tile that covers M in one m-block: a second m-block re-reads
  // the whole weight matrix; the wider tile from M 129 won 1.0-1.4x at M 256.
  // Tuned with bench/kernels/fp8_gemm.py.
  if (m <= 16) STRIX_F8(16, 32, 1, 2, 128);
  else if (m <= 32) STRIX_F8(32, 128, 2, 4, 128);
  else if (m <= 64) STRIX_F8(64, 128, 2, 4, 128);
  else if (m <= 128) STRIX_F8(128, 128, 2, 4, 64);
  else STRIX_F8(128, 256, 4, 4, 64);
#undef STRIX_F8
#undef STRIX_F8_CH
  return c;
}

torch::Tensor fp8_wmma_gemm(const torch::Tensor& w, const torch::Tensor& x,
                            const torch::Tensor& scales) {
  return fp8_gemm_impl(w, x, scales, -1, false);
}

// Bench-only: one of the fixed tile configs instead of the tile-by-M choice.
torch::Tensor fp8_wmma_gemm_cfg(const torch::Tensor& w, const torch::Tensor& x,
                                const torch::Tensor& scales, int64_t cfg) {
  return fp8_gemm_impl(w, x, scales, cfg, false);
}

// fp16 activations, raw decode (see the header); the output is bf16.
torch::Tensor fp8_wmma_gemm_raw(const torch::Tensor& w, const torch::Tensor& x,
                                const torch::Tensor& scales) {
  return fp8_gemm_impl(w, x, scales, -1, true);
}

torch::Tensor fp8_wmma_gemm_raw_cfg(const torch::Tensor& w, const torch::Tensor& x,
                                    const torch::Tensor& scales, int64_t cfg) {
  return fp8_gemm_impl(w, x, scales, cfg, true);
}

}  // namespace strix
