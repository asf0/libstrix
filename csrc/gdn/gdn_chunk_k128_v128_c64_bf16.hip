// SPDX-License-Identifier: MIT
// Gated delta rule, chunked prefill (Qwen3.x GDN layers): head_k = head_v = 128,
// chunk 64, bf16 q/k/v, fp32 gates and state. Varlen over cu_seqlens.
//
// Per value head h (key head h / (Hv / Hk)), chunk of rows t with in-chunk
// cumulative log gate gc, gl = gc[last]:
//   A   = (I + strict_lower(beta_i k_i.k_j e^(gc_i - gc_j)))^-1
//   P   = scale * tril(q_i.k_j e^(gc_i - gc_j))
//   Y   = beta * V - (beta e^gc) * (K S)        S: state [K, V]
//   Vn  = A Y
//   O   = scale e^gc * (Q S) + P Vn
//   S  <- e^gl S + K^T (e^(gl - gc) * Vn)
// Kernel 1 computes A and P for every (chunk, head) in parallel. Kernel 2 walks
// the chunks of one sequence for one head and a slice of V, holding S in
// registers.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kC = 64;      // chunk
constexpr int kD = 128;     // head_k = head_v
// V columns per kernel-2 block: 64 is 8-15% faster than 32 (fewer K/Q/A/P
// re-reads); 128 does not fit in LDS.
constexpr int kBV = 64;
constexpr int kThreads = 256;
constexpr int kPD = kD + 8;  // LDS pitch for [*, 128] tiles
constexpr int kPC = kC + 8;  // LDS pitch for [*, 64] tiles

__device__ __forceinline__ uint16_t to_bf16(float x) {
  return static_cast<uint16_t>(pack_bf16x2(x, 0.f) & 0xffffu);
}
__device__ __forceinline__ float from_bf16(uint16_t x) {
  return __uint_as_float(uint32_t(x) << 16);
}

// One recurrent-state element, fp32 or bf16 (vLLM's --mamba-ssm-cache-dtype).
// The chunk kernel touches every state element once per prefill chunk, so
// accepting the state's dtype directly avoids a full-size fp32 copy per
// sequence per chunk - what the wrapper used to do when the flag was set.
template <typename StateT>
__device__ __forceinline__ float load_state1(const StateT* p, int i);
template <>
__device__ __forceinline__ float load_state1<float>(const float* p, int i) {
  return p[i];
}
template <>
__device__ __forceinline__ float load_state1<uint16_t>(const uint16_t* p, int i) {
  return from_bf16(p[i]);
}
template <typename StateT>
__device__ __forceinline__ void store_state1(StateT* p, int i, float v);
template <>
__device__ __forceinline__ void store_state1<float>(float* p, int i, float v) {
  p[i] = v;
}
template <>
__device__ __forceinline__ void store_state1<uint16_t>(uint16_t* p, int i,
                                                       float v) {
  p[i] = to_bf16(v);
}

struct ChunkPos {
  int seq, t0, rows, chunk;  // chunk = global chunk index
};

__device__ __forceinline__ ChunkPos locate_chunk(const int32_t* cu,
                                                 const int32_t* chunk_off,
                                                 int nseq, int chunk) {
  // Offsets are monotonic, so the owning sequence is the count of later
  // offsets <= chunk. (An early-exit while loop here was miscompiled by the
  // ROCm 10.1 clang: t0 and rows came out of different values of n.)
  int n = 0;
  for (int i = 1; i < nseq; ++i) n += chunk_off[i] <= chunk;
  const int t0 = cu[n] + (chunk - chunk_off[n]) * kC;
  return {n, t0, min(kC, cu[n + 1] - t0), chunk};
}

// Copies `rows` rows of 128 bf16 (row stride `ld`) into an LDS [64][kPD]
// tile, zero-filling the rest.
__device__ __forceinline__ void load_rows_128(uint16_t* dst,
                                              const uint16_t* src, int64_t ld,
                                              int rows, int tid) {
  for (int ch = tid; ch < kC * (kD / 8); ch += kThreads) {
    const int r = ch / (kD / 8), c = (ch % (kD / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < rows) v = *reinterpret_cast<const uint4*>(src + r * ld + c);
    *reinterpret_cast<uint4*>(dst + r * kPD + c) = v;
  }
}

// In-chunk cumulative gate and per-row factors, for threads 0..63.
__device__ __forceinline__ void load_gates(float* s_gc, float* s_beta,
                                           const float* g, const float* beta,
                                           int64_t t0, int rows, int hv, int h,
                                           int tid) {
  if (tid < kC) {
    s_gc[tid] = tid < rows ? g[(t0 + tid) * hv + h] : 0.f;
    s_beta[tid] = tid < rows ? beta[(t0 + tid) * hv + h] : 0.f;
  }
  barrier();
  float acc = 0.f;
  if (tid < kC)
    for (int j = 0; j <= tid; ++j) acc += s_gc[j];
  barrier();
  if (tid < kC) s_gc[tid] = acc;
  barrier();
}

// ---------------------------------------------------------------- kernel 1

__global__ __launch_bounds__(kThreads) void gdn_chunk_prep_kernel(
    uint16_t* __restrict__ a_out, uint16_t* __restrict__ p_out,
    float* __restrict__ gc_out,
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const float* __restrict__ g, const float* __restrict__ beta,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ chunk_off,
    int nseq, int hv, int hk, int64_t ldq_t, int64_t ldq_h, int64_t ldk_t,
    int64_t ldk_h, float scale) {
  __shared__ __attribute__((aligned(16))) uint16_t s_k[kC * kPD];
  __shared__ __attribute__((aligned(16))) uint16_t s_q[kC * kPD];
  __shared__ float s_m[kC][kC + 1];
  __shared__ float s_gc[kC], s_beta[kC];

  // Heads are the fastest grid index: concurrently running blocks then read
  // the same rows at different head offsets. With chunks fastest, every
  // block's rows (4-16 KiB strided) hit the same DRAM channels (8x slower).
  // The grid is an upper bound on the chunk count (no host sync to size it).
  const int chunk = blockIdx.y;
  if (chunk >= chunk_off[nseq]) return;
  const int h = blockIdx.x, tid = threadIdx.x, lane = tid % kWave;
  const int wave = tid / kWave;
  const ChunkPos cp = locate_chunk(cu, chunk_off, nseq, chunk);
  const int kh = h / (hv / hk);

  load_gates(s_gc, s_beta, g, beta, cp.t0, cp.rows, hv, h, tid);
  // Padded rows carry the last valid cumulative gate, so gc[63] = gc_last.
  if (tid < kC) gc_out[(size_t(cp.chunk) * hv + h) * kC + tid] = s_gc[tid];
  load_rows_128(s_k, k + cp.t0 * ldk_t + kh * ldk_h, ldk_t, cp.rows, tid);
  load_rows_128(s_q, q + cp.t0 * ldq_t + kh * ldq_h, ldq_t, cp.rows, tid);
  barrier();

  // 16 tiles of 16x16 over [64 i][64 j]; wave w owns row tile w/2 and column
  // tiles 2*(w%2), 2*(w%2)+1.
  const int ti = wave / 2;
  const size_t base = (size_t(cp.chunk) * hv + h) * kC * kC;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int tj = 2 * (wave % 2) + jj;
    f32x8 kk_acc = {}, qk_acc = {};
    if (tj <= ti) {
#pragma unroll
      for (int d = 0; d < kD; d += 16) {
        const bf16x16 kb = load_frag_bf16(s_k + tj * 16 * kPD + d, kPD, lane);
        kk_acc = wmma_bf16(load_frag_bf16(s_k + ti * 16 * kPD + d, kPD, lane),
                           kb, kk_acc);
        qk_acc = wmma_bf16(load_frag_bf16(s_q + ti * 16 * kPD + d, kPD, lane),
                           kb, qk_acc);
      }
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int i = ti * 16 + acc_row(e, lane), j = tj * 16 + acc_col(lane);
      const float decay = i >= j ? __expf(s_gc[i] - s_gc[j]) : 0.f;
      s_m[i][j] = i > j ? s_beta[i] * kk_acc[e] * decay : 0.f;
      p_out[base + i * kC + j] = to_bf16(i >= j ? scale * qk_acc[e] * decay : 0.f);
    }
  }
  barrier();

  // X = (I + M)^-1 by 16x16 blocks, in fp32 in the space K and Q used:
  // diagonal blocks by forward substitution, then for each block distance d
  // X_ij = -X_ii sum_{k=j..i-1} M_ik X_kj, all 256 threads per level.
  float(*s_x)[kC + 1] = reinterpret_cast<float(*)[kC + 1]>(s_k);
  float(*s_t)[kC + 1] = reinterpret_cast<float(*)[kC + 1]>(s_q);
  static_assert(sizeof(float) * kC * (kC + 1) <= sizeof(uint16_t) * kC * kPD);
  for (int idx = tid; idx < kC * kC; idx += kThreads)
    s_x[idx / kC][idx % kC] = 0.f;
  barrier();
  if (tid < kC) {
    const int blk = tid / 16, c = tid % 16, o = blk * 16;
    float x[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      float acc = r == c ? 1.f : 0.f;
#pragma unroll
      for (int j = 0; j < r; ++j) acc -= s_m[o + r][o + j] * x[j];
      x[r] = r < c ? 0.f : acc;
      s_x[o + r][o + c] = x[r];
    }
  }
  barrier();
  for (int d = 1; d < kC / 16; ++d) {
    const int nblk = kC / 16 - d;  // blocks (i, i - d)
    for (int idx = tid; idx < nblk * 256; idx += kThreads) {
      const int j = idx / 256, i = j + d, r = (idx % 256) / 16, c = idx % 16;
      float acc = 0.f;
      for (int kk = j * 16; kk < i * 16; ++kk)
        acc += s_m[i * 16 + r][kk] * s_x[kk][j * 16 + c];
      s_t[i * 16 + r][j * 16 + c] = acc;
    }
    barrier();
    for (int idx = tid; idx < nblk * 256; idx += kThreads) {
      const int j = idx / 256, i = j + d, r = (idx % 256) / 16, c = idx % 16;
      float acc = 0.f;
#pragma unroll
      for (int kk = 0; kk < 16; ++kk)
        acc -= s_x[i * 16 + r][i * 16 + kk] * s_t[i * 16 + kk][j * 16 + c];
      s_x[i * 16 + r][j * 16 + c] = acc;
    }
    barrier();
  }
  for (int idx = tid; idx < kC * kC / 2; idx += kThreads) {
    const int i = idx / (kC / 2), j = (idx % (kC / 2)) * 2;
    *reinterpret_cast<uint32_t*>(a_out + base + i * kC + j) =
        pack_bf16x2(s_x[i][j], s_x[i][j + 1]);
  }
}

// ---------------------------------------------------------------- kernel 2

// A fragment of K^T scaled per t: element e of lane holds
// K[k0 + e][d0 + lane % 16] * f[k0 + e], read from row-major [t][kPD] LDS.
__device__ __forceinline__ bf16x16 load_frag_kt_scaled(const uint16_t* s_k,
                                                       const float* f, int d0,
                                                       int k0, int lane) {
  bf16x16 frag;
  const int d = d0 + lane % 16;
#pragma unroll
  for (int e = 0; e < 16; ++e)
    frag[e] = static_cast<short>(
        to_bf16(from_bf16(s_k[(k0 + e) * kPD + d]) * f[k0 + e]));
  return frag;
}

// LDS: s_k lives for the whole chunk; the pool holds Q and S^T until Y is
// formed, then P, A and Vn^T. V goes from global straight into registers in
// accumulator layout, so a 64-wide V slice still fits in 64 KB.
template <int BV, typename StateT>
__global__ __launch_bounds__(kThreads) void gdn_chunk_scan_kernel(
    uint16_t* __restrict__ o, StateT* __restrict__ final_state,
    const StateT* __restrict__ init_state, const uint16_t* __restrict__ a_in,
    const uint16_t* __restrict__ p_in, const float* __restrict__ gc_in,
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v, const float* __restrict__ beta,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ chunk_off,
    int hv, int hk, int64_t ldq_t, int64_t ldq_h, int64_t ldk_t, int64_t ldk_h,
    int64_t ldv_t, int64_t ldv_h, int64_t ldo_t, int64_t ldo_h, float scale) {
  constexpr int kNT = BV / 32;  // [16 t][16 v] output tiles per wave
  constexpr int kNS = BV / 16;  // [16 d][16 v] state tiles per wave
  constexpr int kPool1 = (kC + BV) * kPD, kPool2 = (2 * kC + BV) * kPC;
  __shared__ __attribute__((aligned(16))) uint16_t s_k[kC * kPD];
  __shared__ __attribute__((aligned(16)))
  uint16_t s_pool[kPool1 > kPool2 ? kPool1 : kPool2];
  __shared__ __attribute__((aligned(16))) uint16_t s_yt[BV * kPC];  // Y^T
  __shared__ float s_gc[kC], s_beta[kC], s_egl[kC];
  uint16_t* s_q = s_pool;                   // [64][kPD]
  uint16_t* s_st = s_pool + kC * kPD;       // [BV][kPD], S^T
  uint16_t* s_p = s_pool;                   // [64][kPC]
  uint16_t* s_a = s_pool + kC * kPC;        // [64][kPC]
  uint16_t* s_vnt = s_pool + 2 * kC * kPC;  // [BV][kPC]

  // V slice fastest, then head: the slices of a head (and the blocks of a
  // key head) run together and share K/Q/A/P through L2.
  const int vbase = blockIdx.x * BV, h = blockIdx.y, n = blockIdx.z;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int kh = h / (hv / hk);
  const int tt = wave / 2, tv0 = (wave % 2) * kNT;  // output tiles [tt][tv0+j]
  const int dt = wave;                              // state tiles [dt][j]

  const size_t state_off = (size_t(n) * hv + h) * kD * kD;
  f32x8 s_acc[kNS];
#pragma unroll
  for (int j = 0; j < kNS; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int d = dt * 16 + acc_row(e, lane), vv = vbase + j * 16 + acc_col(lane);
      s_acc[j][e] = load_state1(init_state + state_off + vv * kD, d);
    }

  const int seq_t0 = cu[n], seq_len = cu[n + 1] - seq_t0;
  const int nchunks = (seq_len + kC - 1) / kC;

  // Next-chunk staging registers (flat vectors: uint4 arrays went to scratch).
  typedef uint32_t u32x16 __attribute__((ext_vector_type(16)));
  typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));
  typedef uint16_t u16x16 __attribute__((ext_vector_type(16)));
  u32x16 rk = {}, rq = {};
  u32x8 ra = {}, rp = {};
  u16x16 rv = {};
  float rg = 0.f, rb = 0.f;

#define STRIX_PUT4(VEC, I, X) \
  VEC[4 * (I)] = (X).x, VEC[4 * (I) + 1] = (X).y, VEC[4 * (I) + 2] = (X).z, \
  VEC[4 * (I) + 3] = (X).w
#define STRIX_GET4(VEC, I) \
  make_uint4(VEC[4 * (I)], VEC[4 * (I) + 1], VEC[4 * (I) + 2], VEC[4 * (I) + 3])
#define STRIX_PREFETCH_KQ(C)                                                 \
  do {                                                                       \
    const int pt0 = seq_t0 + (C) * kC, prows = min(kC, seq_len - (C) * kC);  \
    _Pragma("unroll") for (int i = 0; i < 4; ++i) {                          \
      const int ch = tid + i * kThreads, r = ch / (kD / 8), c8 = (ch % (kD / 8)) * 8; \
      uint4 xk = {}, xq = {};                                                \
      if (r < prows) {                                                       \
        xk = *reinterpret_cast<const uint4*>(k + (pt0 + r) * ldk_t + kh * ldk_h + c8); \
        xq = *reinterpret_cast<const uint4*>(q + (pt0 + r) * ldq_t + kh * ldq_h + c8); \
      }                                                                      \
      STRIX_PUT4(rk, i, xk);                                                 \
      STRIX_PUT4(rq, i, xq);                                                 \
    }                                                                        \
    if (tid < kC) {                                                          \
      rg = gc_in[(size_t(chunk_off[n] + (C)) * hv + h) * kC + tid];          \
      rb = tid < prows ? beta[(pt0 + tid) * hv + h] : 0.f;                   \
    }                                                                        \
  } while (0)
#define STRIX_PREFETCH_PAV(C)                                                \
  do {                                                                       \
    const int pt0 = seq_t0 + (C) * kC, prows = min(kC, seq_len - (C) * kC);  \
    const size_t pap = (size_t(chunk_off[n] + (C)) * hv + h) * kC * kC;      \
    _Pragma("unroll") for (int i = 0; i < 2; ++i) {                          \
      const int ch = tid + i * kThreads;                                     \
      const uint4 xa = *reinterpret_cast<const uint4*>(a_in + pap + ch * 8); \
      const uint4 xp = *reinterpret_cast<const uint4*>(p_in + pap + ch * 8); \
      STRIX_PUT4(ra, i, xa);                                                 \
      STRIX_PUT4(rp, i, xp);                                                 \
    }                                                                        \
    _Pragma("unroll") for (int j = 0; j < kNT; ++j)                          \
      _Pragma("unroll") for (int e = 0; e < 8; ++e) {                        \
        const int t = tt * 16 + acc_row(e, lane);                            \
        rv[j * 8 + e] = t < prows ? v[(pt0 + t) * ldv_t + h * ldv_h + vbase + \
                                      (tv0 + j) * 16 + acc_col(lane)]        \
                                  : uint16_t(0);                             \
      }                                                                      \
  } while (0)

  if (nchunks > 0) {
    STRIX_PREFETCH_KQ(0);
    STRIX_PREFETCH_PAV(0);
  }
  for (int c = 0; c < nchunks; ++c) {
    const int t0 = seq_t0 + c * kC, rows = min(kC, seq_len - c * kC);
    const bool more = c + 1 < nchunks;
    barrier();  // the previous chunk is done with every LDS tile
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int ch = tid + i * kThreads, r = ch / (kD / 8), c8 = (ch % (kD / 8)) * 8;
      *reinterpret_cast<uint4*>(s_k + r * kPD + c8) = STRIX_GET4(rk, i);
      *reinterpret_cast<uint4*>(s_q + r * kPD + c8) = STRIX_GET4(rq, i);
    }
    if (tid < kC) {
      s_gc[tid] = rg;
      s_beta[tid] = rb;
    }
#pragma unroll
    for (int j = 0; j < kNS; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        s_st[(j * 16 + acc_col(lane)) * kPD + dt * 16 + acc_row(e, lane)] =
            to_bf16(s_acc[j][e]);
    barrier();
    if (tid < kC) s_egl[tid] = __expf(s_gc[kC - 1] - s_gc[tid]);
    if (more) STRIX_PREFETCH_KQ(c + 1);

    // X = K S and QS = Q S for this wave's [16 t][16 v] tiles.
    f32x8 x_acc[kNT] = {}, qs_acc[kNT] = {};
#pragma unroll
    for (int d = 0; d < kD; d += 16) {
      const bf16x16 ka = load_frag_bf16(s_k + tt * 16 * kPD + d, kPD, lane);
      const bf16x16 qa = load_frag_bf16(s_q + tt * 16 * kPD + d, kPD, lane);
#pragma unroll
      for (int j = 0; j < kNT; ++j) {
        const bf16x16 sb = load_frag_bf16(s_st + (tv0 + j) * 16 * kPD + d, kPD, lane);
        x_acc[j] = wmma_bf16(ka, sb, x_acc[j]);
        qs_acc[j] = wmma_bf16(qa, sb, qs_acc[j]);
      }
    }
    // Y = beta V - beta e^gc X, stored transposed.
#pragma unroll
    for (int j = 0; j < kNT; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int t = tt * 16 + acc_row(e, lane), vv = (tv0 + j) * 16 + acc_col(lane);
        const float bt = s_beta[t], eg = __expf(s_gc[t]);
        s_yt[vv * kPC + t] = to_bf16(bt * from_bf16(rv[j * 8 + e]) - bt * eg * x_acc[j][e]);
        qs_acc[j][e] *= scale * eg;
      }
    barrier();  // Y^T complete; the pool is free

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int ch = tid + i * kThreads, r = ch / (kC / 8), c8 = (ch % (kC / 8)) * 8;
      *reinterpret_cast<uint4*>(s_p + r * kPC + c8) = STRIX_GET4(rp, i);
      *reinterpret_cast<uint4*>(s_a + r * kPC + c8) = STRIX_GET4(ra, i);
    }
    barrier();  // A, P complete
    if (more) STRIX_PREFETCH_PAV(c + 1);
    // Vn = A Y
    f32x8 vn_acc[kNT] = {};
#pragma unroll
    for (int jj = 0; jj < kC; jj += 16) {
      const bf16x16 aa = load_frag_bf16(s_a + tt * 16 * kPC + jj, kPC, lane);
#pragma unroll
      for (int j = 0; j < kNT; ++j)
        vn_acc[j] = wmma_bf16(
            aa, load_frag_bf16(s_yt + (tv0 + j) * 16 * kPC + jj, kPC, lane), vn_acc[j]);
    }
#pragma unroll
    for (int j = 0; j < kNT; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        s_vnt[((tv0 + j) * 16 + acc_col(lane)) * kPC + tt * 16 + acc_row(e, lane)] =
            to_bf16(vn_acc[j][e]);
    barrier();  // Vn^T complete

    // O = scale e^gc Q S + P Vn
#pragma unroll
    for (int jj = 0; jj < kC; jj += 16) {
      const bf16x16 pa = load_frag_bf16(s_p + tt * 16 * kPC + jj, kPC, lane);
#pragma unroll
      for (int j = 0; j < kNT; ++j)
        qs_acc[j] = wmma_bf16(
            pa, load_frag_bf16(s_vnt + (tv0 + j) * 16 * kPC + jj, kPC, lane), qs_acc[j]);
    }
#pragma unroll
    for (int j = 0; j < kNT; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int t = tt * 16 + acc_row(e, lane);
        if (t < rows)
          o[(t0 + t) * ldo_t + h * ldo_h + vbase + (tv0 + j) * 16 + acc_col(lane)] =
              to_bf16(qs_acc[j][e]);
      }

    // S = e^gl S + K^T diag(e^(gl - gc)) Vn
    const float egl_chunk = __expf(s_gc[kC - 1]);
#pragma unroll
    for (int j = 0; j < kNS; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) s_acc[j][e] *= egl_chunk;
#pragma unroll
    for (int t = 0; t < kC; t += 16) {
      const bf16x16 kt = load_frag_kt_scaled(s_k, s_egl, dt * 16, t, lane);
#pragma unroll
      for (int j = 0; j < kNS; ++j)
        s_acc[j] = wmma_bf16(kt, load_frag_bf16(s_vnt + j * 16 * kPC + t, kPC, lane),
                             s_acc[j]);
    }
  }
#undef STRIX_PUT4
#undef STRIX_GET4
#undef STRIX_PREFETCH_KQ
#undef STRIX_PREFETCH_PAV

  if (final_state) {
#pragma unroll
    for (int j = 0; j < kNS; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int d = dt * 16 + acc_row(e, lane), vv = vbase + j * 16 + acc_col(lane);
        store_state1(final_state + state_off + vv * kD, d, s_acc[j][e]);
      }
  }
}

void check_qkv(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.dim() == 3 && t.size(2) == kD && t.stride(2) == 1, name,
              " must be [T, H, 128] with a contiguous last dim");
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
  TORCH_CHECK(t.stride(0) % 8 == 0 && t.stride(1) % 8 == 0, name,
              " needs 16-byte aligned rows");
}

}  // namespace

// q, k: [T, Hk, 128]; v, o: [T, Hv, 128] (bf16); g, beta: [T, Hv] fp32;
// state: [N, Hv, 128 (V), 128 (K)] fp32; cu_seqlens: [N + 1] int32;
// chunk_offsets: [N + 1] int32, cumulative ceil(len / 64); max_chunks: any
// upper bound on chunk_offsets[N] (e.g. ceil(T / 64) + N); a, p: bf16
// workspaces of at least [max_chunks, Hv, 64, 64]; gc: fp32 workspace of at
// least [max_chunks, Hv, 64].
void gdn_chunk_fwd_k128_v128_c64_bf16(
    torch::Tensor& o, torch::Tensor& final_state, torch::Tensor& a,
    torch::Tensor& p, torch::Tensor& gc, const torch::Tensor& q,
    const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& g,
    const torch::Tensor& beta, const torch::Tensor& initial_state,
    const torch::Tensor& cu_seqlens, const torch::Tensor& chunk_offsets,
    int64_t max_chunks, double scale) {
  check_qkv(q, "q");
  check_qkv(k, "k");
  check_qkv(v, "v");
  check_qkv(o, "o");
  const int64_t hk = q.size(1), hv = v.size(1), nseq = cu_seqlens.numel() - 1;
  TORCH_CHECK(k.sizes() == q.sizes() && o.sizes() == v.sizes() &&
                  q.size(0) == v.size(0) && hv % hk == 0,
              "q/k/v/o shape mismatch");
  for (const auto* t : {&g, &beta})
    TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous() &&
                    t->dim() == 2 && t->size(0) == v.size(0) && t->size(1) == hv,
                "g and beta must be contiguous fp32 [T, Hv]");
  const auto state_dtype = initial_state.scalar_type();
  TORCH_CHECK(state_dtype == at::kFloat || state_dtype == at::kBFloat16,
              "state must be fp32 or bf16 (--mamba-ssm-cache-dtype)");
  for (const torch::Tensor* t : {&initial_state, static_cast<const torch::Tensor*>(&final_state)})
    TORCH_CHECK(t->scalar_type() == state_dtype && t->is_contiguous() &&
                    t->dim() == 4 && t->size(0) == nseq && t->size(1) == hv &&
                    t->size(2) == kD && t->size(3) == kD,
                "states must be contiguous [N, Hv, 128, 128] in one dtype");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt &&
                  chunk_offsets.scalar_type() == at::kInt &&
                  chunk_offsets.numel() == nseq + 1,
              "cu_seqlens / chunk_offsets must be int32 [N + 1]");
  if (max_chunks == 0) {
    final_state.copy_(initial_state);
    return;
  }

  for (const torch::Tensor* t : {&a, &p})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->is_contiguous() &&
                    t->numel() >= max_chunks * hv * kC * kC,
                "a/p workspaces must be contiguous bf16 >= [max_chunks, Hv, 64, 64]");
  TORCH_CHECK(gc.scalar_type() == at::kFloat && gc.is_contiguous() &&
                  gc.numel() >= max_chunks * hv * kC,
              "gc workspace must be contiguous fp32 >= [max_chunks, Hv, 64]");
  auto stream = at::hip::getCurrentHIPStream();
  const auto* qp = reinterpret_cast<const uint16_t*>(q.data_ptr());
  const auto* kp = reinterpret_cast<const uint16_t*>(k.data_ptr());
  gdn_chunk_prep_kernel<<<dim3(hv, max_chunks), kThreads, 0, stream>>>(
      reinterpret_cast<uint16_t*>(a.data_ptr()),
      reinterpret_cast<uint16_t*>(p.data_ptr()), gc.data_ptr<float>(), qp, kp,
      g.data_ptr<float>(),
      beta.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(),
      chunk_offsets.data_ptr<int32_t>(), nseq, hv, hk, q.stride(0), q.stride(1),
      k.stride(0), k.stride(1), static_cast<float>(scale));
#define STRIX_GDN_SCAN(StateT)                                                \
  gdn_chunk_scan_kernel<kBV, StateT><<<dim3(kD / kBV, hv, nseq), kThreads, 0, \
                                       stream>>>(                             \
      reinterpret_cast<uint16_t*>(o.data_ptr()),                              \
      reinterpret_cast<StateT*>(final_state.data_ptr()),                      \
      reinterpret_cast<const StateT*>(initial_state.data_ptr()),              \
      reinterpret_cast<const uint16_t*>(a.data_ptr()),                        \
      reinterpret_cast<const uint16_t*>(p.data_ptr()), gc.data_ptr<float>(),  \
      qp, kp, reinterpret_cast<const uint16_t*>(v.data_ptr()),                \
      beta.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(),                 \
      chunk_offsets.data_ptr<int32_t>(), hv, hk, q.stride(0), q.stride(1),    \
      k.stride(0), k.stride(1), v.stride(0), v.stride(1), o.stride(0),        \
      o.stride(1), static_cast<float>(scale))
  if (state_dtype == at::kFloat)
    STRIX_GDN_SCAN(float);
  else
    STRIX_GDN_SCAN(uint16_t);
#undef STRIX_GDN_SCAN
}

}  // namespace strix
