// SPDX-License-Identifier: MIT
// Paged attention for decode (one query token per sequence), bf16 Q, bf16 or
// int8_per_token_head KV, optional sliding window, split-KV over key
// segments, optional attention sinks (one learned logit per query head in
// every softmax, as in vLLM's Triton kernel; folded in once, after the
// segments are combined). Built for (head_dim D, GQA G) = (128, 8), (128, 16),
// (256, 2), (256, 6), (256, 8) and (512, 8).
//
// One block owns one (sequence, KV head, key segment) and all G query heads of
// that KV head; the work scales with keys, so the kernels can stay at KV
// bandwidth (attn_prefill's 16-query tiles are mostly idle with one query).
//
// attn_decode_kernel (bf16 KV, and int8 at GQA 2): keys are processed KC at a
// time. The K/V chunk is staged in LDS (each cache row is read once per block),
// the G x KC scores are computed by thread groups that hold a query slice in
// registers, an online softmax runs per head, and each thread accumulates its
// slice of the [G][D] output from the staged V.
//
// attn_decode_wmma_kernel (int8 KV at GQA 6, 8 and 16): half the bytes per key
// leave the score and output arithmetic as the limit of the tile above, so
// these run on fp16 WMMA.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <type_traits>

#include "../strix_common.h"
#include "attn_split_reduce.h"

namespace strix {
namespace {

constexpr int kThreads = 256;

// Raw cache words in flight. HIP's uint4 in an array stays in private memory
// (scratch loads and stores in the chunk loop); these stay in registers.
typedef uint32_t u32x2 __attribute__((ext_vector_type(2), aligned(4)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4), aligned(4)));

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));

// The staged K/V are packed bf16 pairs, or fp16 ones for int8 KV (exact, one op
// per byte to widen). dot2: c + a.lo * b.lo + a.hi * b.hi, in fp32.
template <bool F16>
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  if constexpr (F16)
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(f16x2, a), __builtin_bit_cast(f16x2, b),
                                  c, false);
  else
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a),
                                           __builtin_bit_cast(bf16x2, b), c, false);
}
template <bool F16>
__device__ __forceinline__ float lo32(uint32_t w) {
  if constexpr (F16) return static_cast<float>(__builtin_bit_cast(f16x2, w).x);
  else return bf16_lo(w);
}
template <bool F16>
__device__ __forceinline__ float hi32(uint32_t w) {
  if constexpr (F16) return static_cast<float>(__builtin_bit_cast(f16x2, w).y);
  else return bf16_hi(w);
}

template <int D, int G, int KC, bool Q8, int KT = kThreads>
__global__ __launch_bounds__(KT) void attn_decode_kernel(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    const int8_t* __restrict__ k8, const int8_t* __restrict__ v8,
    const float* __restrict__ ksc, const float* __restrict__ vsc,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ block_table,
    int64_t bt_stride, int block_size, int64_t sq_t, int64_t sq_h, int64_t so_t,
    int64_t so_h, int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b,
    int64_t sv_s, int64_t sv_h, int64_t ss_b, int64_t ss_s, int64_t ss_h,
    float scale_log2, int win_left, int nseg,
    float* __restrict__ o_part, float* __restrict__ ml_part,
    const float* __restrict__ sinks) {
  constexpr int OPT = G * D / KT;   // output values per thread
  constexpr int kChunkLoads = (KC * D / 8 + KT - 1) / KT;  // guarded
  static_assert(G * D % KT == 0 && D % OPT == 0 && OPT % 2 == 0);
  // Each thread owns HPT heads x DPT dims of the [G][D] output; at D = 512 two
  // heads share each V read from LDS.
  constexpr int HPT = D == 512 ? 2 : 1;
  constexpr int DPT = OPT / HPT;
  static_assert(G % HPT == 0 && OPT % HPT == 0 && (HPT == 1 || DPT % 8 == 0));
  // Score phase: TPS threads share one head's dot products and each holds a
  // QS-dim slice of that head's query in registers (packed), then loops over KPG
  // keys: Q read from LDS per (head, key) pair would be ~2/3 of the kernel's
  // LDS traffic.
  constexpr int QS = 32;
  constexpr int TPS = D / QS;
  constexpr int NG = KT / TPS;          // thread groups
  constexpr int KPG = KC * G / NG;      // keys per group
  static_assert(D % QS == 0 && (TPS & (TPS - 1)) == 0 && TPS <= kWave);
  static_assert(NG % G == 0 && KPG * (NG / G) == KC);

  // K rows padded by 16 bytes: unpadded, the KC rows a wave's dot products
  // read sit a multiple of 256 bytes apart, all on one LDS bank.
  constexpr int KP = D + 8;
  __shared__ __attribute__((aligned(16))) uint16_t s_k[KC * KP];
  __shared__ __attribute__((aligned(16))) uint16_t s_v[KC * D];
  __shared__ float s_p[G][KC];
  __shared__ float s_m[G], s_l[G], s_alpha[G];
  // Q8 only: one dequant scale per staged key, read from the packed cache.
  __shared__ float s_ks[Q8 ? KC : 1], s_vs[Q8 ? KC : 1];

  const int n = blockIdx.x, kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x;
  const int kv_len = seq_lens[n], pos = kv_len - 1;
  const int k_lo = win_left >= 0 ? max(0, pos - win_left) : 0;
  const int seg_len = ((kv_len - k_lo + nseg - 1) / nseg + KC - 1) / KC * KC;
  const int k_begin = k_lo + seg * seg_len;
  const int k_end = min(kv_len, k_begin + seg_len);
  const int32_t* bt = block_table + int64_t(n) * bt_stride;
  const int h0 = kvh * G;

  const int grp = tid / TPS, lp = tid % TPS;
  const int sc_h = grp / (NG / G), sc_k0 = grp % (NG / G) * KPG;
  // Slice j*TPS*8 + lp*8 .. +8: the TPS lanes of a group read one contiguous
  // 16*TPS-byte run of a K row per j, so the LDS reads are conflict-free.
  uint32_t qr[QS / 2];
#pragma unroll
  for (int j = 0; j < QS / 8; ++j) {
    const uint4 w = *reinterpret_cast<const uint4*>(
        q + n * sq_t + (h0 + sc_h) * sq_h + j * TPS * 8 + lp * 8);
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      qr[j * 4 + e] =
          Q8 ? f16_bits(bf16_lo(ws[e])) | (f16_bits(bf16_hi(ws[e])) << 16) : ws[e];
    }
  }
  if (tid < G) {
    s_m[tid] = -INFINITY;
    s_l[tid] = 0.f;
  }
  if constexpr (Q8) {
    // Keys past the sequence are never staged; their scores are -inf, so the
    // V-scale multiply sees p=0. Leave a finite scale there, or 0 * garbage
    // (a shared-memory NaN) poisons the accumulator.
    if (tid < KC) {
      s_ks[tid] = 1.f;
      s_vs[tid] = 1.f;
    }
  }
  float acc[OPT];
#pragma unroll
  for (int i = 0; i < OPT; ++i) acc[i] = 0.f;
  const int o_flat = tid * DPT, o_h = o_flat / D * HPT, o_d = o_flat % D;
  barrier();

  // The next chunk's global loads are issued into registers before this
  // chunk's math, so they are in flight across its four barriers (barrier()
  // waits on LDS only). Raw bytes are held (the int8 widening happens at the
  // LDS store) to keep the registers small.
  //
  // Paged index: a chunk's keys span at most kSpan cache blocks when
  // KC <= (kSpan - 1) * block_size, so one scalar divide per chunk and kSpan
  // scalar block ids serve every key; a lane picks its block by comparing its
  // slot. Smaller cache blocks take the divide and table read per key.
  constexpr int kSpan = KC > 16 ? 3 : 2;
  const bool hoisted = KC <= (kSpan - 1) * block_size;
  using raw_t = std::conditional_t<Q8, u32x2, u32x4>;
  raw_t rk[kChunkLoads], rv[kChunkLoads];
  float rks[kChunkLoads], rvs[kChunkLoads];
  auto fetch = [&](int k0) {
    const int b0 = k0 / block_size, r0 = k0 - b0 * block_size;
    const int r_end = r0 + min(KC, k_end - k0);
    int blks[kSpan];
#pragma unroll
    for (int j = 0; j < kSpan; ++j)
      blks[j] = bt[r_end > j * block_size ? b0 + j : b0];  // never past k_end
#pragma unroll
    for (int i = 0; i < kChunkLoads; ++i) {
      const int ch = tid + i * KT;
      rk[i] = raw_t{};
      rv[i] = raw_t{};
      if (ch >= KC * D / 8) continue;
      const int key = ch / (D / 8), d8 = (ch % (D / 8)) * 8;
      if (k0 + key >= k_end) continue;
      int blk, slot;
      if (hoisted) {
        blk = blks[0], slot = r0 + key;
#pragma unroll
        for (int j = 1; j < kSpan; ++j) {
          if (slot >= block_size) blk = blks[j], slot -= block_size;
        }
      } else {
        blk = bt[(k0 + key) / block_size], slot = (k0 + key) % block_size;
      }
      const int64_t ka = blk * sk_b + slot * sk_s + kvh * sk_h;
      const int64_t va = blk * sv_b + slot * sv_s + kvh * sv_h;
      const int64_t ca = blk * ss_b + slot * ss_s + kvh * ss_h;
      if constexpr (Q8) {
        // The byte cache's head stride is only 4-byte aligned.
        const int8_t* pk = k8 + ka + d8;
        const int8_t* pv = v8 + va + d8;
        rk[i] = *reinterpret_cast<const u32x2*>(pk);
        rv[i] = *reinterpret_cast<const u32x2*>(pv);
        if (d8 == 0) {
          rks[i] = ksc[ca];
          rvs[i] = vsc[ca];
        }
      } else {
        rk[i] = *reinterpret_cast<const u32x4*>(kc + ka + d8);
        rv[i] = *reinterpret_cast<const u32x4*>(vc + va + d8);
      }
    }
  };
  if (k_begin < k_end) fetch(k_begin);
  for (int k0 = k_begin; k0 < k_end; k0 += KC) {
    // Stage K and V rows k0 .. k0 + KC (rows past k_end are zeros).
#pragma unroll
    for (int i = 0; i < kChunkLoads; ++i) {
      const int ch = tid + i * KT;
      if (ch >= KC * D / 8) continue;
      const int key = ch / (D / 8), d8 = (ch % (D / 8)) * 8;
      uint4 xk, xv;
      if constexpr (Q8) {
        // Widened to fp16 without the scale (exact); the scale is applied to
        // the score and to P below.
        uint32_t dk[4], dv[4];
        q8x8_to_f16(rk[i].x, rk[i].y, dk);
        q8x8_to_f16(rv[i].x, rv[i].y, dv);
        xk = make_uint4(dk[0], dk[1], dk[2], dk[3]);
        xv = make_uint4(dv[0], dv[1], dv[2], dv[3]);
        // Keys past k_end keep the initial scale of 1 (see above).
        if (d8 == 0 && k0 + key < k_end) {
          s_ks[key] = rks[i];
          s_vs[key] = rvs[i];
        }
      } else {
        xk = make_uint4(rk[i].x, rk[i].y, rk[i].z, rk[i].w);
        xv = make_uint4(rv[i].x, rv[i].y, rv[i].z, rv[i].w);
      }
      *reinterpret_cast<uint4*>(s_k + key * KP + d8) = xk;
      *reinterpret_cast<uint4*>(s_v + key * D + d8) = xv;
    }
    if (k0 + KC < k_end) fetch(k0 + KC);
    barrier();

    // Scores: each group of TPS threads owns one head and KPG keys.
#pragma unroll
    for (int kk = 0; kk < KPG; ++kk) {
      const int sc_key = sc_k0 + kk;
      const uint16_t* kr = s_k + sc_key * KP + lp * 8;
      float sdot = 0.f;
#pragma unroll
      for (int j = 0; j < QS / 8; ++j) {
        const uint4 w = *reinterpret_cast<const uint4*>(kr + j * TPS * 8);
        const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) sdot = dot2<Q8>(qr[j * 4 + e], ws[e], sdot);
      }
      sdot = group_sum<TPS>(sdot);
      if (lp == 0) {
        const bool seen = k0 + sc_key < k_end;  // window/causality: k_lo, kv_len
        float sv = sdot * scale_log2;
        if constexpr (Q8) sv *= s_ks[sc_key];
        s_p[sc_h][sc_key] = seen ? sv : -INFINITY;
      }
    }
    barrier();

    // Online softmax: KC aligned lanes per head, one per key, max and sum by
    // lane shuffles.
    static_assert(G * KC <= KT && KC <= kWave && (KC & (KC - 1)) == 0);
    if (tid < G * KC) {
      const int sh = tid / KC, sk = tid % KC;
      const float v = s_p[sh][sk];
      const float m_c = group_max<KC>(v);
      const float m_old = s_m[sh], m_new = fmaxf(m_old, m_c);
      const float m_ref = m_new == -INFINITY ? 0.f : m_new;
      const float p = __builtin_amdgcn_exp2f(v - m_ref);
      const float l_c = group_sum<KC>(p);
      // Fold V's per-key scale into P; l stays the unweighted sum, as in
      // attn_prefill's int8 path.
      s_p[sh][sk] = Q8 ? p * s_vs[sk] : p;
      // The group's reads of s_m / s_l above precede this write in the same
      // wave, so lane 0 can update them in place.
      if (sk == 0) {
        const float alpha = __builtin_amdgcn_exp2f(m_old - m_ref);
        s_alpha[sh] = alpha;
        s_l[sh] = s_l[sh] * alpha + l_c;
        s_m[sh] = m_new;
      }
    }
    barrier();

    // O[h][d] = alpha O + sum_k p[h][k] V[k][d] over this thread's slice.
    if constexpr (HPT == 1) {
      const float alpha = s_alpha[o_h];
#pragma unroll
      for (int i = 0; i < OPT; ++i) acc[i] *= alpha;
      for (int k = 0; k < KC; ++k) {
        const float p = s_p[o_h][k];
        const uint16_t* vr = s_v + k * D + o_d;
#pragma unroll
        for (int i = 0; i < OPT; i += 2) {
          const uint32_t w = *reinterpret_cast<const uint32_t*>(vr + i);
          acc[i] += p * lo32<Q8>(w);
          acc[i + 1] += p * hi32<Q8>(w);
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < HPT; ++j) {
        const float alpha = s_alpha[o_h + j];
#pragma unroll
        for (int i = 0; i < DPT; ++i) acc[j * DPT + i] *= alpha;
      }
#pragma unroll 2
      for (int k = 0; k < KC; ++k) {
        float p[HPT];
#pragma unroll
        for (int j = 0; j < HPT; ++j) p[j] = s_p[o_h + j][k];
        const uint16_t* vr = s_v + k * D + o_d;
#pragma unroll
        for (int i = 0; i < DPT; i += 8) {
          const uint4 v = *reinterpret_cast<const uint4*>(vr + i);
          const uint32_t ws[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int j = 0; j < HPT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              acc[j * DPT + i + 2 * e] += p[j] * lo32<Q8>(ws[e]);
              acc[j * DPT + i + 2 * e + 1] += p[j] * hi32<Q8>(ws[e]);
            }
        }
      }
    }
    barrier();  // the next chunk overwrites s_k, s_v, s_p
  }

#pragma unroll
  for (int j = 0; j < HPT; ++j) {
    const int oh = o_h + j, h = h0 + oh;
    const float* a = acc + j * DPT;
    if (nseg > 1) {
      const int hq = gridDim.y * G;
      const int64_t r = (int64_t(n) * hq + h) * nseg + seg;
#pragma unroll
      for (int i = 0; i < DPT; ++i) o_part[r * D + o_d + i] = a[i];
      if (o_d == 0) {
        ml_part[2 * r] = s_m[oh];
        ml_part[2 * r + 1] = s_l[oh];
      }
    } else {
      float inv = 1.f / s_l[oh];
      if (sinks != nullptr) {
        // The sink is one more (unscaled, natural-log) logit in the softmax.
        const float s2 = sinks[h] * 1.4426950408889634f, m = s_m[oh];
        const float m_fin = fmaxf(m, s2);
        const float e = __builtin_amdgcn_exp2f(m - m_fin);
        inv = e / (s_l[oh] * e + __builtin_amdgcn_exp2f(s2 - m_fin));
      }
      uint16_t* orow = out + n * so_t + h * so_h + o_d;
#pragma unroll
      for (int i = 0; i < DPT; i += 2)
        *reinterpret_cast<uint32_t*>(orow + i) =
            pack_bf16x2(a[i] * inv, a[i + 1] * inv);
    }
  }
}

// int8 KV on fp16 WMMA. The block's W = D / DW waves each hold DW dims of the
// query heads, one head per lane `lo` (G <= 16; lanes past G repeat heads and
// are not written), and work on 16-key subtiles as attn_prefill does with a
// head in place of a query token: S^T = K Q^T over the wave's dims, partial
// scores summed over the W waves through LDS, the softmax per lane, then O^T
// += V^T P^T. A wave loads its K rows straight from the cache into WMMA
// fragments; V^T goes through LDS, private to the wave, so the exchange of the
// partial scores is a step's only barrier. Its two buffers let a wave write the
// next step's scores while another still reads this step's.
template <int D, int G, int BN, int DW>
__global__ __launch_bounds__(D / DW * kWave) void attn_decode_wmma_kernel(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ q,
    const int8_t* __restrict__ k8, const int8_t* __restrict__ v8,
    const float* __restrict__ ksc, const float* __restrict__ vsc,
    const int32_t* __restrict__ seq_lens, const int32_t* __restrict__ block_table,
    int64_t bt_stride, int block_size, int64_t sq_t, int64_t sq_h, int64_t so_t,
    int64_t so_h, int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b,
    int64_t sv_s, int64_t sv_h, int64_t ss_b, int64_t ss_s, int64_t ss_h,
    float scale_log2, int win_left, int nseg, float* __restrict__ o_part,
    float* __restrict__ ml_part, const float* __restrict__ sinks) {
  constexpr int W = D / DW, NF = DW / 16, NS = BN / 16, PV = BN + 8;
  static_assert(W > 1 && G <= 16 && DW % 16 == 0 && BN % 16 == 0);
  __shared__ __attribute__((aligned(16))) uint16_t s_vt[W * DW * PV];
  __shared__ float s_x[2 * W][NS][kWave][8];
  __shared__ float s_ks[W * BN], s_vs[W * BN];

  const int n = blockIdx.x, kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int lo = lane % 16, hi = lane / 16;
  const int kv_len = seq_lens[n], pos = kv_len - 1;
  const int k_lo = win_left >= 0 ? max(0, pos - win_left) : 0;
  const int seg_len = ((kv_len - k_lo + nseg - 1) / nseg + BN - 1) / BN * BN;
  const int k_begin = k_lo + seg * seg_len;
  const int k_end = min(kv_len, k_begin + seg_len);
  const int32_t* bt = block_table + int64_t(n) * bt_stride;
  const int h = kvh * G + lo % G;
  uint16_t* vt = s_vt + wave * DW * PV;
  float* my_ks = s_ks + wave * BN;
  float* my_vs = s_vs + wave * BN;

  // Q^T B fragments (row = head lo), rounded to fp16: only values outside
  // fp16's range (|q| > 65504) lose anything.
  bf16x16 qf[NF];
  {
    const uint16_t* qrow = q + n * sq_t + h * sq_h + wave * DW;
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      qf[f] = load_frag_bf16(qrow + f * 16, 0, 0);
#pragma unroll
      for (int i = 0; i < 16; ++i)
        qf[f][i] = static_cast<short>(
            f16_bits(__uint_as_float(uint32_t(uint16_t(qf[f][i])) << 16)));
    }
  }
  f32x8 o_acc[NF];
#pragma unroll
  for (int f = 0; f < NF; ++f) o_acc[f] = f32x8{};
  float m_run = -INFINITY, l_run = 0.f;
  // P V-scale is rounded to fp16; times 2^10 (undone in the output) small
  // weights of a flat softmax stay above its subnormals.
  constexpr float kPScale = 1024.f, kPInv = 1.f / kPScale;
  const uint32_t p_sel = hi ? 0x01000504u : 0x05040100u;

  // One step's raw K rows (key 16 j + lo, all DW dims; both halves of the wave
  // load the same ones, as the WMMA fragments need), V chunks (the dims
  // (hi + 2 i) * 8 of that key) and scales, in flight across the math below.
  u32x4 rk[NS * NF];
  u32x2 rv[NS * NF];
  float rks[NS], rvs[NS];
  // Paged index: see attn_decode_kernel.
  constexpr int kSpan = BN > 16 ? 3 : 2;
  const bool hoisted = BN <= (kSpan - 1) * block_size;
  auto fetch = [&](int k0) {
    const int b0 = k0 / block_size, r0 = k0 - b0 * block_size;
    const int r_end = r0 + min(BN, k_end - k0);
    int blks[kSpan];
#pragma unroll
    for (int j = 0; j < kSpan; ++j)
      blks[j] = bt[r_end > j * block_size ? b0 + j : b0];  // never past k_end
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      const int key = 16 * j + lo;
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        rk[j * NF + f] = u32x4{};
        rv[j * NF + f] = u32x2{};
      }
      rks[j] = 1.f;
      rvs[j] = 1.f;
      if (k0 + key >= k_end) continue;
      int blk, slot;
      if (hoisted) {
        blk = blks[0], slot = r0 + key;
#pragma unroll
        for (int s = 1; s < kSpan; ++s) {
          if (slot >= block_size) blk = blks[s], slot -= block_size;
        }
      } else {
        blk = bt[(k0 + key) / block_size], slot = (k0 + key) % block_size;
      }
      const int8_t* pk = k8 + blk * sk_b + slot * sk_s + kvh * sk_h + wave * DW;
      const int8_t* pv = v8 + blk * sv_b + slot * sv_s + kvh * sv_h + wave * DW;
      const int64_t ca = blk * ss_b + slot * ss_s + kvh * ss_h;
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        rk[j * NF + f] = *reinterpret_cast<const u32x4*>(pk + f * 16);
        rv[j * NF + f] = *reinterpret_cast<const u32x2*>(pv + (hi + 2 * f) * 8);
      }
      rks[j] = ksc[ca];
      rvs[j] = vsc[ca];
    }
  };
  if (k_begin < k_end) fetch(k_begin);
  int parity = 0;
  for (int k0 = k_begin; k0 < k_end; k0 += BN, parity ^= 1) {
    // The step's K as fp16 fragments, its V^T and scales in LDS.
    bf16x16 kf[NS * NF];
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      my_ks[16 * j + lo] = rks[j];
      my_vs[16 * j + lo] = rvs[j];
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        const u32x4 r = rk[j * NF + f];
        uint32_t w[2][4];
        q8x8_to_f16(r.x, r.y, w[0]);
        q8x8_to_f16(r.z, r.w, w[1]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          kf[j * NF + f][2 * i] = static_cast<short>(w[i / 4][i % 4] & 0xffffu);
          kf[j * NF + f][2 * i + 1] = static_cast<short>(w[i / 4][i % 4] >> 16);
        }
        uint32_t v[4];
        q8x8_to_f16(rv[j * NF + f].x, rv[j * NF + f].y, v);
        uint16_t* row = vt + (hi + 2 * f) * 8 * PV + 16 * j + lo;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          row[2 * e * PV] = static_cast<uint16_t>(v[e] & 0xffffu);
          row[(2 * e + 1) * PV] = static_cast<uint16_t>(v[e] >> 16);
        }
      }
    }
    if (k0 + BN < k_end) fetch(k0 + BN);
    __builtin_amdgcn_wave_barrier();

    // S^T: element e of subtile j is key k0 + 16 j + 2 e + hi, head lo.
    f32x8 s[NS];
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      s[j] = f32x8{};
#pragma unroll
      for (int f = 0; f < NF; ++f) s[j] = wmma_f16(kf[j * NF + f], qf[f], s[j]);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        s[j][e] *= scale_log2 * my_ks[16 * j + 2 * e + hi];
        s_x[parity * W + wave][j][lane][e] = s[j][e];
      }
    }
    barrier();
    float m_loc = -INFINITY;
#pragma unroll
    for (int j = 0; j < NS; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
#pragma unroll
        for (int o = 1; o < W; ++o)
          s[j][e] += s_x[parity * W + (wave + o) % W][j][lane][e];
        if (k0 + 16 * j + 2 * e + hi >= k_end) s[j][e] = -INFINITY;
        m_loc = fmaxf(m_loc, s[j][e]);
      }
    m_loc = fmaxf(m_loc, xor_shuffle_f32(m_loc, 16));
    const float m_new = fmaxf(m_run, m_loc);
    const float m_ref = m_new == -INFINITY ? 0.f : m_new;
    const float alpha = __builtin_amdgcn_exp2f(m_run - m_ref);
    // The running max rarely moves deep in a context.
    if (__any(alpha != 1.f)) {
#pragma unroll
      for (int f = 0; f < NF; ++f)
#pragma unroll
        for (int e = 0; e < 8; ++e) o_acc[f][e] *= alpha;
    }
    float l_loc = 0.f;
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      // P^T B fragment: row = head lo, 16 keys; each lane rounds its own value
      // and swaps it with the other half (as in attn_prefill). V's per-key
      // scale rides on P; l stays the unweighted sum.
      bf16x16 pf;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float p = __builtin_amdgcn_exp2f(s[j][e] - m_ref);
        l_loc += p;
        const uint32_t mine = f16_bits(p * (kPScale * my_vs[16 * j + 2 * e + hi]));
        const uint32_t other = __float_as_uint(xor_shuffle_f32(__uint_as_float(mine), 16));
        const uint32_t pk = __builtin_amdgcn_perm(other, mine, p_sel);
        pf[2 * e] = static_cast<short>(pk & 0xffffu);
        pf[2 * e + 1] = static_cast<short>(pk >> 16);
      }
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        const bf16x16 vf = load_frag_bf16(vt + f * 16 * PV + j * 16, PV, lane);
        o_acc[f] = wmma_f16(vf, pf, o_acc[f]);
      }
    }
    l_loc += xor_shuffle_f32(l_loc, 16);
    l_run = l_run * alpha + l_loc;
    m_run = m_new;
    __builtin_amdgcn_wave_barrier();  // this step's reads of V^T and the scales
  }

  // O^T[d][head lo]: element e of fragment f is d = wave * DW + 16 f + 2 e + hi.
  if (lo >= G) return;
  if (nseg > 1) {
    const int hq = gridDim.y * G;
    const int64_t r = (int64_t(n) * hq + h) * nseg + seg;
    float* prow = o_part + r * D + wave * DW;
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int e = 0; e < 8; ++e) prow[f * 16 + 2 * e + hi] = o_acc[f][e] * kPInv;
    if (wave == 0 && hi == 0) {
      ml_part[2 * r] = m_run;
      ml_part[2 * r + 1] = l_run;
    }
  } else {
    float inv = 1.f / l_run;
    if (sinks != nullptr) {
      // The sink is one more (unscaled, natural-log) logit in the softmax.
      const float s2 = sinks[h] * 1.4426950408889634f;
      const float m_fin = fmaxf(m_run, s2);
      const float a = __builtin_amdgcn_exp2f(m_run - m_fin);
      inv = a / (l_run * a + __builtin_amdgcn_exp2f(s2 - m_fin));
    }
    uint16_t* orow = out + n * so_t + h * so_h + wave * DW;
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        orow[f * 16 + 2 * e + hi] = static_cast<uint16_t>(
            pack_bf16x2(o_acc[f][e] * (inv * kPInv), 0.f) & 0xffffu);
  }
}

}  // namespace

// q, out: [N, Hq, D] bf16, one query token per sequence (decode); key_cache,
// value_cache: [num_blocks, block_size, Hkv, D] bf16 views; seq_lens: [N]
// int32 (the query is the last position); block_table: [N, max_blocks]
// int32. window_left >= 0 restricts the query at p to keys p - window_left
// .. p (vLLM's (window - 1, 0)). num_splits key segments per sequence.
void attn_decode_bf16(torch::Tensor& out, const torch::Tensor& q,
                      const torch::Tensor& key_cache,
                      const torch::Tensor& value_cache,
                      const torch::Tensor& seq_lens,
                      const torch::Tensor& block_table, double scale,
                      int64_t num_splits, int64_t window_left,
                      const std::optional<torch::Tensor>& k_scale,
                      const std::optional<torch::Tensor>& v_scale,
                      const std::optional<torch::Tensor>& sinks) {
  const int64_t d = q.size(-1), nseq = q.size(0);
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.size(-1) == d &&
                  q.stride(-1) == 1,
              "q must be bf16 with a contiguous head_dim");
  const bool q8 = key_cache.scalar_type() != at::kBFloat16;
  TORCH_CHECK(key_cache.scalar_type() ==
                  (q8 ? at::kChar : at::kBFloat16) &&
                  value_cache.scalar_type() == key_cache.scalar_type(),
              "caches must both be bf16 or both int8");
  for (const auto* t : {&key_cache, &value_cache})
    TORCH_CHECK(t->size(-1) >= d && t->stride(-1) == 1,
                "cache head_dim must be at least q's, contiguous");
  int64_t sc_b = 0, sc_s = 0, sc_h = 0;
  if (q8) {
    TORCH_CHECK(k_scale.has_value() && v_scale.has_value(),
                "int8 KV needs k_scale and v_scale");
    const auto& ks = *k_scale;
    const auto& vs = *v_scale;
    for (const auto* t : {&ks, &vs})
      TORCH_CHECK(t->scalar_type() == at::kFloat && t->dim() == 3 &&
                      t->size(0) == key_cache.size(0) &&
                      t->size(1) == key_cache.size(1) &&
                      t->size(2) == key_cache.size(2),
                  "scales must be fp32 [blocks, block_size, Hkv]");
    sc_b = ks.stride(0), sc_s = ks.stride(1), sc_h = ks.stride(2);
  }
  const float* ksc_ptr = q8 ? k_scale->data_ptr<float>() : nullptr;
  const float* vsc_ptr = q8 ? v_scale->data_ptr<float>() : nullptr;
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.sizes() == q.sizes() &&
                  out.stride(-1) == 1,
              "out must match q");
  TORCH_CHECK(q.dim() == 3 && key_cache.dim() == 4 && value_cache.dim() == 4,
              "q [N, H, D], caches [blocks, block_size, Hkv, D]");
  const int64_t hq = q.size(1), hkv = key_cache.size(2);
  TORCH_CHECK(hq % hkv == 0, "Hq % Hkv");
  const int64_t g = hq / hkv;
  const int64_t cache_align = q8 ? 4 : 8;
  for (const auto* t : {&key_cache, &value_cache})
    TORCH_CHECK(t->stride(0) % cache_align == 0 &&
                    t->stride(1) % cache_align == 0 &&
                    t->stride(2) % cache_align == 0,
                "aligned cache rows");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt && block_table.scalar_type() == at::kInt &&
                  block_table.stride(1) == 1 && seq_lens.numel() >= nseq,
              "int32 metadata");
  TORCH_CHECK(num_splits >= 1 && num_splits <= 64, "num_splits in [1, 64]");
  if (nseq == 0) return;
  torch::Tensor o_part, ml_part;
  if (num_splits > 1) {
    auto f32 = q.options().dtype(at::kFloat);
    o_part = torch::empty({nseq, hq, num_splits, d}, f32);
    ml_part = torch::empty({nseq, hq, num_splits, 2}, f32);
  }
  float* op = num_splits > 1 ? o_part.data_ptr<float>() : nullptr;
  float* mp = num_splits > 1 ? ml_part.data_ptr<float>() : nullptr;
  const float* sink_ptr = nullptr;
  if (sinks.has_value()) {
    TORCH_CHECK(sinks->scalar_type() == at::kFloat && sinks->dim() == 1 &&
                    sinks->size(0) == hq && sinks->is_contiguous(),
                "sinks must be contiguous fp32 [Hq]");
    sink_ptr = sinks->data_ptr<float>();
  }
  auto stream = at::hip::getCurrentHIPStream();
  const float scale_log2 = static_cast<float>(scale) * 1.4426950408889634f;
  const dim3 grid(nseq, hkv, num_splits);
#define STRIX_DECODE(D, G, KC, Q8V, KT)                                          \
  attn_decode_kernel<D, G, KC, Q8V, KT><<<grid, KT, 0, stream>>>(              \
      reinterpret_cast<uint16_t*>(out.data_ptr()),                           \
      reinterpret_cast<const uint16_t*>(q.data_ptr()),                       \
      reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),               \
      reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),             \
      reinterpret_cast<const int8_t*>(key_cache.data_ptr()),                 \
      reinterpret_cast<const int8_t*>(value_cache.data_ptr()), ksc_ptr,      \
      vsc_ptr, seq_lens.data_ptr<int32_t>(),                                 \
      block_table.data_ptr<int32_t>(), block_table.stride(0),               \
      key_cache.size(1), q.stride(0), q.stride(1), out.stride(0),           \
      out.stride(1), key_cache.stride(0), key_cache.stride(1),               \
      key_cache.stride(2), value_cache.stride(0), value_cache.stride(1),     \
      value_cache.stride(2), sc_b, sc_s, sc_h, scale_log2,                   \
      static_cast<int>(window_left), static_cast<int>(num_splits), op, mp,  \
      sink_ptr)
#define STRIX_DECODE_WMMA(D, G, BN, DW)                                          \
  attn_decode_wmma_kernel<D, G, BN, DW><<<grid, D / DW * kWave, 0, stream>>>( \
      reinterpret_cast<uint16_t*>(out.data_ptr()),                           \
      reinterpret_cast<const uint16_t*>(q.data_ptr()),                       \
      reinterpret_cast<const int8_t*>(key_cache.data_ptr()),                 \
      reinterpret_cast<const int8_t*>(value_cache.data_ptr()), ksc_ptr,      \
      vsc_ptr, seq_lens.data_ptr<int32_t>(),                                 \
      block_table.data_ptr<int32_t>(), block_table.stride(0),               \
      key_cache.size(1), q.stride(0), q.stride(1), out.stride(0),           \
      out.stride(1), key_cache.stride(0), key_cache.stride(1),               \
      key_cache.stride(2), value_cache.stride(0), value_cache.stride(1),     \
      value_cache.stride(2), sc_b, sc_s, sc_h, scale_log2,                   \
      static_cast<int>(window_left), static_cast<int>(num_splits), op, mp,  \
      sink_ptr)
  // STRIX_DECODE(D, G, KC, q8, KT): KC keys per chunk, KT threads. KT is 256,
  // or 192 for GQA 6 (its factor of 3 must divide the threads); the softmax
  // phase needs G * KC <= KT, KC a power of two up to 32.
  // STRIX_DECODE_WMMA(D, G, BN, DW), int8 KV: BN keys per step, DW dims per
  // wave.
  //   (256, 2)   Gemma-4 sliding layers
  //   (256, 6)   Qwen3.8-27B
  //   (256, 8)   Qwen3.6-35B-A3B
  //   (512, 8)   Gemma-4 full layers
  //   (128, 8)   Flint
  //   (128, 16)  Muse-Glimmer
  //   (64, 4)    LFM2.5
  if (q8) {
    if (d == 256 && g == 2) STRIX_DECODE(256, 2, 16, true, 256);
    else if (d == 256 && g == 8) STRIX_DECODE_WMMA(256, 8, 32, 64);
    else if (d == 256 && g == 6) STRIX_DECODE_WMMA(256, 6, 32, 64);
    else if (d == 512 && g == 8) STRIX_DECODE_WMMA(512, 8, 16, 64);
    else if (d == 128 && g == 8) STRIX_DECODE_WMMA(128, 8, 32, 32);
    else if (d == 128 && g == 16) STRIX_DECODE_WMMA(128, 16, 32, 32);
    else if (d == 64 && g == 4) STRIX_DECODE_WMMA(64, 4, 32, 32);
    else TORCH_CHECK(false, "attn_decode_bf16 int8: head_dim ", d, " GQA ", g,
                     " not built (64: 4; 128: 8, 16; 256: 2, 6, 8; 512: 8)");
  } else {
    if (d == 256 && g == 2) STRIX_DECODE(256, 2, 16, false, 256);
    else if (d == 256 && g == 8) STRIX_DECODE(256, 8, 32, false, 256);
    else if (d == 256 && g == 6) STRIX_DECODE(256, 6, 16, false, 192);
    else if (d == 512 && g == 8) STRIX_DECODE(512, 8, 16, false, 256);
    else if (d == 128 && g == 8) STRIX_DECODE(128, 8, 32, false, 256);
    else if (d == 128 && g == 16) STRIX_DECODE(128, 16, 16, false, 256);
    else if (d == 64 && g == 4) STRIX_DECODE(64, 4, 32, false, 128);
    else TORCH_CHECK(false, "attn_decode_bf16: head_dim ", d, " GQA ", g,
                     " not built (64: 4; 128: 8, 16; 256: 2, 6, 8; 512: 8)");
  }
#undef STRIX_DECODE
#undef STRIX_DECODE_WMMA
  if (num_splits > 1)
    launch_attn_split_reduce(d, dim3(nseq, hq), stream,
                             reinterpret_cast<uint16_t*>(out.data_ptr()), op, mp,
                             static_cast<int>(num_splits), hq, out.stride(0),
                             out.stride(1), sink_ptr);
}

}  // namespace strix
