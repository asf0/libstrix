// SPDX-License-Identifier: MIT
// Inkling 2 relative attention for decode and verify batches (the small-
// query side of the port's attention, where the Triton kernel split-KVs):
// head 128, GQA 3, bf16 or int8_per_token_head KV, the query's relative
// logits from the sheared table, the output gate and the per-token attention
// sink applied once per (token, head) after the key segments are combined:
//
//     out = o * sigmoid(gate) * sigmoid(lse - sink),  lse natural-log.
//
// Both kernels below are built to keep every wave streaming the KV on its own
// (no barrier while keys remain) and to issue loads of 256 contiguous bytes
// (bf16; int8: 128) per key: DRAM gives 5 % more to those than to 64-byte
// ones. A wave's keys are every nseg-th 16-key chunk, so the waves of a KV
// head read neighbouring keys. The softmax is online against a reference
// exponent per (token, head) that is raised, and the sums rescaled, only when
// a score passes it by kGrow (p stays finite, and it happens a few times).
// Scores run in the log2 domain; int8 K and V are widened to fp16 without the
// scale, and the per-key scale is folded into the score (K) and into P (V).
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <climits>
#include <type_traits>

#include "../strix_common.h"
#include "inkling_kv.h"

namespace strix {
namespace {

constexpr int D = 128, G = 3;  // head dim, query heads per KV head
constexpr int KC = 16;         // keys per chunk; a cache block holds at least one
constexpr float kLOG2E = 1.4426950408889634f;
constexpr float kLN2 = 0.6931471805599453f;
constexpr float kGrow = 24.f, kNoKeys = -1e30f;

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
// int8 cache rows are 4-byte aligned. Not HIP's uint4: an array of those
// stays in private memory.
typedef uint32_t u32x4 __attribute__((ext_vector_type(4), aligned(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2), aligned(4)));

__device__ __forceinline__ f16x2 as_f16x2(uint32_t w) { return __builtin_bit_cast(f16x2, w); }

// c + a . b over a pair: fp16 for int8 KV, bf16 otherwise.
template <bool Q8>
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  if constexpr (Q8) return __builtin_amdgcn_fdot2(as_f16x2(a), as_f16x2(b), c, false);
  else return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a),
                                              __builtin_bit_cast(bf16x2, b), c, false);
}

__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t w) {
  const f16x2 h = {static_cast<_Float16>(bf16_lo(w)), static_cast<_Float16>(bf16_hi(w))};
  return __builtin_bit_cast(uint32_t, h);
}

__device__ __forceinline__ float sigmoid(float x) {
  return 1.f / (1.f + __builtin_amdgcn_exp2f(-x * kLOG2E));
}

struct Batch {
  const uint16_t *q, *rel, *gate, *sink;
  const uint8_t *kc, *vc;
  const float *ksc, *vsc;
  const int32_t *cu_q, *seq_lens, *block_table;
  // Element strides; the KV ones in bytes, the scales' in floats.
  int64_t sq_t, sq_h, sr_t, sr_h, sg_t, ss_t, bt, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, sc_b, sc_s,
      sc_h;
  int nreq, T, hq, e_pad, block_size, win_left;
  float scale_log2;
};

// The tokens [u, u + nq) of tile x (QN consecutive tokens of a request, the
// requests' tiles in order) and their keys. False past the last tile.
struct Tile {
  int u, nq, pos, lo, hi, key0;  // pos: of token u; key0: of the sheared table
  const int32_t* bt;
};
template <int QN>
__device__ __forceinline__ bool find_tile(const Batch& b, int x, Tile& t) {
  int req = 0;
  if constexpr (QN == 1) {
    if (x >= b.cu_q[b.nreq]) return false;
    // The largest req with cu_q[req] <= x (cu_q is nondecreasing).
    for (int end = b.nreq; end - req > 1;) {
      const int mid = (req + end) / 2;
      if (b.cu_q[mid] <= x) req = mid;
      else end = mid;
    }
    t.u = x;
    t.nq = 1;
  } else {
    for (; req < b.nreq; ++req) {
      const int n = (b.cu_q[req + 1] - b.cu_q[req] + QN - 1) / QN;
      if (x < n) break;
      x -= n;
    }
    if (req == b.nreq) return false;
    t.u = b.cu_q[req] + x * QN;
    t.nq = min(QN, b.cu_q[req + 1] - t.u);
  }
  const int seq_len = b.seq_lens[req];
  t.pos = seq_len - (b.cu_q[req + 1] - t.u);
  t.lo = b.win_left >= 0 ? max(0, t.pos - b.win_left) : 0;
  t.hi = t.pos + t.nq;
  // Sheared table: key j reads column j - key0, the same for all tokens of a
  // decode or verify batch (one 128-query tile); earlier keys have no logit.
  t.key0 = (seq_len + 127) / 128 * 128 - b.e_pad;
  t.bt = b.block_table + int64_t(req) * b.bt;
  return true;
}

// A uniform table entry by a scalar load: as a plain global load it is a
// vector one wherever the compiler sees a possible write (the verify kernel's
// `ordered()`), and waiting for that one waits for all KV loads before it.
__device__ __forceinline__ int64_t table_entry(const int32_t* p) {
  return *(const __attribute__((address_space(4))) int32_t*)p;
}

// The (at most two) cache blocks of a chunk, at one KV head.
struct Pages {
  const uint8_t *a, *b;
  const float *sa, *sb;
  int r0;  // slot of the chunk's first key in block a
};
__device__ __forceinline__ Pages pages(const Batch& b, const Tile& t, int k0, int k_end, int kvh,
                                       const uint8_t* base, int64_t sb, int64_t sh,
                                       const float* scales) {
  const int b0 = k0 / b.block_size;
  const int64_t p0 = table_entry(t.bt + b0);
  const int64_t p1 = table_entry(t.bt + (min(k0 + KC, k_end) - 1) / b.block_size);
  return {base + p0 * sb + kvh * sh, base + p1 * sb + kvh * sh,
          scales + p0 * b.sc_b + kvh * b.sc_h, scales + p1 * b.sc_b + kvh * b.sc_h,
          k0 - b0 * b.block_size};
}

// One token: a block of kNW waves owns a (token, KV head). Each lane group of
// TPS lanes takes a key, and its lanes a slice of the 128 dims of K, V and the
// G query heads: the group's scores are summed by lane shuffles, and every
// group runs its own softmax (reference, l, O += p V), so a wave needs neither
// LDS nor a partner until its keys are done. Then the groups, and through LDS
// the waves, combine and write the gated output: no second launch.
constexpr int kNW = 16;

template <bool Q8, int TPS>
__global__ __launch_bounds__(32 * kNW) void inkling_rel_decode_kernel(uint16_t* __restrict__ out,
                                                                      Batch b) {
  constexpr int KPW = 32 / TPS, DL = D / TPS, IT = KC / KPW;  // keys per wave, dims per lane
  constexpr int EB = Q8 ? 1 : 2;
  constexpr int LB = DL * EB < 16 ? DL * EB : 16;       // bytes per load
  constexpr int U = LB / EB, NL = DL / U, LW = LB / 4;  // dims, loads, words
  static_assert(KC % KPW == 0 && LB >= 8 && G <= TPS);
  using raw_t = std::conditional_t<LB == 16, u32x4, u32x2>;
  __shared__ __attribute__((aligned(16))) float s_o[kNW][G][D];
  __shared__ float s_m[kNW][G], s_l[kNW][G];
  const int kvh = blockIdx.y, wave = threadIdx.x / 32, tid = threadIdx.x % 32;
  const int h0 = kvh * G;
  // The first G * D / 2 threads write two dims of the output each.
  const int o_r = threadIdx.x / (D / 2), o_d = threadIdx.x % (D / 2) * 2;
  Tile t;
  if (!find_tile<1>(b, blockIdx.x, t)) {
    if (o_r < G)
      *reinterpret_cast<uint32_t*>(out + (int64_t(blockIdx.x) * b.hq + h0 + o_r) * D + o_d) = 0;
    return;
  }
  const int j = tid / TPS, s = tid % TPS;  // key group, slice
  uint32_t qw[G][DL / 2];
#pragma unroll
  for (int r = 0; r < G; ++r) {
    const uint16_t* qp = b.q + int64_t(t.u) * b.sq_t + (h0 + r) * b.sq_h + s * U;
#pragma unroll
    for (int c = 0; c < NL; ++c)
#pragma unroll
      for (int e = 0; e < U / 2; ++e) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(qp + c * U * TPS + 2 * e);
        qw[r][c * U / 2 + e] = Q8 ? bf16x2_to_f16x2(w) : w;
      }
  }
  float ref[G], l[G], acc[G][DL];
#pragma unroll
  for (int r = 0; r < G; ++r) {
    ref[r] = kNoKeys, l[r] = 0.f;
#pragma unroll
    for (int i = 0; i < DL; ++i) acc[r][i] = 0.f;
  }
  for (int k0 = t.lo + wave * KC; k0 < t.hi; k0 += kNW * KC) {
    // The whole chunk is loaded before its math, 16 keys in flight per wave.
    raw_t rk[IT][NL], rv[IT][NL];
    float ks[IT], vs[IT], bias[IT];
    const Pages pk = pages(b, t, k0, t.hi, kvh, b.kc, b.sk_b, b.sk_h, b.ksc);
    const Pages pv = pages(b, t, k0, t.hi, kvh, b.vc, b.sv_b, b.sv_h, b.vsc);
    const int last = min(KC, t.hi - k0) - 1;  // keys past it repeat it, unseen
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      int slot = pk.r0 + min(it * KPW + j, last);
      const bool x = slot >= b.block_size;
      if (x) slot -= b.block_size;
      const uint8_t* kp = (x ? pk.b : pk.a) + int64_t(slot) * b.sk_s + s * LB;
      const uint8_t* vp = (x ? pv.b : pv.a) + int64_t(slot) * b.sv_s + s * LB;
#pragma unroll
      for (int c = 0; c < NL; ++c) {
        rk[it][c] = *reinterpret_cast<const raw_t*>(kp + c * LB * TPS);
        rv[it][c] = *reinterpret_cast<const raw_t*>(vp + c * LB * TPS);
      }
      if constexpr (Q8) {
        ks[it] = (x ? pk.sb : pk.sa)[int64_t(slot) * b.sc_s];
        vs[it] = (x ? pv.sb : pv.sa)[int64_t(slot) * b.sc_s];
      }
    }
    // Lane s of a key group loads head s's relative logit.
#pragma unroll
    for (int it = 0; it < IT; ++it) bias[it] = 0.f;
    if (k0 + KC > t.key0) {
#pragma unroll
      for (int it = 0; it < IT; ++it) {
        const int key = k0 + it * KPW + j;
        if (s < G && key >= t.key0 && key < t.hi)
          bias[it] =
              bf16_lo(b.rel[int64_t(t.u) * b.sr_t + (h0 + s) * b.sr_h + (key - t.key0)]) * kLOG2E;
      }
    }
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      uint32_t kw[DL / 2], vw[DL / 2];  // bf16 pairs, int8: fp16 pairs
#pragma unroll
      for (int c = 0; c < NL; ++c) {
        if constexpr (Q8) {
#pragma unroll
          for (int e = 0; e < LW; e += 2) {
            uint32_t ok[4], ov[4];
            q8x8_to_f16(rk[it][c][e], rk[it][c][e + 1], ok);
            q8x8_to_f16(rv[it][c][e], rv[it][c][e + 1], ov);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              kw[(c * LW + e) * 2 + i] = ok[i];
              vw[(c * LW + e) * 2 + i] = ov[i];
            }
          }
        } else {
#pragma unroll
          for (int e = 0; e < LW; ++e) {
            kw[c * LW + e] = rk[it][c][e];
            vw[c * LW + e] = rv[it][c][e];
          }
        }
      }
      const float cs = Q8 ? b.scale_log2 * ks[it] : b.scale_log2;
      const bool seen = k0 + it * KPW + j < t.hi;
      float sc[G];
      bool grow = false;
#pragma unroll
      for (int r = 0; r < G; ++r) {
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < DL / 2; ++i) d = dot2<Q8>(qw[r][i], kw[i], d);
        d = group_sum<TPS>(d * cs + (s == r ? bias[it] : 0.f));
        sc[r] = seen ? d : -INFINITY;
        grow |= sc[r] > ref[r] + kGrow;
      }
      if (grow) {
#pragma unroll
        for (int r = 0; r < G; ++r) {
          const float m = fmaxf(ref[r], sc[r]), a = __builtin_amdgcn_exp2f(ref[r] - m);
          ref[r] = m;
          l[r] *= a;
#pragma unroll
          for (int i = 0; i < DL; ++i) acc[r][i] *= a;
        }
      }
#pragma unroll
      for (int r = 0; r < G; ++r) {
        float p = __builtin_amdgcn_exp2f(sc[r] - ref[r]);
        l[r] += p;
        if constexpr (Q8) p *= vs[it];
#pragma unroll
        for (int i = 0; i < DL / 2; ++i) {
          if constexpr (Q8) {
            const f16x2 x = as_f16x2(vw[i]);
            acc[r][2 * i] = __builtin_fmaf(p, (float)x.x, acc[r][2 * i]);
            acc[r][2 * i + 1] = __builtin_fmaf(p, (float)x.y, acc[r][2 * i + 1]);
          } else {
            acc[r][2 * i] += p * bf16_lo(vw[i]);
            acc[r][2 * i + 1] += p * bf16_hi(vw[i]);
          }
        }
      }
    }
  }
  // The key groups of the wave, then the waves.
#pragma unroll
  for (int r = 0; r < G; ++r) {
    float m = ref[r];
#pragma unroll
    for (int st = TPS; st < kWave; st *= 2) m = fmaxf(m, xor_shuffle_f32(m, st));
    const float a = __builtin_amdgcn_exp2f(ref[r] - m);
    l[r] *= a;
#pragma unroll
    for (int st = TPS; st < kWave; st *= 2) l[r] += xor_shuffle_f32(l[r], st);
#pragma unroll
    for (int i = 0; i < DL; ++i) {
      acc[r][i] *= a;
#pragma unroll
      for (int st = TPS; st < kWave; st *= 2) acc[r][i] += xor_shuffle_f32(acc[r][i], st);
    }
    if (j == r % KPW) {
#pragma unroll
      for (int i = 0; i < DL; ++i) s_o[wave][r][(i / U) * U * TPS + s * U + i % U] = acc[r][i];
      if (s == 0) s_m[wave][r] = m, s_l[wave][r] = l[r];
    }
  }
  barrier();
  if (o_r >= G) return;
  float m = kNoKeys, lt = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll
  for (int w = 0; w < kNW; ++w) m = fmaxf(m, s_m[w][o_r]);
#pragma unroll 8
  for (int w = 0; w < kNW; ++w) {
    const float f = __builtin_amdgcn_exp2f(s_m[w][o_r] - m);
    lt += f * s_l[w][o_r];
    a0 += f * s_o[w][o_r][o_d];
    a1 += f * s_o[w][o_r][o_d + 1];
  }
  // A token sees at least its own key: lt > 0.
  const int h = h0 + o_r;
  const uint32_t gw = *reinterpret_cast<const uint32_t*>(b.gate + t.u * b.sg_t + h * D + o_d);
  const float f = sigmoid((m + log2f(lt)) * kLN2 - bf16_lo(b.sink[t.u * b.ss_t + h])) / lt;
  *reinterpret_cast<uint32_t*>(out + (int64_t(t.u) * b.hq + h) * D + o_d) =
      pack_bf16x2(a0 * f * sigmoid(bf16_lo(gw)), a1 * f * sigmoid(bf16_hi(gw)));
}

// Up to QN tokens of a request: a block (one wave) owns a (tile, KV head, key
// segment) and reads its keys once for all of them. The R = QN x G rows of
// queries (in LDS) and the 16 K rows of a chunk (staged in LDS, int8 as bytes)
// give 16 x 16 scores by WMMA; the lanes that hold a key's scores run the
// softmax, P goes through LDS, and every lane accumulates the 4 output dims it
// owns, of all rows, from V. Causality and the window are masked per token.
// Against a VALU tile (one dot product per (row, key)) a 3-token int8 step at
// 131k keys takes 1.22 instead of 2.02 ms: that one was issue-bound.
template <bool Q8, int QN>
__global__ __launch_bounds__(32) void inkling_rel_verify_kernel(Batch b, int nseg,
                                                                float* __restrict__ o_part,
                                                                float* __restrict__ ml_part) {
  constexpr int R = G * QN, NF = D / 16, VD = D / 32;  // rows, K fragments, dims per lane
  constexpr int RL = (R + 1) / 2, RP = (R + 3) / 4 * 4;
  constexpr int EB = Q8 ? 1 : 2;
  constexpr int QP = D + 8;                           // LDS row pitches: padded, so that the
  constexpr int KB = D * EB + 16;                     // rows do not share LDS banks
  constexpr int KPI = 32 * 16 / (D * EB), LPK = 32 / KPI, KL = KC / KPI;  // keys per load
  static_assert(R <= 16);
  __shared__ __attribute__((aligned(16))) uint16_t s_q[R * QP];
  __shared__ __attribute__((aligned(16))) uint8_t s_k[KC * KB];
  __shared__ __attribute__((aligned(16))) float s_p[KC][RP];
  const int kvh = blockIdx.y, seg = blockIdx.z, tid = threadIdx.x;
  const int lk = tid % 16, half = tid / 16, h0 = kvh * G;
  Tile t;
  if (!find_tile<QN>(b, blockIdx.x, t)) return;
  // The wave's LDS accesses are in order; only the compiler has to be kept
  // from moving them across `ordered()` (the pointers escape here, so that the
  // memory clobbers cover them), which also bounds how many of the chunk's
  // loads it hoists: without it QN = 4 spills.
  asm volatile("" :: "s"(&s_q[0]), "s"(&s_k[0]), "s"(&s_p[0][0]) : "memory");
  auto ordered = [] { asm volatile("" ::: "memory"); };
  {
    // Lane (row lk, half of the 16 dims) of every query fragment.
    const int r = min(lk, R - 1);
    const uint16_t* qp = b.q + int64_t(t.u + min(r / G, t.nq - 1)) * b.sq_t +
                         (h0 + r % G) * b.sq_h + half * 8;
    u32x4 w[NF];
#pragma unroll
    for (int f = 0; f < NF; ++f) w[f] = *reinterpret_cast<const u32x4*>(qp + f * 16);
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      uint32_t x[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) x[e] = Q8 ? bf16x2_to_f16x2(w[f][e]) : w[f][e];
      if (lk < R)
        *reinterpret_cast<uint4*>(s_q + lk * QP + f * 16 + half * 8) =
            make_uint4(x[0], x[1], x[2], x[3]);
    }
  }
  ordered();
  float ref[RL], l[RL], acc[R][VD];  // rows 2 e + half; rows r, dims 4 tid + i
#pragma unroll
  for (int e = 0; e < RL; ++e) ref[e] = kNoKeys, l[e] = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int i = 0; i < VD; ++i) acc[r][i] = 0.f;
  u32x4 rk[KL];
  uint32_t rv[KC][EB];
  float ks = 1.f, vs = 1.f;
  uint32_t bias[RL];
  // The next chunk's K (contiguous 16 bytes per lane), its lane key's scale
  // and relative logits; its V is loaded after the current one was used.
  auto fetch_k = [&](int k0) __attribute__((always_inline)) {
    const Pages p = pages(b, t, k0, t.hi, kvh, b.kc, b.sk_b, b.sk_h, b.ksc);
    const int last = min(KC, t.hi - k0) - 1;
#pragma unroll
    for (int n = 0; n < KL; ++n) {
      int slot = p.r0 + min(n * KPI + tid / LPK, last);
      const bool x = slot >= b.block_size;
      if (x) slot -= b.block_size;
      rk[n] = *reinterpret_cast<const u32x4*>((x ? p.b : p.a) +
                                              ((uint32_t)slot * (uint32_t)b.sk_s +
                                               (uint32_t)(tid % LPK) * 16));
    }
    if constexpr (Q8) {
      int slot = p.r0 + min(lk, last);
      const bool x = slot >= b.block_size;
      if (x) slot -= b.block_size;
      ks = (x ? p.sb : p.sa)[int64_t(slot) * b.sc_s];
    }
#pragma unroll
    for (int e = 0; e < RL; ++e) bias[e] = 0;
    const int key = k0 + lk;
    if (k0 + KC > t.key0)
      if (key >= t.key0 && key < t.hi) {
#pragma unroll
        for (int e = 0; e < RL; ++e) {
          const int r = min(2 * e + half, R - 1);
          bias[e] = b.rel[int64_t(t.u + r / G) * b.sr_t + (h0 + r % G) * b.sr_h +
                          (key - t.key0)];
        }
      }
  };
  auto fetch_v = [&](int k0) __attribute__((always_inline)) {
    const Pages p = pages(b, t, k0, t.hi, kvh, b.vc, b.sv_b, b.sv_h, b.vsc);
    const int last = min(KC, t.hi - k0) - 1;
    const uint32_t off = (uint32_t)tid * (VD * EB);
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      int slot = p.r0 + min(k, last);
      const uint8_t* vp = p.a;
      if (slot >= b.block_size) slot -= b.block_size, vp = p.b;
      vp += int64_t(slot) * b.sv_s;
      if constexpr (Q8) {
        rv[k][0] = *reinterpret_cast<const uint32_t*>(vp + off);
      } else {
        const u32x2 w = *reinterpret_cast<const u32x2*>(vp + off);
        rv[k][0] = w.x, rv[k][1] = w.y;
      }
    }
    if constexpr (Q8) {
      int slot = p.r0 + min(lk, last);
      const bool x = slot >= b.block_size;
      if (x) slot -= b.block_size;
      vs = (x ? p.sb : p.sa)[int64_t(slot) * b.sc_s];
    }
  };
  const int k_begin = t.lo + seg * KC, step = nseg * KC;
  if (k_begin < t.hi) {
    fetch_k(k_begin);
    fetch_v(k_begin);
  }
  for (int k0 = k_begin; k0 < t.hi; k0 += step) {
#pragma unroll
    for (int n = 0; n < KL; ++n)
      *reinterpret_cast<uint4*>(s_k + (n * KPI + tid / LPK) * KB + tid % LPK * 16) =
          make_uint4(rk[n].x, rk[n].y, rk[n].z, rk[n].w);
    ordered();
    const float cs = Q8 ? b.scale_log2 * ks : b.scale_log2, v_scale = vs;
    uint32_t lb[RL];
#pragma unroll
    for (int e = 0; e < RL; ++e) lb[e] = bias[e];
    if (k0 + step < t.hi) fetch_k(k0 + step);
    f32x8 sc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      const bf16x16 qf = load_frag_bf16(s_q + f * 16, QP, lk < R ? lk : 0);
      bf16x16 kf;
      if constexpr (Q8) {
        const uint4 w = *reinterpret_cast<const uint4*>(s_k + lk * KB + f * 16);
        uint32_t o[2][4];
        q8x8_to_f16(w.x, w.y, o[0]);
        q8x8_to_f16(w.z, w.w, o[1]);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          kf[2 * e] = static_cast<short>(o[e / 4][e % 4] & 0xffffu);
          kf[2 * e + 1] = static_cast<short>(o[e / 4][e % 4] >> 16);
        }
        sc = wmma_f16(qf, kf, sc);
      } else {
        kf = load_frag_bf16(reinterpret_cast<const uint16_t*>(s_k) + f * 16, KB / 2, tid);
        sc = wmma_bf16(qf, kf, sc);
      }
      if (f % 2 == 1) ordered();
    }
    // Lane (key lk, half) holds the scores of rows 2 e + half.
    const int key = k0 + lk;
    float x[RL];
    bool grow = false;
#pragma unroll
    for (int e = 0; e < RL; ++e) {
      const int r = 2 * e + half, qi = r / G;
      // A key past the token is a negative distance, a huge unsigned.
      const bool seen = r < R && qi < t.nq && key < t.hi &&
                        unsigned(t.pos + qi - key) <=
                            unsigned(b.win_left >= 0 ? b.win_left : INT_MAX);
      x[e] = seen ? sc[e] * cs + bf16_lo(lb[e]) * kLOG2E : -INFINITY;
      grow |= x[e] > ref[e] + kGrow;
    }
    if (__builtin_amdgcn_ballot_w32(grow) != 0) {
#pragma unroll
      for (int e = 0; e < RL; ++e) {
        const float m = fmaxf(ref[e], group_max<16>(x[e]));
        const float a = __builtin_amdgcn_exp2f(ref[e] - m), o = xor_shuffle_f32(a, 16);
        ref[e] = m;
        l[e] *= a;
#pragma unroll
        for (int i = 0; i < VD; ++i) {
          acc[2 * e][i] *= half ? o : a;
          if (2 * e + 1 < R) acc[2 * e + 1][i] *= half ? a : o;
        }
      }
    }
#pragma unroll
    for (int e = 0; e < RL; ++e) {
      const float p = __builtin_amdgcn_exp2f(x[e] - ref[e]);
      l[e] += p;
      if (2 * e + half < RP) s_p[lk][2 * e + half] = Q8 ? p * v_scale : p;
    }
    ordered();
#pragma unroll
    for (int k = 0; k < KC; ++k) {
      float p[RP];
#pragma unroll
      for (int c = 0; c < RP; c += 4) {
        const float4 w = *reinterpret_cast<const float4*>(&s_p[k][c]);
        p[c] = w.x, p[c + 1] = w.y, p[c + 2] = w.z, p[c + 3] = w.w;
      }
      uint32_t o[4];
      if constexpr (Q8) q8x8_to_f16(rv[k][0], 0, o);
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int i = 0; i < VD; i += 2) {
          if constexpr (Q8) {
            const f16x2 v = as_f16x2(o[i / 2]);
            acc[r][i] = __builtin_fmaf(p[r], (float)v.x, acc[r][i]);
            acc[r][i + 1] = __builtin_fmaf(p[r], (float)v.y, acc[r][i + 1]);
          } else {
            acc[r][i] += p[r] * bf16_lo(rv[k][i / 2]);
            acc[r][i + 1] += p[r] * bf16_hi(rv[k][i / 2]);
          }
        }
      // Finish four keys before the next four's P loads.
      if (k % 4 == 3) {
#pragma unroll
        for (int r = 0; r < R; ++r)
          asm volatile("" : "+v"(acc[r][0]), "+v"(acc[r][1]), "+v"(acc[r][2]), "+v"(acc[r][3])
                       :: "memory");
      }
    }
    if (k0 + step < t.hi) fetch_v(k0 + step);
  }
  auto partial = [&](int r) {
    return (int64_t(t.u + r / G) + int64_t(seg) * b.T) * b.hq + h0 + r % G;
  };
#pragma unroll
  for (int e = 0; e < RL; ++e) {
    const float ls = group_sum<16>(l[e]);
    const int r = 2 * e + half;
    if (lk == 0 && r < R && r / G < t.nq) {
      ml_part[2 * partial(r)] = ref[e];
      ml_part[2 * partial(r) + 1] = ls;
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
    if (r / G < t.nq)
      *reinterpret_cast<float4*>(o_part + partial(r) * D + tid * VD) =
          make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
}

// Combines the split-KV partials of one (token, head) and applies the output
// gate and attention sink: one block per (token, head), one thread per dim.
__global__ __launch_bounds__(D) void inkling_rel_verify_reduce_kernel(
    uint16_t* __restrict__ out, const float* __restrict__ o_part,
    const float* __restrict__ ml_part, Batch b, int nseg) {
  const int u = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  uint16_t* o = out + (int64_t(u) * b.hq + h) * D + d;
  if (u >= b.cu_q[b.nreq]) {  // a row of no request
    *o = 0;
    return;
  }
  float m = kNoKeys, l = 0.f, acc = 0.f;
#pragma unroll 8
  for (int s = 0; s < nseg; ++s)
    m = fmaxf(m, ml_part[2 * ((int64_t(u) + int64_t(s) * b.T) * b.hq + h)]);
#pragma unroll 8
  for (int s = 0; s < nseg; ++s) {
    const int64_t r = (int64_t(u) + int64_t(s) * b.T) * b.hq + h;
    const float w = __builtin_amdgcn_exp2f(ml_part[2 * r] - m);
    l += w * ml_part[2 * r + 1];
    acc += w * o_part[r * D + d];
  }
  const float f = sigmoid((m + log2f(l)) * kLN2 - bf16_lo(b.sink[u * b.ss_t + h])) / l;
  *o = static_cast<uint16_t>(
      pack_bf16x2(acc * f * sigmoid(bf16_lo(b.gate[u * b.sg_t + h * D + d])), 0.f) & 0xffffu);
}

}  // namespace

// q: [T, Hq, 128] bf16 with GQA 3 (row-strided); key_cache, value_cache:
// [num_blocks, block_size, Hkv, 128] bf16 or int8 views (int8 with k_scale /
// v_scale, as attn_decode_bf16); rel_logits: the sheared bf16 table
// [T + 128, Hq, e_pad] (row u holds the token-u logits, column key - key0);
// gate [T, Hq * 128] and sinks [T, Hq] bf16 (row-strided); out [T, Hq * 128]
// bf16 contiguous, written for every row (tokens of no request zeroed).
// cu_seqlens_q [Nreq + 1], seq_lens [Nreq] int32; block_table [Nreq,
// max_blocks] int32; window_left >= 0 restricts the token at p to keys
// p - window_left .. p. q_rows: 1, or up to 3 or 5 query tokens of a request
// per block, which read its keys once and split them over num_splits blocks
// per KV head (q_rows 1 does not split).
void inkling_rel_attn_decode(torch::Tensor& out, const torch::Tensor& q,
                             const torch::Tensor& key_cache,
                             const torch::Tensor& value_cache,
                             const std::optional<torch::Tensor>& k_scale,
                             const std::optional<torch::Tensor>& v_scale,
                             const torch::Tensor& rel_logits,
                             const torch::Tensor& gate,
                             const torch::Tensor& sinks,
                             const torch::Tensor& cu_seqlens_q,
                             const torch::Tensor& seq_lens,
                             const torch::Tensor& block_table, double scale,
                             int64_t num_splits, int64_t window_left, int64_t q_rows) {
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && q.scalar_type() == at::kBFloat16 &&
                  q.stride(2) == 1 && q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0,
              "q must be bf16 [T, Hq, 128] with 16-byte aligned rows");
  const int64_t T = q.size(0), hq = q.size(1), hkv = key_cache.size(2);
  TORCH_CHECK(hq == G * hkv, "GQA 3 only");
  const InklingKV kv = inkling_kv(key_cache, value_cache, k_scale, v_scale);
  TORCH_CHECK(key_cache.size(1) >= KC, "cache blocks of at least 16 tokens");
  const int64_t e_pad = rel_logits.size(2);
  TORCH_CHECK(rel_logits.dim() == 3 && rel_logits.scalar_type() == at::kBFloat16 &&
                  rel_logits.size(0) >= T + 128 && rel_logits.size(1) == hq &&
                  rel_logits.stride(2) == 1 && rel_logits.stride(0) % 8 == 0 &&
                  rel_logits.stride(1) % 8 == 0,
              "rel_logits must be bf16 [T + 128, Hq, e_pad] with 16-byte aligned rows");
  TORCH_CHECK(gate.dim() == 2 && gate.scalar_type() == at::kBFloat16 && gate.size(0) >= T &&
                  gate.size(1) == hq * 128 && gate.stride(1) == 1 && gate.stride(0) % 8 == 0,
              "gate must be bf16 [T, Hq * 128] with 16-byte aligned rows");
  TORCH_CHECK(sinks.dim() == 2 && sinks.scalar_type() == at::kBFloat16 &&
                  sinks.size(0) >= T && sinks.size(1) == hq && sinks.stride(1) == 1,
              "sinks must be bf16 [T, Hq]");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) >= T &&
                  out.size(1) == hq * 128 && out.is_contiguous(),
              "out must be contiguous bf16 [T, Hq * 128]");
  const int64_t nreq = cu_seqlens_q.numel() - 1;
  for (const auto* t : {&cu_seqlens_q, &seq_lens, &block_table})
    TORCH_CHECK(t->scalar_type() == at::kInt, "int32 metadata");
  TORCH_CHECK(seq_lens.numel() >= nreq && block_table.stride(1) == 1 &&
                  block_table.size(0) >= nreq,
              "metadata must cover the requests");
  TORCH_CHECK(num_splits >= 1 && num_splits <= 64, "num_splits in [1, 64]");
  TORCH_CHECK(q_rows == 1 || q_rows == 3 || q_rows == 5, "q_rows is 1, 3 or 5");
  if (T == 0 || nreq <= 0) return;
  const int64_t eb = kv.q8 ? 1 : 2;
  Batch b;
  b.q = reinterpret_cast<const uint16_t*>(q.data_ptr());
  b.rel = reinterpret_cast<const uint16_t*>(rel_logits.data_ptr());
  b.gate = reinterpret_cast<const uint16_t*>(gate.data_ptr());
  b.sink = reinterpret_cast<const uint16_t*>(sinks.data_ptr());
  b.kc = reinterpret_cast<const uint8_t*>(key_cache.data_ptr());
  b.vc = reinterpret_cast<const uint8_t*>(value_cache.data_ptr());
  b.ksc = kv.k_scale, b.vsc = kv.v_scale;
  b.cu_q = cu_seqlens_q.data_ptr<int32_t>();
  b.seq_lens = seq_lens.data_ptr<int32_t>();
  b.block_table = block_table.data_ptr<int32_t>();
  b.sq_t = q.stride(0), b.sq_h = q.stride(1);
  b.sr_t = rel_logits.stride(0), b.sr_h = rel_logits.stride(1);
  b.sg_t = gate.stride(0), b.ss_t = sinks.stride(0), b.bt = block_table.stride(0);
  b.sk_b = key_cache.stride(0) * eb, b.sk_s = key_cache.stride(1) * eb;
  b.sk_h = key_cache.stride(2) * eb;
  b.sv_b = value_cache.stride(0) * eb, b.sv_s = value_cache.stride(1) * eb;
  b.sv_h = value_cache.stride(2) * eb;
  b.sc_b = kv.sc_b, b.sc_s = kv.sc_s, b.sc_h = kv.sc_h;
  b.nreq = static_cast<int>(nreq), b.T = static_cast<int>(T), b.hq = static_cast<int>(hq);
  b.e_pad = static_cast<int>(e_pad), b.block_size = static_cast<int>(key_cache.size(1));
  b.win_left = static_cast<int>(window_left);
  b.scale_log2 = static_cast<float>(scale) * kLOG2E;
  auto* o = reinterpret_cast<uint16_t*>(out.data_ptr());
  auto stream = at::hip::getCurrentHIPStream();
  if (q_rows == 1) {
    // Lanes per key. bf16 full layers load whole 256-byte rows (16); over a
    // window's 1024 keys, and for int8 (twice the keys per byte), the fewer
    // instructions per key of 8 win.
    auto kernel = kv.q8              ? inkling_rel_decode_kernel<true, 8>
                  : window_left >= 0 ? inkling_rel_decode_kernel<false, 8>
                                     : inkling_rel_decode_kernel<false, 16>;
    kernel<<<dim3(T, hkv), 32 * kNW, 0, stream>>>(o, b);
    return;
  }
  auto f32 = q.options().dtype(at::kFloat);
  torch::Tensor o_part = torch::empty({T, num_splits, hq, 128}, f32);
  torch::Tensor ml_part = torch::empty({T, num_splits, hq, 2}, f32);
  // A bound on the tiles of q_rows tokens: each request may end in a partial
  // one, and the lengths are on the device.
  const dim3 grid((T + nreq * (q_rows - 1)) / q_rows, hkv, num_splits);
  auto kernel = kv.q8 ? (q_rows == 3 ? inkling_rel_verify_kernel<true, 3>
                                     : inkling_rel_verify_kernel<true, 5>)
                      : (q_rows == 3 ? inkling_rel_verify_kernel<false, 3>
                                     : inkling_rel_verify_kernel<false, 5>);
  kernel<<<grid, 32, 0, stream>>>(b, static_cast<int>(num_splits), o_part.data_ptr<float>(),
                                  ml_part.data_ptr<float>());
  inkling_rel_verify_reduce_kernel<<<dim3(T, hq), D, 0, stream>>>(
      o, o_part.data_ptr<float>(), ml_part.data_ptr<float>(), b, static_cast<int>(num_splits));
}

}  // namespace strix
