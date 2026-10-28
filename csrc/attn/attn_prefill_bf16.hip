// SPDX-License-Identifier: MIT
// Causal paged attention for prefill: head_dim D = 64 (GQA 4, LFM2.5), 256
// (GQA 6 or 8) or 512 (GQA 8, Gemma-4 full-attention layers), bf16 Q/KV, any
// KV block size.
// Varlen over cu_seqlens_q; the queries of a sequence are its last q_len
// positions of seq_lens[n] keys.
//
// One block owns 16 query tokens of one sequence and HPB query heads of one
// KV head (G / HPB blocks share each K/V tile stream). Each head is W = D/128
// waves that split head_dim: wave `part` holds Q[:, part*128 : +128] (64
// VGPRs) and the matching slice of O (64 VGPRs); partial scores are summed
// over the W waves through LDS.
//
// Everything is computed transposed, S^T = K Q^T and O^T = V^T P^T, so in the
// WMMA accumulator layout a lane owns one query (column): softmax row stats
// and the O rescale are per-lane scalars, and P^T becomes the B operand of
// the next WMMA through one cross-half lane swap instead of an LDS transpose.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <optional>

#include "../strix_common.h"

namespace strix {
namespace {

// Head dims per wave: a head of D <= 128 is one wave; larger heads split
// into W = D / 128 waves.
constexpr int dims_per_wave(int d) { return d < 128 ? d : 128; }
// 16-wave blocks (4 per SIMD) at 200+ VGPRs fit one block per WGP, which
// leaves every barrier bubble unfilled; capping VGPRs at 192 (8 waves/SIMD)
// lets a second block in. 32-wave blocks need the same cap to launch at
// all. Smaller blocks already get two.
constexpr int waves_target(int waves) { return waves >= 16 ? 8 : 1; }

// Per-shape code-generation choices, keyed by (head_dim, GQA, int8 KV) - the
// geometry, not the model: a new model on a known shape inherits its row,
// and a new shape runs the defaults until swept. Each flag only changes how
// the same math is scheduled; all shapes stay bit-compatible with the tests.
//   mask_skip: skip the causal mask on steps wholly below the tile's first
//              query (duplicates the softmax loop, so it costs VGPRs)
//   hoist:     precompute each lane's loop-invariant slot-offset products
//              (fewer VALU per step, ~6 more VGPRs per staged load)
// Measured at an 8k chunk on 32k context.
template <int D, int G, bool Q8>
struct PrefillTune {
  static constexpr bool mask_skip = true;
  static constexpr bool hoist = false;
};
// ms at an 8k chunk @ 32k, (none / hoist / mask / both), written by
// tools/tune_prefill.py --write; the row picks the
// fastest. Unlisted shapes use the default (mask only).
#define STRIX_TUNE(D, G, Q8, M, H)        \
  template <>                             \
  struct PrefillTune<D, G, Q8> {          \
    static constexpr bool mask_skip = M;  \
    static constexpr bool hoist = H;      \
  };
STRIX_TUNE(256, 6, false, true, true)  // 234.2 / 248.4 / 232.2 / 228.6
STRIX_TUNE(256, 6, true, true, false)  // 260.1 / 262.8 / 243.6 / 359.6
STRIX_TUNE(256, 8, false, true, true)  // 157.5 / 162.8 / 155.4 / 151.8
STRIX_TUNE(256, 8, true, true, true)  // 165.8 / 188.3 / 163.2 / 158.3
STRIX_TUNE(512, 8, false, true, true)  // 699.0 / 800.2 / 692.1 / 690.1
STRIX_TUNE(512, 8, true, true, false)  // 770.0 / 882.3 / 740.8 / 762.7
STRIX_TUNE(64, 4, false, true, true)  // 113.5 / 114.1 / 107.8 / 107.0
STRIX_TUNE(64, 4, true, true, false)  // 122.4 / 123.5 / 116.2 / 117.7
STRIX_TUNE(256, 2, false, true, true)  // 13.8 / 12.9 / 12.8 / 12.3
STRIX_TUNE(256, 2, true, true, false)  // 13.8 / 13.8 / 13.7 / 13.8
#undef STRIX_TUNE
constexpr int kBM = 16;     // query tokens per block


// G query heads per KV head, HPB of them per block (G / HPB blocks share a
// K/V tile stream), BN keys per step. The partial-S exchange buffer grows
// with HPB * BN; XT = _Float16 halves it (partials are pre-scaled to the
// log2 softmax domain, |S| << 65504). ALIAS puts the exchange buffer over
// the K tile (one more barrier per step) so 8 heads x 32 keys fit in 64 KB.
//
// Q8: K/V arrive as int8 with one fp32 scale per (key, head) (vLLM's
// int8_per_token_head cache). The bytes are widened to bf16 on the way into
// LDS *without* the scale, which is exact (|v| <= 127); the scale is applied
// once per (query, key) instead - to the score for K, to the P fragment for
// V - so the per-element cost is one convert, not a convert and a multiply.
template <int D, int G, int HPB, int BN, typename XT, bool ALIAS, bool Q8>
__global__ __launch_bounds__(HPB * (D / dims_per_wave(D)) * kWave)
__attribute__((amdgpu_waves_per_eu(waves_target(HPB * (D / dims_per_wave(D)))))) void
attn_prefill_kernel(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    const int8_t* __restrict__ k8, const int8_t* __restrict__ v8,
    const float* __restrict__ ksc, const float* __restrict__ vsc,
    const int32_t* __restrict__ cu_q, const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ tile_off, int nseq, int64_t bt_stride,
    int block_size, int64_t sq_t, int64_t sq_h, int64_t so_t, int64_t so_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s,
    int64_t sv_h, int64_t sc_b, int64_t sc_s, int64_t sc_h, float scale_log2,
    int win_left, int nseg, float* __restrict__ o_part,
    float* __restrict__ ml_part) {
  constexpr int kHalf = dims_per_wave(D);
  constexpr int kHD = D, W = D / kHalf;  // waves per head
  constexpr int kPK = kHD + 8;     // LDS pitch of the K tile [key][d]
  constexpr int kThreads = HPB * W * kWave;
  constexpr int NS = BN / 16;      // 16-key subtiles per step
  constexpr int kPV = BN + 8;      // LDS pitch of the V^T tile [d][key]
  typedef XT xbuf_t[W * HPB][NS][kWave][8];
  constexpr int kKBytes = BN * kPK * 2, kXBytes = sizeof(xbuf_t);
  __shared__ __attribute__((aligned(16)))
  uint8_t s_kx[ALIAS ? (kKBytes > kXBytes ? kKBytes : kXBytes) : kKBytes + kXBytes];
  __shared__ __attribute__((aligned(16))) uint16_t s_vt[kHD * kPV];
  uint16_t* s_k = reinterpret_cast<uint16_t*>(s_kx);
  xbuf_t& s_x = *reinterpret_cast<xbuf_t*>(s_kx + (ALIAS ? 0 : kKBytes));
  // Q8 only: per-key dequant scales of the current K/V tile, staged next to
  // the tiles so the score and P loops read them from LDS.
  __shared__ float s_ks[Q8 ? BN : 1], s_vs[Q8 ? BN : 1];

  // Last tiles first: causal tiles grow with position, so the longest blocks
  // start early and the short ones fill the tail.
  const int kvh = blockIdx.x / (G / HPB), hgroup = blockIdx.x % (G / HPB);
  const int tile = tile_off[nseq] - 1 - int(blockIdx.y);
  if (tile < 0) return;
  int n = 0;
  for (int i = 1; i < nseq; ++i) n += tile_off[i] <= tile;

  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int pair = wave / W, half = wave % W, lo = lane % 16, hi = lane / 16;
  const int h = kvh * G + hgroup * HPB + pair;
  const int q0 = cu_q[n], q_len = cu_q[n + 1] - q0;
  const int kv_len = seq_lens[n], context = kv_len - q_len;
  const int local0 = (tile - tile_off[n]) * kBM;
  const int rows = min(kBM, q_len - local0);
  // Invalid rows reuse the last valid row's position (never written back).
  const int my_row = min(lo, rows - 1);
  const int my_pos = context + local0 + my_row;
  const int last_pos = context + local0 + rows - 1;
  const int32_t* bt = block_table + int64_t(n) * bt_stride;

  // Q^T B fragments: row = query lo, k = this half's dims.
  bf16x16 qf[kHalf / 16];
  {
    const uint16_t* qrow =
        q + int64_t(q0 + local0 + my_row) * sq_t + h * sq_h + half * kHalf;
#pragma unroll
    for (int kk = 0; kk < kHalf / 16; ++kk) qf[kk] = load_frag_bf16(qrow + kk * 16, 0, 0);
    // Q8 runs the WMMAs in fp16 (the K/V bytes widen to fp16 with one perm
    // per pair, see q8x8_to_f16): Q is converted once here. fp16 carries 3
    // more mantissa bits than the bf16 it comes from, so this only rounds
    // values outside fp16's range (|q| > 65504), which normed queries are not.
    if constexpr (Q8) {
#pragma unroll
      for (int kk = 0; kk < kHalf / 16; ++kk)
#pragma unroll
        for (int i = 0; i < 16; ++i)
          qf[kk][i] = static_cast<short>(
              f16_bits(__uint_as_float(uint32_t(uint16_t(qf[kk][i])) << 16)));
    }
  }

  f32x8 o_acc[kHalf / 16];
#pragma unroll
  for (int i = 0; i < kHalf / 16; ++i) o_acc[i] = f32x8{};
  float m_run = -INFINITY, l_run = 0.f;
  // perm selector for the P pair: low half from key 2e (this lane if hi == 0).
  // (bf16 rounds into the high half of each word, fp16 into the low half.)
  const uint32_t p_sel = Q8 ? (hi ? 0x01000504u : 0x05040100u)
                            : (hi ? 0x03020706u : 0x07060302u);

  constexpr int kChunks = BN * kHD / 8;
  constexpr int kLoads = (kChunks + kThreads - 1) / kThreads;
  typedef uint32_t stage_t __attribute__((ext_vector_type(4 * kLoads)));
  stage_t rk = {}, rv = {};
  // Q8 stages 8 bytes per chunk (two words); the rest of the vector is unused.
  float rks[kLoads] = {}, rvs[kLoads] = {};
  auto chunk_addr = [&](int key, int64_t sb, int64_t ss, int64_t sh) {
    const int64_t blk = bt[key / block_size];
    return blk * sb + int64_t(key % block_size) * ss + kvh * sh;
  };
  // K chunks are key-major (row-major tile). V chunks are key-fastest so the
  // transposed 2-byte stores of a wave land side by side (key-major mapping
  // was a 32-way LDS bank conflict and half the kernel time).
  // A step's BN keys span at most two cache blocks when BN <= block_size
  // (every served config): their bases (64-bit) are computed once per step
  // in scalar registers, and each lane only picks one and adds a 32-bit
  // slot offset (the host checks it fits). The per-lane 64-bit
  // `bt[key / bs] * stride` was ~150 VALU instructions per step.
  const bool two_blocks = BN <= block_size;
  const int ss_k = int(sk_s), ss_v = int(sv_s), ss_c = int(sc_s);
  constexpr bool kHoist = PrefillTune<D, G, Q8>::hoist;
  // kHoist: each lane's key offsets within a step never change, so their
  // stride products are taken once here and a step only adds scalars.
  int lk_off[kLoads], lv_off[kLoads], lk_mem[kLoads], lv_mem[kLoads], lk_sc[kLoads],
      lv_sc[kLoads];
#pragma unroll
  for (int i = 0; i < kLoads; ++i) {
    const int ch = tid + i * kThreads;
    lk_off[i] = ch / (kHD / 8);
    lv_off[i] = ch % BN;
    lk_mem[i] = lk_off[i] * ss_k;
    lv_mem[i] = lv_off[i] * ss_v;
    lk_sc[i] = lk_off[i] * ss_c;
    lv_sc[i] = lv_off[i] * ss_c;
  }
#define STRIX_FETCH(K0)                                                      \
  const int fb0 = (K0) / block_size, fr0 = (K0) - fb0 * block_size;          \
  const bool fspan = two_blocks && fr0 + BN > block_size;                    \
  const int64_t fbA = bt[fb0], fbB = fspan ? bt[fb0 + 1] : fbA;              \
  const int64_t fkA = fbA * sk_b + kvh * sk_h, fkB = fbB * sk_b + kvh * sk_h; \
  const int64_t fvA = fbA * sv_b + kvh * sv_h, fvB = fbB * sv_b + kvh * sv_h; \
  const int64_t fcA = fbA * sc_b + kvh * sc_h, fcB = fbB * sc_b + kvh * sc_h; \
  /* kHoist: slot fr0's offset folded in, B bases shifted back one block */  \
  const int64_t fkA2 = fkA + int64_t(fr0) * ss_k,                            \
                fkB2 = fkB + int64_t(fr0 - block_size) * ss_k;               \
  const int64_t fvA2 = fvA + int64_t(fr0) * ss_v,                            \
                fvB2 = fvB + int64_t(fr0 - block_size) * ss_v;               \
  const int64_t fcA2 = fcA + int64_t(fr0) * ss_c,                            \
                fcB2 = fcB + int64_t(fr0 - block_size) * ss_c;               \
  _Pragma("unroll") for (int i = 0; i < kLoads; ++i) {                       \
    const int ch = tid + i * kThreads;                                       \
    uint4 xk = {}, xv = {};                                                  \
    if (ch < kChunks) {                                                      \
      const int koff = ch / (kHD / 8), kd = (ch % (kHD / 8)) * 8;            \
      const int voff = ch % BN, vd = (ch / BN) * 8;                          \
      const int kkey = (K0) + koff, vkey = (K0) + voff;                      \
      int64_t ka, va, kca, vca;                                              \
      if (kHoist && two_blocks) {                                            \
        const bool kn = fr0 + lk_off[i] >= block_size;                       \
        const bool vn = fr0 + lv_off[i] >= block_size;                       \
        ka = (kn ? fkB2 : fkA2) + lk_mem[i];                                 \
        va = (vn ? fvB2 : fvA2) + lv_mem[i];                                 \
        kca = (kn ? fcB2 : fcA2) + lk_sc[i];                                 \
        vca = (vn ? fcB2 : fcA2) + lv_sc[i];                                 \
      } else if (two_blocks) {                                               \
        const int kr = fr0 + koff, vr = fr0 + voff;                          \
        const bool kn = kr >= block_size, vn = vr >= block_size;             \
        const int ks2 = kn ? kr - block_size : kr, vs2 = vn ? vr - block_size : vr; \
        ka = (kn ? fkB : fkA) + ks2 * ss_k;                                  \
        va = (vn ? fvB : fvA) + vs2 * ss_v;                                  \
        kca = (kn ? fcB : fcA) + ks2 * ss_c;                                 \
        vca = (vn ? fcB : fcA) + vs2 * ss_c;                                 \
      } else {                                                               \
        ka = chunk_addr(kkey, sk_b, sk_s, sk_h);                             \
        va = chunk_addr(vkey, sv_b, sv_s, sv_h);                             \
        kca = chunk_addr(kkey, sc_b, sc_s, sc_h);                            \
        vca = chunk_addr(vkey, sc_b, sc_s, sc_h);                            \
      }                                                                      \
      if constexpr (Q8) {                                                    \
        if (kkey < kv_len) {                                                 \
          const int8_t* p = k8 + ka + kd;                                    \
          xk.x = *reinterpret_cast<const uint32_t*>(p);                      \
          xk.y = *reinterpret_cast<const uint32_t*>(p + 4);                  \
          rks[i] = ksc[(size_t)kca];                                         \
        }                                                                    \
        if (vkey < kv_len) {                                                 \
          const int8_t* p = v8 + va + vd;                                    \
          xv.x = *reinterpret_cast<const uint32_t*>(p);                      \
          xv.y = *reinterpret_cast<const uint32_t*>(p + 4);                  \
          rvs[i] = vsc[(size_t)vca];                                         \
        }                                                                    \
      } else {                                                               \
        (void)kca;                                                           \
        (void)vca;                                                           \
        if (kkey < kv_len) xk = *reinterpret_cast<const uint4*>(kc + ka + kd); \
        if (vkey < kv_len) xv = *reinterpret_cast<const uint4*>(vc + va + vd); \
      }                                                                      \
    }                                                                        \
    rk[4 * i] = xk.x, rk[4 * i + 1] = xk.y, rk[4 * i + 2] = xk.z, rk[4 * i + 3] = xk.w; \
    rv[4 * i] = xv.x, rv[4 * i + 1] = xv.y, rv[4 * i + 2] = xv.z, rv[4 * i + 3] = xv.w; \
  }

  // Split-KV: segment blockIdx.z covers keys [k_begin, k_end) of the keys
  // this tile needs; with nseg == 1 it is all of them.
  // Sliding window (vLLM's (window - 1, 0)): a query at p sees keys
  // p - win_left .. p, so the block starts at its first row's window.
  const int k_lo = win_left >= 0 ? max(0, context + local0 - win_left) : 0;
  const int seg = blockIdx.z, keys_needed = last_pos + 1;
  const int seg_len = ((keys_needed - k_lo + nseg - 1) / nseg + BN - 1) / BN * BN;
  const int k_begin = k_lo + seg * seg_len;
  const int k_end = min(keys_needed, k_begin + seg_len);
  const int n_steps = k_end > k_begin ? (k_end - k_begin + BN - 1) / BN : 0;
  if (n_steps > 0) {
    STRIX_FETCH(k_begin)
  }
  for (int step = 0; step < n_steps; ++step) {
    const int k0 = k_begin + step * BN;
    barrier();  // previous step is done with s_k / s_vt / s_x
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      const int ch = tid + i * kThreads;
      if (ch < kChunks) {
        const int kkey = ch / (kHD / 8), kd = (ch % (kHD / 8)) * 8;
        const int vkey = ch % BN, vd = (ch / BN) * 8;
        uint32_t kk[4] = {rk[4 * i], rk[4 * i + 1], rk[4 * i + 2], rk[4 * i + 3]};
        uint32_t vv[4] = {rv[4 * i], rv[4 * i + 1], rv[4 * i + 2], rv[4 * i + 3]};
        if constexpr (Q8) {
          // Two words of int8 each widen to four words of fp16; the bytes
          // past the first two words are the zero padding of the stage.
          uint32_t d0[4], d1[4];
          q8x8_to_f16(kk[0], kk[1], d0);
          q8x8_to_f16(vv[0], vv[1], d1);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            kk[q] = d0[q];
            vv[q] = d1[q];
          }
          // One scale per key: the first d-chunk of each key commits it, so
          // the score and P loops can read it from LDS.
          if (kd == 0) s_ks[kkey] = rks[i];
          if (vd == 0) s_vs[vkey] = rvs[i];
        }
        *reinterpret_cast<uint4*>(s_k + kkey * kPK + kd) =
            make_uint4(kk[0], kk[1], kk[2], kk[3]);
        const uint32_t w[4] = {vv[0], vv[1], vv[2], vv[3]};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          s_vt[(vd + 2 * e) * kPV + vkey] = static_cast<uint16_t>(w[e] & 0xffffu);
          s_vt[(vd + 2 * e + 1) * kPV + vkey] = static_cast<uint16_t>(w[e] >> 16);
        }
      }
    }
    barrier();
    if (step + 1 < n_steps) {
      STRIX_FETCH(k0 + BN)
    }

    // Partial S^T over this wave's half of head_dim, per 16-key subtile.
    f32x8 s[NS];
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      s[j] = f32x8{};
#pragma unroll
      for (int kk = 0; kk < kHalf / 16; ++kk) {
        const bf16x16 kf =
            load_frag_bf16(s_k + j * 16 * kPK + half * kHalf + kk * 16, kPK, lane);
        s[j] = Q8 ? wmma_f16(kf, qf[kk], s[j]) : wmma_bf16(kf, qf[kk], s[j]);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e)
        s[j][e] *= Q8 ? scale_log2 * s_ks[16 * j + 2 * e + hi] : scale_log2;
    }
    if constexpr (W > 1) {  // one-wave heads have their full scores already
      if constexpr (ALIAS) barrier();  // every wave is done reading s_k
#pragma unroll
      for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) s_x[wave][j][lane][e] = static_cast<XT>(s[j][e]);
      barrier();
    }

    // Online softmax for query lo; element e of subtile j is key
    // k0 + 16 j + 2e + hi.
    // A step wholly below the tile's first query, and (with a window) wholly
    // inside its last query's window, needs no mask - every step but the
    // first and last one or two. The block-uniform branch drops ~2 compares
    // and a select per element.
    constexpr bool kMaskSkip = PrefillTune<D, G, Q8>::mask_skip;
    const bool full_step = kMaskSkip && k0 + BN - 1 <= context + local0 &&
                           (win_left < 0 || last_pos - k0 <= win_left);
    float m_loc = -INFINITY;
    if (full_step) {
#pragma unroll
      for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
#pragma unroll
          for (int o = 1; o < W; ++o)
            s[j][e] += static_cast<float>(s_x[pair * W + (half + o) % W][j][lane][e]);
          m_loc = fmaxf(m_loc, s[j][e]);
        }
    } else {
#pragma unroll
      for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int key = k0 + 16 * j + 2 * e + hi;
          float v = s[j][e];
#pragma unroll
          for (int o = 1; o < W; ++o)
            v += static_cast<float>(s_x[pair * W + (half + o) % W][j][lane][e]);
          const bool seen = key <= my_pos && (win_left < 0 || my_pos - key <= win_left);
          s[j][e] = seen ? v : -INFINITY;
          m_loc = fmaxf(m_loc, s[j][e]);
        }
    }
    m_loc = fmaxf(m_loc, xor_shuffle_f32(m_loc, 16));
    const float m_new = fmaxf(m_run, m_loc);
    // A row may see no key in this split-KV segment yet: keep p = 0, not NaN.
    const float m_ref = m_new == -INFINITY ? 0.f : m_new;
    // Raw v_exp_f32 (no denormal fix-up; a softmax weight below 2^-126 is
    // 0 either way): exp2f's range check was 3 extra VALU ops per element.
    const float alpha = __builtin_amdgcn_exp2f(m_run - m_ref);
    // Deep in a long context the running max rarely moves, so alpha is
    // exactly 1 for the whole wave on most steps: skip the 64-multiply
    // rescale then.
    if (__any(alpha != 1.f)) {
#pragma unroll
      for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
        for (int e = 0; e < 8; ++e) o_acc[dt][e] *= alpha;
    }
    // One 16-key subtile at a time keeps a single P^T fragment live.
    float l_loc = 0.f;
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      bf16x16 pf;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float p = __builtin_amdgcn_exp2f(s[j][e] - m_ref);
        l_loc += p;
        // P^T B fragment: row = query lo, 16 keys; this half holds key
        // 2e + hi, the other half 2e + (1 - hi). Each lane rounds its own
        // value once and swaps the rounded word; one perm with a per-half
        // selector puts key 2e low and 2e + 1 high (was two roundings and
        // two selects per element).
        float pv = p;
        // V's per-key scale rides on P so the accumulator needs no
        // per-element pass, and l stays the unweighted sum.
        if constexpr (Q8) pv *= s_vs[16 * j + 2 * e + hi];
        // Q8: P in fp16 for the fp16 WMMA, in the low half of the word.
        const uint32_t mine = Q8 ? f16_bits(pv) : round_bf16_hi(pv);
        const uint32_t other = __float_as_uint(xor_shuffle_f32(__uint_as_float(mine), 16));
        const uint32_t pk = __builtin_amdgcn_perm(other, mine, p_sel);
        pf[2 * e] = static_cast<short>(pk & 0xffffu);
        pf[2 * e + 1] = static_cast<short>(pk >> 16);
      }
#pragma unroll
      for (int dt = 0; dt < kHalf / 16; ++dt) {
        const bf16x16 vf =
            load_frag_bf16(s_vt + (half * kHalf + dt * 16) * kPV + j * 16, kPV, lane);
        o_acc[dt] = Q8 ? wmma_f16(vf, pf, o_acc[dt]) : wmma_bf16(vf, pf, o_acc[dt]);
      }
    }
    l_loc += xor_shuffle_f32(l_loc, 16);
    l_run = l_run * alpha + l_loc;
    m_run = m_new;
  }
#undef STRIX_FETCH

  // O^T[d][query lo]: element e of tile dt is d = dt*16 + 2e + hi.
  if (lo < rows && nseg > 1) {
    // Unnormalized partial plus (max, sum) for attn_split_reduce_kernel.
    const int hq = gridDim.x * HPB;
    const int64_t r = (int64_t(q0 + local0 + lo) * hq + h) * nseg + seg;
    float* prow = o_part + r * kHD + half * kHalf;
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
      for (int e = 0; e < 8; ++e) prow[dt * 16 + 2 * e + hi] = o_acc[dt][e];
    if (half == 0 && hi == 0) {
      ml_part[2 * r] = m_run;
      ml_part[2 * r + 1] = l_run;
    }
  } else if (lo < rows) {
    const float inv = 1.f / l_run;
    uint16_t* orow = out + int64_t(q0 + local0 + lo) * so_t + h * so_h + half * kHalf;
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        orow[dt * 16 + 2 * e + hi] =
            static_cast<uint16_t>(pack_bf16x2(o_acc[dt][e] * inv, 0.f) & 0xffffu);
  }
}

// Combines split-KV partials: one block per (token, head), one thread per dim.
template <int kHD>
__global__ __launch_bounds__(kHD) void attn_split_reduce_kernel(
    uint16_t* __restrict__ out, const float* __restrict__ o_part,
    const float* __restrict__ ml_part, int nseg, int hq, int64_t so_t,
    int64_t so_h) {
  const int64_t row = int64_t(blockIdx.x) * hq + blockIdx.y;
  const float* ml = ml_part + row * nseg * 2;
  float m = -INFINITY;
  for (int s = 0; s < nseg; ++s) m = fmaxf(m, ml[2 * s]);
  float l = 0.f, acc = 0.f;
  for (int s = 0; s < nseg; ++s) {
    const float w = ml[2 * s] == -INFINITY ? 0.f : exp2f(ml[2 * s] - m);
    l += w * ml[2 * s + 1];
    acc += w * o_part[(row * nseg + s) * kHD + threadIdx.x];
  }
  out[blockIdx.x * so_t + blockIdx.y * so_h + threadIdx.x] =
      static_cast<uint16_t>(pack_bf16x2(acc / l, 0.f) & 0xffffu);
}

}  // namespace

// q, out: [T, Hq, D] bf16 with D = 256 or 512 (last dim contiguous);
// key_cache, value_cache: [num_blocks, block_size, Hkv, D] views (any strides,
// last dim contiguous), bf16 or int8. For int8 (vLLM's int8_per_token_head
// cache) k_scale/v_scale are the matching [num_blocks, block_size, Hkv] fp32
// dequant scales; the views may have the packed cache's padded last dim.
// cu_seqlens_q, seq_lens: int32; block_table: [N, max_blocks] int32;
// tile_offsets: [N + 1] int32, cumulative ceil(q_len / 16); max_tiles: any
// upper bound on tile_offsets[N]; num_splits > 1 splits each tile's keys into
// segments (split-KV for short queries over long contexts) and combines them
// with attn_split_reduce_kernel. window_left >= 0 restricts each query at p to
// keys p - window_left .. p.
void attn_prefill_bf16(torch::Tensor& out, const torch::Tensor& q,
                       const torch::Tensor& key_cache,
                       const torch::Tensor& value_cache,
                       const std::optional<torch::Tensor>& k_scale,
                       const std::optional<torch::Tensor>& v_scale,
                       const torch::Tensor& cu_seqlens_q,
                       const torch::Tensor& seq_lens,
                       const torch::Tensor& block_table,
                       const torch::Tensor& tile_offsets, int64_t max_tiles,
                       double scale, int64_t num_splits, int64_t window_left) {
  const int64_t d = q.size(-1);
  TORCH_CHECK(d == 64 || d == 256 || d == 512, "head_dim must be 64, 256 or 512");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.size(-1) == d &&
                  q.stride(-1) == 1,
              "q must be bf16 with a contiguous head_dim");
  const bool q8 = key_cache.scalar_type() != at::kBFloat16;
  const auto cache_dtype = q8 ? at::kChar : at::kBFloat16;
  for (const auto* t : {&key_cache, &value_cache})
    TORCH_CHECK(t->scalar_type() == cache_dtype && t->size(-1) >= d &&
                    t->stride(-1) == 1,
                "key/value caches must be bf16 or int8 with a contiguous "
                "head_dim of at least head_dim elements");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.sizes() == q.sizes() &&
                  out.stride(-1) == 1,
              "out must match q");
  TORCH_CHECK(q.dim() == 3 && key_cache.dim() == 4 && value_cache.dim() == 4,
              "q [T, H, D], caches [blocks, block_size, Hkv, D]");
  const int64_t hq = q.size(1), hkv = key_cache.size(2);
  TORCH_CHECK(hq % hkv == 0, "Hq % Hkv");
  const int64_t g = hq / hkv, nseq = cu_seqlens_q.numel() - 1;
  const float* ksc_ptr = nullptr;
  const float* vsc_ptr = nullptr;
  int64_t sc_b = 0, sc_s = 0, sc_h = 0;
  if (q8) {
    TORCH_CHECK(k_scale.has_value() && v_scale.has_value(),
                "int8 KV needs k_scale and v_scale");
    const auto& ks = *k_scale;
    const auto& vs = *v_scale;
    for (const auto* t : {&ks, &vs})
      // The scales live inline in the packed cache, so their head stride is
      // the whole (K | scale | V | scale) slot (130 fp32 for head 256), not 1.
      TORCH_CHECK(t->scalar_type() == at::kFloat && t->dim() == 3 &&
                      t->size(0) == key_cache.size(0) &&
                      t->size(1) == key_cache.size(1) && t->size(2) == hkv,
                  "k_scale/v_scale must be fp32 [blocks, block_size, Hkv]");
    ksc_ptr = k_scale->data_ptr<float>();
    vsc_ptr = v_scale->data_ptr<float>();
    sc_b = ks.stride(0), sc_s = ks.stride(1), sc_h = ks.stride(2);
  }
  for (const torch::Tensor* t : {&q, static_cast<const torch::Tensor*>(&out)})
    TORCH_CHECK(t->stride(0) % 8 == 0 && t->stride(1) % 8 == 0, "16-byte aligned rows");
  // An int8 cache is read 4 bytes at a time (its padded head stride is a
  // multiple of 4, not of 8).
  const int64_t cache_align = q8 ? 4 : 8;
  // The kernel adds a slot's offset inside a cache block in 32 bits.
  for (const auto* t : {&key_cache, &value_cache})
    TORCH_CHECK(key_cache.size(1) * t->stride(1) < (int64_t(1) << 31),
                "block_size * slot stride must fit in int32");
  if (q8)
    TORCH_CHECK(key_cache.size(1) * sc_s < (int64_t(1) << 31),
                "block_size * scale slot stride must fit in int32");
  for (const auto* t : {&key_cache, &value_cache})
    TORCH_CHECK(t->stride(0) % cache_align == 0 && t->stride(1) % cache_align == 0 &&
                    t->stride(2) % cache_align == 0,
                "aligned cache rows");
  for (const auto* t : {&cu_seqlens_q, &seq_lens, &block_table, &tile_offsets})
    TORCH_CHECK(t->scalar_type() == at::kInt, "int32 metadata");
  TORCH_CHECK(block_table.stride(1) == 1, "block_table rows contiguous");
  TORCH_CHECK(num_splits >= 1 && num_splits <= 64, "num_splits in [1, 64]");
  if (max_tiles == 0) return;
  torch::Tensor o_part, ml_part;
  if (num_splits > 1) {
    auto f32 = q.options().dtype(at::kFloat);
    o_part = torch::empty({q.size(0), hq, num_splits, d}, f32);
    ml_part = torch::empty({q.size(0), hq, num_splits, 2}, f32);
  }
  float* op = num_splits > 1 ? o_part.data_ptr<float>() : nullptr;
  float* mp = num_splits > 1 ? ml_part.data_ptr<float>() : nullptr;

  auto stream_grid = [&](int hpb) {
    return dim3(hkv * (g / hpb), max_tiles, num_splits);
  };
  auto stream = at::hip::getCurrentHIPStream();
  const float scale_log2 = static_cast<float>(scale) * 1.4426950408889634f;
#define STRIX_ATTN(D, G, HPB, BN, XT, ALIAS, Q8)                              \
  attn_prefill_kernel<D, G, HPB, BN, XT, ALIAS, Q8>                          \
      <<<stream_grid(HPB), HPB * (D / dims_per_wave(D)) * kWave, 0, stream>>>( \
      reinterpret_cast<uint16_t*>(out.data_ptr()),                           \
      reinterpret_cast<const uint16_t*>(q.data_ptr()),                       \
      reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),               \
      reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),             \
      reinterpret_cast<const int8_t*>(key_cache.data_ptr()),                 \
      reinterpret_cast<const int8_t*>(value_cache.data_ptr()), ksc_ptr,      \
      vsc_ptr, cu_seqlens_q.data_ptr<int32_t>(),                             \
      seq_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),         \
      tile_offsets.data_ptr<int32_t>(), nseq, block_table.stride(0),         \
      key_cache.size(1), q.stride(0), q.stride(1), out.stride(0),            \
      out.stride(1), key_cache.stride(0), key_cache.stride(1),               \
      key_cache.stride(2), value_cache.stride(0), value_cache.stride(1),     \
      value_cache.stride(2), sc_b, sc_s, sc_h, scale_log2,                   \
      static_cast<int>(window_left), static_cast<int>(num_splits), op, mp)
// Every tile exchanges its cross-wave score partials in fp16 (XT): ~5e-4
// relative on a partial, the order of P's own bf16 rounding, and the LDS
// and VGPRs it frees are worth 2-4 % at 256 and make HPB 8 fit at 512.
#define STRIX_ATTN_CONFIGS(Q8)                                                \
  if (d == 256 && g == 6) STRIX_ATTN(256, 6, 6, 32, _Float16, false, Q8);     \
  /* G = 8 at BN = 32 needs > 256 VGPRs per 512-thread block and spills. */    \
  else if (d == 256 && g == 8) STRIX_ATTN(256, 8, 8, 16, _Float16, false, Q8); \
  /* 512: four waves per head, all eight heads of a KV head per block so     */ \
  /* each K/V tile is staged once (four heads staged it twice); fp16 score   */ \
  /* partials make that fit (58 KB) and the 32-wave block runs at 192 VGPRs: */ \
  /* 910 -> 694 ms (21.7 -> 28.5 TF) at an 8k chunk @ 32k.                  */ \
  else if (d == 512 && g == 8) STRIX_ATTN(512, 8, 8, 16, _Float16, false, Q8); \
  /* Gemma-4 sliding layers (32 q / 16 kv heads). BN 16: the 4-wave block   */ \
  /* at BN 32 spilled at 256 VGPRs and its LDS held 2 waves per SIMD (5 TF */ \
  /* windowed); BN 16 + fp16 partials runs 20 TF.                          */ \
  else if (d == 256 && g == 2) STRIX_ATTN(256, 2, 2, 16, _Float16, false, Q8); \
  /* 64 (LFM2.5): one wave per head, all four heads of a KV head per block; */ \
  /* BN 32 (185 VGPRs) beat 64 by 12% at 8k. */                                \
  else if (d == 64 && g == 4) STRIX_ATTN(64, 4, 4, 32, float, false, Q8);     \
  else TORCH_CHECK(false, "attn_prefill_bf16: head_dim ", d, " GQA ", g,      \
                   " not built (64: 4; 256: 2, 6, 8; 512: 8)")
  if (q8) {
    STRIX_ATTN_CONFIGS(true);
  } else {
    STRIX_ATTN_CONFIGS(false);
  }
#undef STRIX_ATTN_CONFIGS
#undef STRIX_ATTN
  if (num_splits > 1) {
    auto reduce = d == 64    ? attn_split_reduce_kernel<64>
                  : d == 256 ? attn_split_reduce_kernel<256>
                             : attn_split_reduce_kernel<512>;
    reduce<<<dim3(q.size(0), hq), d, 0, stream>>>(
        reinterpret_cast<uint16_t*>(out.data_ptr()), op, mp,
        static_cast<int>(num_splits), hq, out.stride(0), out.stride(1));
  }
}

}  // namespace strix
