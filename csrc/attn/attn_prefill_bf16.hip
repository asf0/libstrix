// SPDX-License-Identifier: MIT
// Causal paged attention for prefill: head_dim D = 64 (GQA 4, LFM2.5), 128
// (GQA 8, Flint; GQA 16, Muse-Glimmer), 256 (GQA 2, Gemma-4 sliding layers;
// GQA 6; GQA 8) or 512 (GQA 8, Gemma-4 full-attention layers), bf16 Q, bf16 or
// int8_per_token_head KV, any KV block size, optional sliding window and
// split-KV. Optional attention sinks (one learned logit per query head that
// joins every softmax denominator, as in vLLM's Triton kernel), folded in
// once: in the epilogue, or in the reduce of a split launch.
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
#include "attn_split_reduce.h"
#include "inkling_kv.h"

namespace strix {
namespace {

// Head dims per wave: a head of D <= 128 is one wave; larger heads split
// into W = D / 128 waves.
constexpr int dims_per_wave(int d) { return d < 128 ? d : 128; }
// Waves per SIMD the register allocation must leave room for (VGPRs come in
// steps of 24): 16-wave blocks (4 per SIMD) need <= 192 VGPRs for a second
// block per WGP to fill the barrier bubbles, 32-wave blocks to launch at all;
// 12-wave blocks (3 per SIMD) need <= 240 for two.
constexpr int waves_target(int waves) { return waves >= 16 ? 8 : waves >= 8 ? 6 : 1; }

constexpr int kBM = 16;     // query tokens per row tile

// x from the same lane of the other half of the wave. Every lane of a wave is
// active, so the destination's old value is never read: leaving it undefined
// saves the copy xor_shuffle_f32 needs.
__device__ __forceinline__ uint32_t swap_half(uint32_t x) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
  int old;
  return __builtin_amdgcn_permlanex16(old, int(x), permlane_identity_lo,
                                      permlane_identity_hi, false, false);
#pragma clang diagnostic pop
}
__device__ __forceinline__ float swap_half(float x) {
  return __uint_as_float(swap_half(__float_as_uint(x)));
}

// Eight floats from a 16-byte aligned address.
__device__ __forceinline__ void read_scales(const float* p, float (&o)[8]) {
  const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  o[0] = a.x, o[1] = a.y, o[2] = a.z, o[3] = a.w, o[4] = b.x, o[5] = b.y, o[6] = b.z, o[7] = b.w;
}

// G query heads per KV head, HPB of them per block (G / HPB blocks share a
// K/V tile stream), BN keys per step. The partial-S exchange buffer grows
// with HPB * BN; XT = _Float16 halves it (partials are pre-scaled to the
// log2 softmax domain, |S| << 65504). ALIAS puts the exchange buffer over
// the K tile (one more barrier per step) so 8 heads x 32 keys fit in 64 KB.
//
// Q8: K/V arrive as int8 with one fp32 scale per (key, head) (vLLM's
// int8_per_token_head cache). The bytes are widened to fp16 on the way into
// LDS *without* the scale, which is exact (|v| <= 127); the scale is applied
// once per (query, key) instead - to the score for K, to the P fragment for
// V - so the per-element cost is one convert, not a convert and a multiply.
//
// QT: 16-query tiles per block (each its own wave per head): more queries
// per staged K/V tile for small head groups.
//
// PACK (split launches with few queries per sequence): the rows of a block are
// the (query head, query) pairs of one KV head, packed 16 to a wave, and HPB
// is the row tiles per block. q queries cost ceil(G * q / 16) row tiles
// instead of G; blocks past the last row tile exit, and waves past the last
// row only stage.
//
// REL (Inkling 2): relative attention. A sheared bf16 logit table
// rel[T + 128][Hq][e_pad] holds, for the queries of each 128-query tile of a
// sequence, the logits of the e_pad keys ending at the tile's last 128-key
// boundary (see Inkling2's attention_prologue): key j of a query in that tile
// reads column j - key0, key0 = ceil128(context + tile_end) - e_pad; keys
// before key0 have no logit. The epilogue applies the output gate and the
// per-token attention sink: out = o * sigmoid(gate) * sigmoid(lse - sink),
// with gate [T, Hq * D] and sink [T, Hq] (bf16, row strides sg_t / ss_t).
//
// REL 2 computes the logits instead of reading a table: rel is then r
// [T][Hq][16] (bf16, row strides sr_t / sr_h), each query's clipped relative
// vector, and relp [extent][16] (bf16) the transposed relative projection;
// key j of the query at p gets r . relp[p - j] for p - j < extent (passed as
// e_pad), else nothing. For a 16-key subtile the 16 queries of a wave need
// only 31 distances: one WMMA per 16 distances (relp rows x the wave's r)
// gives them all, and a per-wave LDS scratch shears them into the score
// layout. Adjacent subtiles share a 16-distance tile, so a step of NS
// subtiles costs NS + 1 WMMAs.
template <int D, int G, int HPB, int BN, typename XT, bool ALIAS, bool Q8, int QT = 1,
          int REL = 0, bool PACK = false>
__global__ __launch_bounds__(HPB * (D / dims_per_wave(D)) * QT * kWave)
__attribute__((amdgpu_waves_per_eu(waves_target(HPB * (D / dims_per_wave(D)) * QT)))) void
attn_prefill_kernel(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    const int8_t* __restrict__ k8, const int8_t* __restrict__ v8,
    const float* __restrict__ ksc, const float* __restrict__ vsc,
    const int32_t* __restrict__ cu_q, const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table, int nseq, int64_t bt_stride,
    int block_size, int64_t sq_t, int64_t sq_h, int64_t so_t, int64_t so_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s,
    int64_t sv_h, int64_t sc_b, int64_t sc_s, int64_t sc_h, float scale_log2,
    int win_left, int nseg, float* __restrict__ o_part,
    float* __restrict__ ml_part, const float* __restrict__ sinks,
    const uint16_t* __restrict__ rel, int64_t sr_t, int64_t sr_h, int e_pad,
    const uint16_t* __restrict__ gate, int64_t sg_t,
    const uint16_t* __restrict__ tsink, int64_t ss_t,
    const uint16_t* __restrict__ relp) {
  constexpr int kHalf = dims_per_wave(D);
  constexpr int kHD = D, W = D / kHalf;  // waves per head
  constexpr int kPK = kHD + 8;     // LDS pitch of the K tile [key][d]
  constexpr int kThreads = HPB * W * QT * kWave;
  constexpr int NS = BN / 16;      // 16-key subtiles per step
  constexpr int kPV = BN + 8;      // LDS pitch of the V^T tile [d][key]
  static_assert(REL != 1 || (128 % (16 * QT) == 0 && 128 % BN == 0),
                "REL 1: a block lies in one 128-query tile, a step in one side of key0");
  static_assert(!REL || W == 1, "REL: one-wave heads");
  static_assert(!Q8 || kThreads >= BN, "Q8: the first BN lanes stage the scales");
  // Cross-wave score partials: only heads split over W > 1 waves use them.
  typedef XT xbuf_t[W > 1 ? W * HPB * QT : 1][W > 1 ? NS : 1][kWave][8];
  constexpr int kKBytes = BN * kPK * 2, kXBytes = sizeof(xbuf_t);
  __shared__ __attribute__((aligned(16)))
  uint8_t s_kx[ALIAS ? (kKBytes > kXBytes ? kKBytes : kXBytes) : kKBytes + kXBytes];
  __shared__ __attribute__((aligned(16))) uint16_t s_vt[kHD * kPV];
  uint16_t* s_k = reinterpret_cast<uint16_t*>(s_kx);
  xbuf_t& s_x = *reinterpret_cast<xbuf_t*>(s_kx + (ALIAS ? 0 : kKBytes));
  // Q8 only: per-key dequant scales of the current K/V tile, staged next to
  // the tiles so the score and P loops read them from LDS.
  __shared__ __attribute__((aligned(16))) float s_ks[Q8 ? BN : 4], s_vs[Q8 ? BN : 4];

  // Last tiles first: causal tiles grow with position, so the longest blocks
  // start early and the short ones fill the tail.
  const int kvh = blockIdx.x / (G / HPB), hgroup = blockIdx.x % (G / HPB);
  // blockIdx.y counts the blocks' 16 * QT query tiles from the last one of the
  // last sequence.
  int n = nseq - 1, tile = -1;
  for (int rest = blockIdx.y; n >= 0; --n) {
    const int tiles = (cu_q[n + 1] - cu_q[n] + kBM * QT - 1) / (kBM * QT);
    if (rest < tiles) {
      tile = tiles - 1 - rest;
      break;
    }
    rest -= tiles;
  }
  if (tile < 0) return;

  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  // grp: (query tile, head) of this wave; W waves split the head's dims.
  const int grp = wave / W, half = wave % W, lo = lane % 16, hi = lane / 16;
  const int qt = grp / HPB;
  const int q0 = cu_q[n], q_len = cu_q[n + 1] - q0;
  const int kv_len = seq_lens[n], context = kv_len - q_len;
  const int block0 = tile * kBM * QT;
  const int local0 = block0 + qt * kBM;
  // A wave past the sequence's last query (QT > 1) has no rows.
  const int rows = max(0, min(kBM, q_len - local0));
  // PACK: row `pack_r` of the KV head's (query head, query) pairs.
  static_assert(!PACK || (QT == 1 && !REL), "PACK: plain 16-query tiles");
  const int nq = min(kBM, q_len - block0), pack_rows = G * nq;
  const int pack_base = hgroup * HPB * 16;
  if (PACK && pack_base >= pack_rows) return;
  const int pack_r = pack_base + grp * 16 + lo;
  // Invalid rows reuse the last valid row's query (never written back).
  const int pack_rr = min(pack_r, pack_rows - 1);
  const bool active = !PACK || pack_base + grp * 16 < pack_rows;
  const bool valid = PACK ? pack_r < pack_rows : lo < rows;
  const int h = kvh * G + (PACK ? pack_rr / nq : hgroup * HPB + grp % HPB);
  const int my_q = PACK ? block0 + pack_rr % nq
                   : rows > 0 ? local0 + min(lo, rows - 1)
                              : q_len - 1;
  // Draft blocks (win_left <= -2): non-causal, so every query also sees the
  // keys after it, up to the last of the sequence. -2 is no window, -3 - w a
  // window of w keys before the query.
  const bool non_causal = win_left <= -2;
  const int win = non_causal ? (win_left == -2 ? -1 : -3 - win_left) : win_left;
  const int my_pos = context + my_q;
  // The block's first and last query positions bound its keys.
  const int first_pos = context + block0;
  const int last_pos = context + min(block0 + kBM * QT, q_len) - 1;
  // Every row sees keys up to hi_first, the block's keys end at key_end.
  const int hi_first = non_causal ? kv_len - 1 : first_pos;
  const int key_end = non_causal ? kv_len : last_pos + 1;
  const int32_t* bt = block_table + int64_t(n) * bt_stride;
  // REL: the block's queries share key0 (one 128-query tile); rel_row[j]
  // is key j's logit for this lane's query.
  int key0 = 0;
  const uint16_t* rel_row = nullptr;
  if constexpr (REL == 1) {
    const int tile_end = min((block0 / 128 + 1) * 128, q_len);
    key0 = (context + tile_end + 127) / 128 * 128 - e_pad;
    rel_row = rel + int64_t(q0 + my_q) * sr_t + h * sr_h - key0;
  }
  // REL 2: this lane's query's r (the B fragment of the distance WMMAs) and
  // the wave's scratch of [distance][query] logits for one step.
  bf16x16 rfrag = {};
  if constexpr (REL == 2) rfrag = load_frag_bf16(rel + int64_t(q0 + my_q) * sr_t + h * sr_h, 0, 0);
  __shared__ float s_g[REL == 2 ? HPB * QT : 1][REL == 2 ? 16 * (NS + 1) : 1][16];

  // Q^T B fragments: row = query lo, k = this half's dims.
  bf16x16 qf[kHalf / 16];
  {
    const uint16_t* qrow =
        q + int64_t(q0 + my_q) * sq_t + h * sq_h + half * kHalf;
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
  // Every lane has a chunk for each of its loads.
  constexpr bool kFullGrid = kChunks % kThreads == 0;
  typedef uint32_t stage_t __attribute__((ext_vector_type((Q8 ? 2 : 4) * kLoads)));
  constexpr int kWords = Q8 ? 2 : 4;  // words of one 8-dim chunk
  stage_t rk = {}, rv = {};
  // Q8: the block's first BN lanes load (and stage) one key's two scales each.
  float rks = 0.f, rvs = 0.f;
  // P times the V scale is rounded to fp16: the gain keeps small values out of
  // fp16's subnormals (|v| up to 32k stays in range).
  constexpr float kVGain = 256.f;
  auto chunk_addr = [&](int key, int64_t sb, int64_t ss, int64_t sh) {
    const int64_t blk = bt[key / block_size];
    return blk * sb + int64_t(key % block_size) * ss + kvh * sh;
  };
  // K chunks are key-major (row-major tile). V chunks are key-fastest so the
  // transposed 2-byte stores of a wave land side by side (key-major mapping
  // was a 32-way LDS bank conflict and half the kernel time).
  //
  // The page of the current step: its bases in scalar registers, recomputed
  // only when a step enters another page. A lane adds its constant byte offset
  // within a step (one load instruction, no address arithmetic). Steps that
  // cross a page or the end of the sequence take the general path.
  const int ss_k = int(sk_s), ss_v = int(sv_s), ss_c = int(sc_s);
  constexpr int kEs = Q8 ? 1 : 2;  // bytes per cache element
  const uint8_t* kbase = Q8 ? reinterpret_cast<const uint8_t*>(k8)
                            : reinterpret_cast<const uint8_t*>(kc);
  const uint8_t* vbase = Q8 ? reinterpret_cast<const uint8_t*>(v8)
                            : reinterpret_cast<const uint8_t*>(vc);
  uint32_t lkb[kLoads], lvb[kLoads];
#pragma unroll
  for (int i = 0; i < kLoads; ++i) {
    const int ch = tid + i * kThreads;
    lkb[i] = uint32_t((ch / (kHD / 8)) * ss_k + (ch % (kHD / 8)) * 8) * kEs;
    lvb[i] = uint32_t((ch % BN) * ss_v + (ch / BN) * 8) * kEs;
  }
  int pg_lo = 0, pg_hi = 0;
  const uint8_t *pgk = nullptr, *pgv = nullptr;
  const float *pgks = nullptr, *pgvs = nullptr;
  // One 8-dim chunk. The asm keeps a lane's offset 32-bit inside the loop, so
  // that it is the load's own (scalar base + 32-bit offset) address.
  typedef uint32_t chunk_t __attribute__((ext_vector_type(kWords), aligned(4)));
#define STRIX_AT(BASE, OFF) (*reinterpret_cast<const chunk_t*>((BASE) + (OFF)))
#define STRIX_FETCH(K0)                                                      \
  if ((K0) >= pg_hi) {                                                       \
    const int fb = (K0) / block_size;                                        \
    const int64_t blk = bt[fb];                                              \
    pg_lo = fb * block_size, pg_hi = pg_lo + block_size;                     \
    const int64_t ko = blk * sk_b + kvh * sk_h, vo = blk * sv_b + kvh * sv_h; \
    pgk = kbase + ko * kEs, pgv = vbase + vo * kEs;                          \
    const int64_t co = blk * sc_b + kvh * sc_h;                              \
    if constexpr (Q8) pgks = ksc + co, pgvs = vsc + co;                      \
  }                                                                          \
  const int f_key = (K0) - pg_lo;                                            \
  const uint8_t *fkp = pgk + int64_t(f_key) * ss_k * kEs;                    \
  const uint8_t *fvp = pgv + int64_t(f_key) * ss_v * kEs;                    \
  if ((K0) + BN <= min(pg_hi, kv_len)) {                                     \
    _Pragma("unroll") for (int i = 0; i < kLoads; ++i) {                     \
      const int ch = tid + i * kThreads;                                     \
      if (kFullGrid || ch < kChunks) {                                       \
        uint32_t ok = lkb[i], ov = lvb[i];                                   \
        asm volatile("" : "+v"(ok), "+v"(ov));                               \
        const chunk_t a = STRIX_AT(fkp, ok), b = STRIX_AT(fvp, ov);          \
        _Pragma("unroll") for (int w = 0; w < kWords; ++w)                   \
          rk[kWords * i + w] = a[w], rv[kWords * i + w] = b[w];              \
      }                                                                      \
    }                                                                        \
    if (Q8 && tid < BN) {                                                    \
      const uint32_t o = uint32_t(tid * ss_c);                               \
      rks = pgks[int64_t(f_key) * ss_c + o], rvs = pgvs[int64_t(f_key) * ss_c + o]; \
    }                                                                        \
  } else {                                                                   \
    _Pragma("unroll") for (int i = 0; i < kLoads; ++i) {                     \
      const int ch = tid + i * kThreads;                                     \
      const int kkey = (K0) + ch / (kHD / 8), kd = (ch % (kHD / 8)) * 8;     \
      const int vkey = (K0) + ch % BN, vd = (ch / BN) * 8;                   \
      chunk_t a = {}, b = {};                                                \
      if (ch < kChunks && kkey < kv_len) {                                   \
        const int64_t o = (chunk_addr(kkey, sk_b, sk_s, sk_h) + kd) * kEs;   \
        a = STRIX_AT(kbase, o);                                              \
      }                                                                      \
      if (ch < kChunks && vkey < kv_len) {                                   \
        const int64_t o = (chunk_addr(vkey, sv_b, sv_s, sv_h) + vd) * kEs;   \
        b = STRIX_AT(vbase, o);                                              \
      }                                                                      \
      _Pragma("unroll") for (int w = 0; w < kWords; ++w)                     \
        rk[kWords * i + w] = a[w], rv[kWords * i + w] = b[w];                \
    }                                                                        \
    if (Q8 && tid < BN && (K0) + tid < kv_len) {                             \
      const int64_t o = chunk_addr((K0) + tid, sc_b, sc_s, sc_h);            \
      rks = ksc[o], rvs = vsc[o];                                            \
    }                                                                        \
  }

  // Split-KV: segment blockIdx.z covers keys [k_begin, k_end) of the keys
  // this tile needs; with nseg == 1 it is all of them.
  // Sliding window (vLLM's (window - 1, 0)): a query at p sees keys
  // p - win_left .. p, so the block starts at its first row's window.
  // REL aligns it down to a step, so a step lies on one side of key0 and
  // its logits are 16-byte aligned (the extra keys are masked).
  int k_lo = win >= 0 ? max(0, first_pos - win) : 0;
  if constexpr (REL == 1) k_lo = k_lo / BN * BN;
  // Aligned pages: a step that starts on a step boundary stays in one page (the
  // keys before the window are masked, and lie in a page that is still live).
  if (block_size % BN == 0) k_lo = k_lo / BN * BN;
  const int seg = blockIdx.z, keys_needed = key_end;
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
      if (kFullGrid || ch < kChunks) {
        const int kkey = ch / (kHD / 8), kd = (ch % (kHD / 8)) * 8;
        const int vkey = ch % BN, vd = (ch / BN) * 8;
        uint32_t kk[4], vv[4];
        if constexpr (Q8) {
          // Two words of int8 each widen to four words of fp16.
          q8x8_to_f16(rk[2 * i], rk[2 * i + 1], kk);
          q8x8_to_f16(rv[2 * i], rv[2 * i + 1], vv);
        } else {
#pragma unroll
          for (int w = 0; w < 4; ++w) kk[w] = rk[4 * i + w], vv[w] = rv[4 * i + w];
        }
        *reinterpret_cast<uint4*>(s_k + kkey * kPK + kd) =
            make_uint4(kk[0], kk[1], kk[2], kk[3]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          s_vt[(vd + 2 * e) * kPV + vkey] = static_cast<uint16_t>(vv[e] & 0xffffu);
          s_vt[(vd + 2 * e + 1) * kPV + vkey] = static_cast<uint16_t>(vv[e] >> 16);
        }
      }
    }
    if (Q8 && tid < BN) {
      // Keys of one parity side by side: a lane reads the eight it needs as two
      // 16-byte loads. The K scale carries the softmax scale.
      const int p = (tid & ~15) + (tid & 1) * 8 + (tid % 16) / 2;
      s_ks[p] = rks * scale_log2;
      s_vs[p] = rvs * kVGain;
    }
    barrier();
    if (step + 1 < n_steps) {
      STRIX_FETCH(k0 + BN)
    }

    // Partial S^T over this wave's half of head_dim, per 16-key subtile.
    constexpr bool kRaw = !Q8 && !REL;  // unscaled scores (the scale joins a later fma)
    f32x8 s[NS];
    if (active) {
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      s[j] = f32x8{};
#pragma unroll
      for (int kk = 0; kk < kHalf / 16; ++kk) {
        const bf16x16 kf =
            load_frag_bf16(s_k + j * 16 * kPK + half * kHalf + kk * 16, kPK, lane);
        s[j] = Q8 ? wmma_f16(kf, qf[kk], s[j]) : wmma_bf16(kf, qf[kk], s[j]);
      }
      float ksv[8] = {};
      if constexpr (Q8) read_scales(s_ks + 16 * j + 8 * hi, ksv);
#pragma unroll
      for (int e = 0; e < 8; ++e)
        if constexpr (!kRaw) s[j][e] *= Q8 ? ksv[e] : scale_log2;
    }
    }
    if constexpr (W > 1) {  // one-wave heads have their full scores already
      if constexpr (ALIAS) barrier();  // every wave is done reading s_k
      if (active) {
#pragma unroll
        for (int j = 0; j < NS; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e)
            s_x[wave][j][lane][e] = static_cast<XT>(kRaw ? s[j][e] * scale_log2 : s[j][e]);
      }
      barrier();
    }

    if (!active) continue;

    // Online softmax for query lo; element e of subtile j is key
    // k0 + 16 j + 2e + hi.
    // A step wholly below the tile's first query (its last key when non-causal), and (with a window) wholly
    // inside its last query's window, needs no mask - every step but the
    // first and last one or two. The block-uniform branch drops ~2 compares
    // and a select per element.
    const bool full_step = k0 + BN - 1 <= hi_first &&
                           (win < 0 || last_pos - k0 <= win);
    if constexpr (REL == 1) {
      // Relative logits (natural log) of keys from key0 on. Key 2e + hi of a
      // subtile is the low (hi = 0) or high half of word e of its 16 logits.
      if (k0 >= key0) {
        const uint32_t sel = hi ? 0x03020c0cu : 0x01000c0cu;
#pragma unroll
        for (int j = 0; j < NS; ++j) {
          const uint4* p = reinterpret_cast<const uint4*>(rel_row + k0 + 16 * j);
          const uint4 a = p[0], b = p[1];
          const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
          for (int e = 0; e < 8; ++e)
            s[j][e] = fmaf(__uint_as_float(__builtin_amdgcn_perm(0u, w[e], sel)),
                           1.4426950408889634f, s[j][e]);
        }
      }
    }
    if constexpr (REL == 2) {
      // Keys within the extent of some query of the block. Scratch row d is
      // distance base + d for query column lo; subtile j's element e (key
      // k0 + 16 j + 2 e + hi) is row lo - 2 e - hi + 15 + 16 (NS - 1 - j).
      if (k0 + BN - 1 > first_pos - e_pad) {
        const int base = context + local0 - k0 - 16 * (NS - 1) - 15;
        float(*g)[16] = s_g[grp];
#pragma unroll
        for (int t = 0; t <= NS; ++t) {
          const int row = min(max(base + 16 * t + lo, 0), e_pad - 1);
          const f32x8 c = wmma_bf16(load_frag_bf16(relp + row * 16, 0, 0), rfrag, f32x8{});
#pragma unroll
          for (int e = 0; e < 8; ++e) g[16 * t + 2 * e + hi][lo] = c[e];
        }
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int j = 0; j < NS; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int d = lo - 2 * e - hi + 15 + 16 * (NS - 1 - j);
            const bool in = base + d < e_pad;
            s[j][e] = fmaf(in ? g[d][lo] : 0.f, 1.4426950408889634f, s[j][e]);
          }
        __builtin_amdgcn_wave_barrier();
      }
    }
    float m_loc = -INFINITY;
    if (full_step) {
#pragma unroll
      for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
#pragma unroll
          for (int o = 1; o < W; ++o) {
            const float x = static_cast<float>(s_x[grp * W + (half + o) % W][j][lane][e]);
            s[j][e] = kRaw && o == 1 ? fmaf(s[j][e], scale_log2, x) : s[j][e] + x;
          }
          m_loc = fmaxf(m_loc, s[j][e]);
        }
    } else {
      // Key k0 + c + hi is visible for t_lo <= c <= t_hi.
      const int t_hi = (non_causal ? kv_len - 1 : my_pos) - hi - k0;
      const int t_lo = win >= 0 ? my_pos - hi - k0 - win : -1;
#pragma unroll
      for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int c = 16 * j + 2 * e;
          float v = s[j][e];
#pragma unroll
          for (int o = 1; o < W; ++o) {
            const float x = static_cast<float>(s_x[grp * W + (half + o) % W][j][lane][e]);
            v = kRaw && o == 1 ? fmaf(v, scale_log2, x) : v + x;
          }
          const bool seen = c <= t_hi && c >= t_lo;
          s[j][e] = seen ? v : -INFINITY;
          m_loc = fmaxf(m_loc, s[j][e]);
        }
    }
    m_loc = fmaxf(m_loc, swap_half(m_loc));
    const float m_new = fmaxf(m_run, m_loc);
    // A row may see no key in this split-KV segment yet: keep p = 0, not NaN.
    const float m_ref = m_new == -INFINITY ? 0.f : m_new;
    // Raw v_exp_f32 (no denormal fix-up; a softmax weight below 2^-126 is
    // 0 either way): exp2f's range check was 3 extra VALU ops per element.
    constexpr bool kRawMax = kRaw && W == 1;  // maxima in unscaled units
    const float alpha =
        __builtin_amdgcn_exp2f(kRawMax ? (m_run - m_ref) * scale_log2 : m_run - m_ref);
    const float m_neg = kRawMax ? -m_ref * scale_log2 : 0.f;
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
      float vsv[8] = {};
      if constexpr (Q8) read_scales(s_vs + 16 * j + 8 * hi, vsv);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float p = __builtin_amdgcn_exp2f(
            kRawMax ? fmaf(s[j][e], scale_log2, m_neg) : s[j][e] - m_ref);
        l_loc += p;
        // P^T B fragment: row = query lo, 16 keys; this half holds key
        // 2e + hi, the other half 2e + (1 - hi). Each lane rounds its own
        // value once and swaps the rounded word; one perm with a per-half
        // selector puts key 2e low and 2e + 1 high.
        float pv = p;
        // V's per-key scale rides on P so the accumulator needs no
        // per-element pass, and l stays the unweighted sum.
        if constexpr (Q8) pv *= vsv[e];
        // Q8: P in fp16 for the fp16 WMMA, in the low half of the word; bf16
        // (ties round up) in the high half.
        const uint32_t mine = Q8 ? f16_bits(pv) : __float_as_uint(pv) + 0x8000u;
        const uint32_t other = swap_half(mine);
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
    l_loc += swap_half(l_loc);
    l_run = l_run * alpha + l_loc;
    m_run = m_new;
  }
#undef STRIX_FETCH
#undef STRIX_AT
  if constexpr (!Q8 && !REL && W == 1) m_run *= scale_log2;  // kRawMax
  if constexpr (Q8) {
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
      for (int e = 0; e < 8; ++e) o_acc[dt][e] *= 1.f / kVGain;
  }

  // O^T[d][query lo]: element e of tile dt is d = dt*16 + 2e + hi.
  if (valid && nseg > 1) {
    // Unnormalized partial plus (max, sum) for attn_split_reduce_kernel.
    const int hq = gridDim.x * HPB;
    const int64_t r = (int64_t(q0 + my_q) * hq + h) * nseg + seg;
    float* prow = o_part + r * kHD + half * kHalf;
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
      for (int e = 0; e < 8; ++e) prow[dt * 16 + 2 * e + hi] = o_acc[dt][e];
    if (half == 0 && hi == 0) {
      ml_part[2 * r] = m_run;
      ml_part[2 * r + 1] = l_run;
    }
  } else if (REL && valid) {
    // out = o / l * sigmoid(gate) * sigmoid(lse - sink), lse natural-log.
    const int64_t token = q0 + my_q;
    const float inv = 1.f / l_run;
    const float lse = (m_run + __log2f(l_run)) * 0.6931471805599453f;
    const float snk = bf16_lo(tsink[token * ss_t + h]);
    const float g_sink = 1.f / (1.f + __builtin_amdgcn_exp2f((snk - lse) * 1.4426950408889634f));
    const uint16_t* grow = gate + token * sg_t + h * kHD;
    uint16_t* orow = out + token * so_t + h * so_h;
    const uint32_t sel = hi ? 0x03020c0cu : 0x01000c0cu;
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt) {
      const uint4* p = reinterpret_cast<const uint4*>(grow + dt * 16);
      const uint4 a = p[0], b = p[1];
      const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float g = __uint_as_float(__builtin_amdgcn_perm(0u, w[e], sel));
        const float y = o_acc[dt][e] * inv *
                        (1.f / (1.f + __builtin_amdgcn_exp2f(-g * 1.4426950408889634f))) *
                        g_sink;
        orow[dt * 16 + 2 * e + hi] = static_cast<uint16_t>(round_bf16_hi(y) >> 16);
      }
    }
  } else if (valid) {
    float inv = 1.f / l_run;
    if (sinks != nullptr) {
      // The sink is one more logit (natural-log units, unscaled) in the
      // softmax: fold it into the final max and sum.
      const float s2 = sinks[h] * 1.4426950408889634f;
      const float m_fin = fmaxf(m_run, s2);
      const float a = __builtin_amdgcn_exp2f(m_run - m_fin);
      inv = a / (l_run * a + __builtin_amdgcn_exp2f(s2 - m_fin));
    }
    uint16_t* orow = out + int64_t(q0 + my_q) * so_t + h * so_h + half * kHalf;
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        orow[dt * 16 + 2 * e + hi] =
            static_cast<uint16_t>(pack_bf16x2(o_acc[dt][e] * inv, 0.f) & 0xffffu);
  }
}

}  // namespace

// q, out: [T, Hq, D] bf16 (last dim contiguous), (D, Hq / Hkv) as listed above;
// key_cache, value_cache: [num_blocks, block_size, Hkv, D] views (any strides,
// last dim contiguous), bf16 or int8. For int8 (vLLM's int8_per_token_head
// cache) k_scale/v_scale are the matching [num_blocks, block_size, Hkv] fp32
// dequant scales; the views may have the packed cache's padded last dim.
// cu_seqlens_q, seq_lens: int32; block_table: [N, max_blocks] int32;
// tile_offsets is not read (the kernel takes the tiles from cu_seqlens_q);
// max_tiles: any upper bound on the sum of ceil(q_len / 16); num_splits > 1
// splits each tile's keys into segments (split-KV for short queries over long
// contexts) and combines them with attn_split_reduce_kernel. window_left >= 0
// restricts each query at p to keys p - window_left .. p. sinks: optional [Hq]
// fp32.
void attn_prefill_bf16(torch::Tensor& out, const torch::Tensor& q,
                       const torch::Tensor& key_cache,
                       const torch::Tensor& value_cache,
                       const std::optional<torch::Tensor>& k_scale,
                       const std::optional<torch::Tensor>& v_scale,
                       const torch::Tensor& cu_seqlens_q,
                       const torch::Tensor& seq_lens,
                       const torch::Tensor& block_table,
                       const torch::Tensor& tile_offsets, int64_t max_tiles,
                       double scale, int64_t num_splits, int64_t window_left,
                       const std::optional<torch::Tensor>& sinks) {
  const int64_t d = q.size(-1);
  TORCH_CHECK(d == 64 || d == 128 || d == 256 || d == 512,
              "head_dim must be 64, 128, 256 or 512");
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
  const float* sink_ptr = nullptr;
  if (sinks.has_value()) {
    TORCH_CHECK(sinks->scalar_type() == at::kFloat && sinks->dim() == 1 &&
                    sinks->size(0) == hq && sinks->is_contiguous(),
                "sinks must be contiguous fp32 [Hq]");
    sink_ptr = sinks->data_ptr<float>();
  }
  float* mp = num_splits > 1 ? ml_part.data_ptr<float>() : nullptr;

  auto stream = at::hip::getCurrentHIPStream();
  const float scale_log2 = static_cast<float>(scale) * 1.4426950408889634f;
  const bool split = num_splits > 1;
  // Split launches serve speculative verify, up to 16 queries per sequence;
  // with fewer, (head, query) rows share row tiles.
  const bool packed = split && q.size(0) <= 12 * nseq;
  auto launch = [&](auto kernel, int hpb, int qt) {
    const dim3 grid(hkv * (g / hpb), qt > 1 ? max_tiles / qt + nseq : max_tiles, num_splits);
    kernel<<<grid, static_cast<uint32_t>(hpb * qt * (d / dims_per_wave(d)) * kWave), 0, stream>>>(
        reinterpret_cast<uint16_t*>(out.data_ptr()),
        reinterpret_cast<const uint16_t*>(q.data_ptr()),
        reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),
        reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),
        reinterpret_cast<const int8_t*>(key_cache.data_ptr()),
        reinterpret_cast<const int8_t*>(value_cache.data_ptr()), ksc_ptr, vsc_ptr,
        cu_seqlens_q.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
        block_table.data_ptr<int32_t>(), nseq,
        block_table.stride(0), key_cache.size(1), q.stride(0), q.stride(1), out.stride(0),
        out.stride(1), key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), sc_b, sc_s, sc_h,
        scale_log2, static_cast<int>(window_left), static_cast<int>(num_splits), op, mp,
        sink_ptr, nullptr, 0, 0, 0, nullptr, 0, nullptr, 0, nullptr);
  };
#define STRIX_K(D, G, HPB, BN, XT, Q8, QT, PACK) \
  attn_prefill_kernel<D, G, HPB, BN, XT, false, Q8, QT, 0, PACK>
// HPB x BN x QT for unsplit launches, HPB x SBN for split ones, PHPB x PBN for
// packed ones. Every tile exchanges its cross-wave score partials in XT: fp16
// costs ~5e-4 relative on a partial, the order of P's own bf16 rounding, and
// frees LDS and VGPRs.
#define STRIX_TILE(D, G, XT, HPB, BN, QT, SBN, PHPB, PBN, Q8)                        \
  (d == D && g == G) launch(!split   ? STRIX_K(D, G, HPB, BN, XT, Q8, QT, false)     \
                            : packed ? STRIX_K(D, G, PHPB, PBN, XT, Q8, 1, true)     \
                                     : STRIX_K(D, G, HPB, SBN, XT, Q8, 1, false),    \
                            packed ? PHPB : HPB, split ? 1 : QT);
#define STRIX_TILES(Q8)                                                             \
  /* 27B */                                                                         \
  if STRIX_TILE(256, 6, _Float16, 6, 32, 1, 32, 3, 16, Q8)                           \
  /* 35B: 512 threads at BN 32 need more than 192 VGPRs. */                         \
  else if STRIX_TILE(256, 8, _Float16, 8, 16, 1, 16, 4, 32, Q8)                     \
  /* Gemma-4 full layers: four waves per head, all eight heads of a KV head per */  \
  /* block, so each K/V tile is staged once. */                                     \
  else if STRIX_TILE(512, 8, _Float16, 8, 16, 1, 16, 8, 16, Q8)                     \
  /* Gemma-4 sliding layers: only two heads share a KV head, so four query tiles */ \
  /* share the staged K/V tile. */                                                  \
  else if STRIX_TILE(256, 2, _Float16, 2, 16, 4, 16, 2, 16, Q8)                     \
  /* LFM2.5 */                                                                      \
  else if STRIX_TILE(64, 4, float, 4, 64, 4, 32, 2, 32, Q8)                         \
  /* DFlash drafts: 32 query / 8 KV heads, 2048-key window; packed blocks of 4 */ \
  /* row tiles (1 or 2 were 0.65-1.0x Triton at 4 drafts, 4 is 1.2-3.8x). */       \
  else if STRIX_TILE(128, 4, float, 4, 32, 4, 32, 4, 32, Q8)                         \
  /* Flint (QT 4 is 3 % faster on full layers, 12 % slower on 513-key windows) */   \
  else if STRIX_TILE(128, 8, float, 8, 32, 1, 32, 4, 32, Q8)                        \
  /* Muse-Glimmer: BN 16 fits the 192 VGPRs of a 16-wave block, BN 32 does not. */  \
  else if STRIX_TILE(128, 16, float, 16, 16, 1, 16, 16, 16, Q8)                     \
  else TORCH_CHECK(false, "attn_prefill_bf16: head_dim ", d, " GQA ", g,       \
                   " not built (64: 4; 128: 4, 8, 16; 256: 2, 6, 8; 512: 8)")
  if (q8) {
    STRIX_TILES(true);
  } else {
    STRIX_TILES(false);
  }
#undef STRIX_TILES
#undef STRIX_TILE
#undef STRIX_K
  if (num_splits > 1)
    launch_attn_split_reduce(d, dim3(q.size(0), hq), stream,
                             reinterpret_cast<uint16_t*>(out.data_ptr()), op, mp,
                             static_cast<int>(num_splits), hq, out.stride(0),
                             out.stride(1), sink_ptr);
}

// Inkling 2 relative attention for prefill (REL above): q [T, Hq, 128] bf16
// with GQA 3; caches as attn_prefill_bf16 (bf16 or int8 with k_scale /
// v_scale); rel_logits [T + 128, Hq, E] sheared bf16 logits (always bf16: the
// relative vectors are not part of the KV cache); gate [T, Hq * 128] and
// sinks [T, Hq] bf16 views (row strides free); out [T, Hq * 128] bf16,
// written for the tokens of the sequences. tile_offsets is not read; max_tiles
// bounds the sum of ceil(q_len / (16 * q_tiles)). window_left as
// attn_prefill_bf16.
void inkling_rel_attn_prefill(torch::Tensor& out, const torch::Tensor& q,
                              const torch::Tensor& key_cache,
                              const torch::Tensor& value_cache,
                              const std::optional<torch::Tensor>& k_scale,
                              const std::optional<torch::Tensor>& v_scale,
                              const torch::Tensor& rel_logits,
                              const torch::Tensor& gate, const torch::Tensor& sinks,
                              const torch::Tensor& cu_seqlens_q,
                              const torch::Tensor& seq_lens,
                              const torch::Tensor& block_table,
                              const torch::Tensor& tile_offsets, int64_t max_tiles,
                              double scale, int64_t window_left, int64_t q_tiles,
                              int64_t block_n,
                              const std::optional<torch::Tensor>& rel_proj_t) {
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && q.scalar_type() == at::kBFloat16 &&
                  q.stride(2) == 1 && q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0,
              "q must be bf16 [T, H, 128] with 16-byte aligned rows");
  const int64_t T = q.size(0), hq = q.size(1), hkv = key_cache.size(2);
  TORCH_CHECK(hq == 3 * hkv, "GQA 3 only");
  const InklingKV kv = inkling_kv(key_cache, value_cache, k_scale, v_scale);
  const int64_t e_pad = rel_logits.size(2);
  TORCH_CHECK(rel_logits.dim() == 3 && rel_logits.scalar_type() == at::kBFloat16 &&
                  rel_logits.size(1) == hq && rel_logits.stride(2) == 1 &&
                  rel_logits.stride(0) % 8 == 0 && rel_logits.stride(1) % 8 == 0,
              "rel_logits must be bf16 [., H, .] with 16-byte aligned rows");
  TORCH_CHECK(rel_proj_t.has_value() || (rel_logits.size(0) >= T + 128 && e_pad % 128 == 0),
              "sheared rel_logits must be [T + 128, H, E] (E % 128 == 0)");
  TORCH_CHECK(gate.dim() == 2 && gate.scalar_type() == at::kBFloat16 && gate.size(0) >= T &&
                  gate.size(1) == hq * 128 && gate.stride(1) == 1 && gate.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(gate.data_ptr()) % 16 == 0,
              "gate must be bf16 [T, H * 128] with 16-byte aligned rows");
  TORCH_CHECK(sinks.dim() == 2 && sinks.scalar_type() == at::kBFloat16 && sinks.size(0) >= T &&
                  sinks.size(1) == hq && sinks.stride(1) == 1,
              "sinks must be bf16 [T, H]");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) >= T &&
                  out.size(1) == hq * 128 && out.is_contiguous(),
              "out must be contiguous bf16 [T, H * 128]");
  for (const auto* t : {&cu_seqlens_q, &seq_lens, &block_table, &tile_offsets})
    TORCH_CHECK(t->scalar_type() == at::kInt, "int32 metadata");
  TORCH_CHECK(block_table.stride(1) == 1, "block_table rows contiguous");
  TORCH_CHECK(key_cache.size(1) % block_n == 0, "block_size % block_n");
  if (max_tiles == 0 || T == 0) return;
  const int nseq = cu_seqlens_q.numel() - 1;
  auto stream = at::hip::getCurrentHIPStream();
  const float scale_log2 = static_cast<float>(scale) * 1.4426950408889634f;
  const bool fused = rel_proj_t.has_value();
  int64_t rel_w = e_pad;  // REL 1: table width; REL 2: the extent
  if (fused) {
    TORCH_CHECK(rel_logits.size(2) == 16 && rel_logits.size(0) >= T,
                "fused: rel_logits is r [T, H, 16]");
    const auto& rp = *rel_proj_t;
    TORCH_CHECK(rp.scalar_type() == at::kBFloat16 && rp.dim() == 2 && rp.size(1) == 16 &&
                    rp.is_contiguous(),
                "rel_proj_t must be contiguous bf16 [extent, 16]");
    rel_w = rp.size(0);
  }
  const uint16_t* relp =
      fused ? reinterpret_cast<const uint16_t*>(rel_proj_t->data_ptr()) : nullptr;
#define STRIX_REL(Q8, QT, BN, MODE)                                                \
  attn_prefill_kernel<128, 3, 3, BN, float, false, Q8, QT, MODE>                   \
      <<<dim3(hkv, max_tiles, 1), 3 * QT * kWave, 0, stream>>>(                    \
          reinterpret_cast<uint16_t*>(out.data_ptr()),                             \
          reinterpret_cast<const uint16_t*>(q.data_ptr()),                         \
          reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),                 \
          reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),               \
          reinterpret_cast<const int8_t*>(key_cache.data_ptr()),                   \
          reinterpret_cast<const int8_t*>(value_cache.data_ptr()), kv.k_scale,     \
          kv.v_scale, cu_seqlens_q.data_ptr<int32_t>(),                            \
          seq_lens.data_ptr<int32_t>(), block_table.data_ptr<int32_t>(),           \
          nseq, block_table.stride(0),                                             \
          key_cache.size(1), q.stride(0), q.stride(1), out.stride(0), 128,         \
          key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),           \
          value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),     \
          kv.sc_b, kv.sc_s, kv.sc_h, scale_log2, static_cast<int>(window_left), 1, \
          nullptr, nullptr, nullptr,                                               \
          reinterpret_cast<const uint16_t*>(rel_logits.data_ptr()),                \
          rel_logits.stride(0), rel_logits.stride(1), static_cast<int>(rel_w),     \
          reinterpret_cast<const uint16_t*>(gate.data_ptr()), gate.stride(0),      \
          reinterpret_cast<const uint16_t*>(sinks.data_ptr()), sinks.stride(0), relp)
#define STRIX_REL_MODES(Q8, QT, BN)                    \
  if (kv.q8 == Q8 && q_tiles == QT && block_n == BN) { \
    if (fused) STRIX_REL(Q8, QT, BN, 2);               \
    else STRIX_REL(Q8, QT, BN, 1);                     \
    return;                                            \
  }
  STRIX_REL_MODES(false, 1, 16)
  STRIX_REL_MODES(false, 2, 16)
  STRIX_REL_MODES(false, 2, 32)
  STRIX_REL_MODES(false, 4, 16)
  STRIX_REL_MODES(false, 4, 32)
  STRIX_REL_MODES(true, 2, 32)
  STRIX_REL_MODES(true, 4, 32)
  TORCH_CHECK(false, "inkling_rel_attn_prefill: no kernel for ", kv.q8 ? "int8" : "bf16",
              " KV, q_tiles ", q_tiles, ", block_n ", block_n);
#undef STRIX_REL_MODES
#undef STRIX_REL
}

}  // namespace strix
