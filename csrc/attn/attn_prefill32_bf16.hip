// SPDX-License-Identifier: MIT
// D=512 causal prefill with 32 query columns that share one K/V
// stage (the rdna35-fa layout), bf16 KV, no window, one KV segment.
//
// Warps have different roles per phase, which is what makes 32 columns fit:
//   * score: 8 warps = 2 column tiles x 4 D-slices, 128 dims each; partial
//     scores are summed per column tile through LDS (s_x).
//   * softmax: the ds == 0 warp of each column tile owns the row stats, writes
//     P^T and (alpha, l, m) to LDS.
//   * PV: each warp owns DV/NW = 64 V dims for BOTH column tiles, so a warp
//     holds 4 accumulator fragments per tile instead of 8, and Q holds 8
//     fragments for one tile: 128 VGPRs of state instead of 256.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int p32_kBM = 16;

template <int D, int G, int HPB, int BN>
__global__ __launch_bounds__(HPB * 8 * kWave) __attribute__((amdgpu_waves_per_eu(8))) void attn_prefill32_kernel(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    const int32_t* __restrict__ cu_q, const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ tile_off, int nseq, int64_t bt_stride,
    int block_size, int64_t sq_t, int64_t sq_h, int64_t so_t, int64_t so_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s,
    int64_t sv_h, float scale_log2) {
  constexpr int kHD = D;
  constexpr int kDS = D / 128;                // D-slices (4 at D=512)
  constexpr int kHW = D / kDS;                // dims per D-slice (128)
  constexpr int NCT = 2;                      // column tiles
  constexpr int NW = NCT * kDS;               // warps per head (8)
  constexpr int OT = D / NW;                  // V dims per warp (64)
  constexpr int kThreads = HPB * NW * kWave;
  constexpr int kPK = kHD + 8;                // K tile pitch [key][d]
  constexpr int kPV = BN + 8;                 // V^T tile pitch [d][key]
  constexpr int NS = BN / 16;                 // 16-key subtiles per step
  static_assert(D == 512, "prototype is D=512 only");
  static_assert(HPB <= 2, "per-head LDS buffers allow two heads per block");

  __shared__ __attribute__((aligned(16))) uint16_t s_k[BN * kPK];
  __shared__ __attribute__((aligned(16))) uint16_t s_vt[kHD * kPV];
  __shared__ __attribute__((aligned(16))) uint16_t s_p[HPB][NCT][p32_kBM][kPV];
  // bf16 partial scores: halves this buffer (8 KB at HPB=2) at ~3e-3
  // relative error on P, leaving LDS room for a deeper pipeline.
  __shared__ uint16_t s_x[HPB][NCT][kDS][kWave][8];
  __shared__ float s_alpha[HPB][NCT][p32_kBM], s_l[HPB][NCT][p32_kBM];
  __shared__ float s_m[HPB][NCT][p32_kBM];

  // Block geometry: NCOL column tiles of 16 queries, only the group starter
  // runs (host grid keeps one entry per 16-query tile).
  const int kvh = blockIdx.x / (G / HPB), hgroup = blockIdx.x % (G / HPB);
  const int tile = tile_off[nseq] - 1 - int(blockIdx.y);
  if (tile < 0) return;
  int n = 0;
  for (int i = 1; i < nseq; ++i) n += tile_off[i] <= tile;
  if ((tile - tile_off[n]) % NCT) return;

  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int lo = lane % 16, hi = lane / 16;
  const int head_in_block = wave / NW, w = wave % NW;
  const int ct = w % NCT, ds = w / NCT;
  const int h = kvh * G + hgroup * HPB + head_in_block;

  const int q0 = cu_q[n], q_len = cu_q[n + 1] - q0;
  const int kv_len = seq_lens[n], context = kv_len - q_len;
  const int local0 = (tile - tile_off[n]) * p32_kBM;
  const int rows = min(p32_kBM * NCT, q_len - local0);
  const int rows_c[NCT] = {min(p32_kBM, q_len - local0),
                           max(0, min(p32_kBM, q_len - local0 - p32_kBM))};
  const int last_pos = context + local0 + rows - 1;
  const int32_t* bt = block_table + int64_t(n) * bt_stride;

  // This warp's Q^T fragments: column tile ct, D-slice ds.
  bf16x16 qf[kHW / 16];
  {
    const int mr = min(lo, max(rows_c[ct], 1) - 1);
    const uint16_t* qrow =
        q + int64_t(q0 + local0 + ct * p32_kBM + mr) * sq_t + h * sq_h + ds * kHW;
#pragma unroll
    for (int kk = 0; kk < kHW / 16; ++kk)
      qf[kk] = load_frag_bf16(qrow + kk * 16, 0, 0);
  }
  // O^T accumulators: this warp owns V dims [w*OT, w*OT + OT) of both tiles.
  f32x8 o_acc[NCT][OT / 16];
#pragma unroll
  for (int c = 0; c < NCT; ++c)
#pragma unroll
    for (int j = 0; j < OT / 16; ++j) o_acc[c][j] = f32x8{};

  if (ds == 0 && hi == 0 && lo < p32_kBM) {
#pragma unroll
    for (int c = 0; c < NCT; ++c) {
      s_m[head_in_block][c][lo] = -INFINITY;
      s_l[head_in_block][c][lo] = 0.f;
      s_alpha[head_in_block][c][lo] = 1.f;
    }
  }
  __syncthreads();

  constexpr int kChunks = BN * kHD / 8;  // 16-byte K chunks and V chunks
  constexpr int kLoads = (kChunks + kThreads - 1) / kThreads;
  typedef uint32_t stage_t __attribute__((ext_vector_type(4 * kLoads)));
  stage_t rk = {}, rv = {};

  auto kv_addr = [&](int key, int64_t sb, int64_t ss, int64_t sh) {
    const int64_t blk = bt[key / block_size];
    return blk * sb + int64_t(key % block_size) * ss + kvh * sh;
  };
  // Per-step scalar block bases, as in attn_prefill: a step's BN keys span
  // at most two cache blocks when BN <= block_size.
  const bool two_blocks = BN <= block_size;
  const int ss_k = int(sk_s), ss_v = int(sv_s);
#define STRIX_FETCH32(K0)                                                    \
  const int fb0 = (K0) / block_size, fr0 = (K0) - fb0 * block_size;          \
  const bool fspan = two_blocks && fr0 + BN > block_size;                    \
  const int64_t fbA = bt[fb0], fbB = fspan ? bt[fb0 + 1] : fbA;              \
  const int64_t fkA = fbA * sk_b + kvh * sk_h, fkB = fbB * sk_b + kvh * sk_h; \
  const int64_t fvA = fbA * sv_b + kvh * sv_h, fvB = fbB * sv_b + kvh * sv_h; \
  _Pragma("unroll") for (int i = 0; i < kLoads; ++i) {                       \
    const int ch = tid + i * kThreads;                                       \
    uint4 xk = {}, xv = {};                                                  \
    if (ch < kChunks) {                                                      \
      const int koff = ch / (kHD / 8), kd = (ch % (kHD / 8)) * 8;            \
      const int voff = ch % BN, vd = (ch / BN) * 8;                          \
      const int kkey = (K0) + koff, vkey = (K0) + voff;                      \
      int64_t ka, va;                                                        \
      if (two_blocks) {                                                      \
        const int kr = fr0 + koff, vr = fr0 + voff;                          \
        const bool kn = kr >= block_size, vn = vr >= block_size;             \
        ka = (kn ? fkB : fkA) + (kn ? kr - block_size : kr) * ss_k;          \
        va = (vn ? fvB : fvA) + (vn ? vr - block_size : vr) * ss_v;          \
      } else {                                                               \
        ka = kv_addr(kkey, sk_b, sk_s, sk_h);                                \
        va = kv_addr(vkey, sv_b, sv_s, sv_h);                                \
      }                                                                      \
      if (kkey < kv_len) xk = *reinterpret_cast<const uint4*>(kc + ka + kd); \
      if (vkey < kv_len) xv = *reinterpret_cast<const uint4*>(vc + va + vd); \
    }                                                                        \
    rk[4 * i] = xk.x, rk[4 * i + 1] = xk.y, rk[4 * i + 2] = xk.z,            \
    rk[4 * i + 3] = xk.w;                                                    \
    rv[4 * i] = xv.x, rv[4 * i + 1] = xv.y, rv[4 * i + 2] = xv.z,            \
    rv[4 * i + 3] = xv.w;                                                    \
  }

  const int k_begin = 0, keys_needed = last_pos + 1;
  const int n_steps = (keys_needed + BN - 1) / BN;
  for (int step = 0; step < n_steps; ++step) {
    const int k0 = k_begin + step * BN;
    STRIX_FETCH32(k0)
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      const int ch = tid + i * kThreads;
      if (ch < kChunks) {
        const int kkey = ch / (kHD / 8), kd = (ch % (kHD / 8)) * 8;
        const int vkey = ch % BN, vd = (ch / BN) * 8;
        *reinterpret_cast<uint4*>(s_k + kkey * kPK + kd) =
            make_uint4(rk[4 * i], rk[4 * i + 1], rk[4 * i + 2], rk[4 * i + 3]);
        const uint32_t wv[4] = {rv[4 * i], rv[4 * i + 1], rv[4 * i + 2],
                                rv[4 * i + 3]};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          s_vt[(vd + 2 * e) * kPV + vkey] =
              static_cast<uint16_t>(wv[e] & 0xffffu);
          s_vt[(vd + 2 * e + 1) * kPV + vkey] =
              static_cast<uint16_t>(wv[e] >> 16);
        }
      }
    }
    __syncthreads();

    // Score: this warp's column tile over its D-slice.
    f32x8 s[NS];
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      s[j] = f32x8{};
#pragma unroll
      for (int kk = 0; kk < kHW / 16; ++kk)
        s[j] = wmma_bf16(
            load_frag_bf16(s_k + j * 16 * kPK + ds * kHW + kk * 16, kPK, lane),
            qf[kk], s[j]);
#pragma unroll
      for (int e = 0; e < 8; ++e) s[j][e] *= scale_log2;
    }
#pragma unroll
    for (int j = 0; j < NS; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        s_x[head_in_block][ct][ds][lane][e] =
            static_cast<uint16_t>(pack_bf16x2(s[j][e], 0.f) & 0xffffu);
    __syncthreads();

    // Softmax owner: the ds == 0 warp of this column tile.
    if (ds == 0) {
      const int my_pos_c =
          context + local0 + ct * p32_kBM + min(lo, max(rows_c[ct], 1) - 1);
      float m_loc = -INFINITY;
#pragma unroll
      for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float v = 0.f;
#pragma unroll
          for (int o = 0; o < kDS; ++o)
            v += __uint_as_float(uint32_t(s_x[head_in_block][ct][o][lane][e]) << 16);
          const int key = k0 + 16 * j + 2 * e + hi;
          const bool seen = key <= my_pos_c && key < kv_len;
          s[j][e] = seen ? v : -INFINITY;
          m_loc = fmaxf(m_loc, s[j][e]);
        }
      m_loc = fmaxf(m_loc, xor_shuffle_f32(m_loc, 16));
      const float m_new = fmaxf(s_m[head_in_block][ct][lo], m_loc);
      const float m_ref = m_new == -INFINITY ? 0.f : m_new;
      const float alpha = exp2f(s_m[head_in_block][ct][lo] - m_ref);
      float l_loc = 0.f;
#pragma unroll
      for (int j = 0; j < NS; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float p = exp2f(s[j][e] - m_ref);
          l_loc += p;
          // P^T row = this query: this lane owns key 2e + hi, its partner
          // (same query, opposite hi) owns the other one.
          const float other = xor_shuffle_f32(p, 16);
          s_p[head_in_block][ct][lo][16 * j + 2 * e + hi] =
              static_cast<uint16_t>(pack_bf16x2(p, 0.f) & 0xffffu);
          s_p[head_in_block][ct][lo][16 * j + 2 * e + (1 - hi)] =
              static_cast<uint16_t>(pack_bf16x2(other, 0.f) & 0xffffu);
        }
      l_loc += xor_shuffle_f32(l_loc, 16);
      s_l[head_in_block][ct][lo] = s_l[head_in_block][ct][lo] * alpha + l_loc;
      s_alpha[head_in_block][ct][lo] = alpha;
      s_m[head_in_block][ct][lo] = m_new;
    }
    __syncthreads();

    // PV: every warp covers its V slice for both column tiles.
#pragma unroll
    for (int c = 0; c < NCT; ++c) {
      const float alpha = s_alpha[head_in_block][c][lo];
#pragma unroll
      for (int j = 0; j < OT / 16; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) o_acc[c][j][e] *= alpha;
#pragma unroll
      for (int j = 0; j < NS; ++j) {
        bf16x16 pf;
        __builtin_memcpy(&pf, &s_p[head_in_block][c][lo][16 * j], 32);
#pragma unroll
        for (int dt = 0; dt < OT / 16; ++dt)
          o_acc[c][dt] = wmma_bf16(
              load_frag_bf16(s_vt + (w * OT + dt * 16) * kPV + 16 * j, kPV, lane),
              pf, o_acc[c][dt]);
      }
    }
  }
#undef STRIX_FETCH32

  // O^T[d][query] for this warp's V slice, both column tiles.
#pragma unroll
  for (int c = 0; c < NCT; ++c) {
    if (lo >= rows_c[c]) continue;
    const float inv = 1.f / s_l[head_in_block][c][lo];
    uint16_t* orow =
        out + int64_t(q0 + local0 + c * p32_kBM + lo) * so_t + h * so_h;
#pragma unroll
    for (int dt = 0; dt < OT / 16; ++dt)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        orow[w * OT + dt * 16 + 2 * e + hi] =
            static_cast<uint16_t>(pack_bf16x2(o_acc[c][dt][e] * inv, 0.f) & 0xffffu);
  }
}

}  // namespace

// out: [T, Hq, 512] bf16; q: the same; kc/vc: [blocks, bs, Hkv, 512] views;
// tile_off indexes 16-query tiles (only the first tile of each 32-query group
// runs). Not routed: production attn_prefill is faster at this geometry.
void attn_prefill32_bf16(torch::Tensor& out, const torch::Tensor& q,
                         const torch::Tensor& key_cache,
                         const torch::Tensor& value_cache,
                         const torch::Tensor& cu_seqlens_q,
                         const torch::Tensor& seq_lens,
                         const torch::Tensor& block_table,
                         const torch::Tensor& tile_offsets, int64_t max_tiles,
                         double scale) {
  const int64_t d = q.size(-1);
  TORCH_CHECK(d == 512, "attn_prefill32_bf16: head_dim must be 512");
  const int g = q.size(1) / key_cache.size(2);
  TORCH_CHECK(g == 8, "attn_prefill32_bf16: GQA must be 8");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                  key_cache.scalar_type() == at::kBFloat16 &&
                  value_cache.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                  q.stride(2) == 1,
              "bf16, contiguous out, last dim contiguous");
  for (const auto* t : {&key_cache, &value_cache})
    TORCH_CHECK(key_cache.size(1) * t->stride(1) < (int64_t(1) << 31),
                "block_size * slot stride must fit in int32");
  const float scale_log2 = static_cast<float>(scale) * 1.4426950408889634f;
  const int nseq = cu_seqlens_q.numel() - 1;
  auto stream = at::hip::getCurrentHIPStream();
  constexpr int HPB = 2;
  const dim3 grid(key_cache.size(2) * (g / HPB), max_tiles, 1);
  attn_prefill32_kernel<512, 8, HPB, 16><<<grid, HPB * 8 * kWave, 0, stream>>>(
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      reinterpret_cast<const uint16_t*>(q.data_ptr()),
      reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),
      reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),
      cu_seqlens_q.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      block_table.data_ptr<int32_t>(), tile_offsets.data_ptr<int32_t>(), nseq,
      block_table.stride(0), key_cache.size(1), q.stride(0), q.stride(1),
      out.stride(0), out.stride(1), key_cache.stride(0), key_cache.stride(1),
      key_cache.stride(2), value_cache.stride(0), value_cache.stride(1),
      value_cache.stride(2), scale_log2);
}

}  // namespace strix
