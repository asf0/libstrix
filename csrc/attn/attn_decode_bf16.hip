// SPDX-License-Identifier: MIT
// Paged attention for decode (one query token per sequence), bf16 Q/KV,
// optional sliding window, split-KV over key segments.
//
// One block owns one (sequence, KV head, key segment) and all G query heads of
// that KV head. Keys are processed KC at a time: the K/V chunk is staged in
// LDS (each cache row is read once per block), the G x KC scores are computed
// by thread groups that hold a query slice in registers, an online softmax
// runs per head, and each thread accumulates its slice of the [G][D] output
// from the staged V. Unlike
// attn_prefill (16-query tiles, mostly idle with one query), the work here
// scales with keys, so the kernel can stay at KV bandwidth.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <type_traits>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kThreads = 256;

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
    float* __restrict__ o_part, float* __restrict__ ml_part) {
  constexpr int OPT = G * D / KT;   // output values per thread
  constexpr int kChunkLoads = (KC * D / 8 + KT - 1) / KT;  // guarded
  static_assert(G * D % KT == 0 && D % OPT == 0 && OPT % 2 == 0);
  // Score phase: TPS threads share one head's dot products and each holds a
  // QS-dim slice of that head's query in registers, then loops over KPG
  // keys. Reading Q from LDS per (head, key) pair instead was ~2/3 of the
  // kernel's LDS traffic and capped D=512 at ~118 GB/s (bf16) / ~73 (int8).
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
  float qr[QS];
#pragma unroll
  for (int j = 0; j < QS / 8; ++j) {
    const uint4 w = *reinterpret_cast<const uint4*>(
        q + n * sq_t + (h0 + sc_h) * sq_h + j * TPS * 8 + lp * 8);
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      qr[j * 8 + 2 * e] = bf16_lo(ws[e]);
      qr[j * 8 + 2 * e + 1] = bf16_hi(ws[e]);
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
  const int o_flat = tid * OPT, o_h = o_flat / D, o_d = o_flat % D;
  barrier();

  // The next chunk's global loads are issued into registers before this
  // chunk's math, so they are in flight across its four barriers (barrier()
  // waits on LDS only): without it each 16-key D=512 chunk paid the full DRAM
  // latency, ~118 GB/s bf16 and ~57 int8 at depth. Raw bytes are held (the
  // int8 widening happens at the LDS store) to keep the registers small.
  using raw_t = std::conditional_t<Q8, uint2, uint4>;
  raw_t rk[kChunkLoads], rv[kChunkLoads];
  float rks[kChunkLoads], rvs[kChunkLoads];
  auto fetch = [&](int k0) {
#pragma unroll
    for (int i = 0; i < kChunkLoads; ++i) {
      const int ch = tid + i * KT;
      rk[i] = raw_t{};
      rv[i] = raw_t{};
      if (ch >= KC * D / 8) continue;
      const int key = ch / (D / 8), d8 = (ch % (D / 8)) * 8;
      if (k0 + key >= k_end) continue;
      const int kk = k0 + key;
      const int64_t blk = bt[kk / block_size], slot = kk % block_size;
      if constexpr (Q8) {
        // The byte cache's head stride is only 4-byte aligned: two words.
        const int8_t* pk = k8 + blk * sk_b + slot * sk_s + kvh * sk_h + d8;
        const int8_t* pv = v8 + blk * sv_b + slot * sv_s + kvh * sv_h + d8;
        rk[i] = make_uint2(*reinterpret_cast<const uint32_t*>(pk),
                           *reinterpret_cast<const uint32_t*>(pk + 4));
        rv[i] = make_uint2(*reinterpret_cast<const uint32_t*>(pv),
                           *reinterpret_cast<const uint32_t*>(pv + 4));
        if (d8 == 0) {
          const size_t sc = (size_t)(blk * ss_b + slot * ss_s + kvh * ss_h);
          rks[i] = ksc[sc];
          rvs[i] = vsc[sc];
        }
      } else {
        rk[i] = *reinterpret_cast<const uint4*>(kc + blk * sk_b + slot * sk_s + kvh * sk_h + d8);
        rv[i] = *reinterpret_cast<const uint4*>(vc + blk * sv_b + slot * sv_s + kvh * sv_h + d8);
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
        // Widened to bf16 without the scale (|v| <= 127 is exact); the
        // scale is applied to the score and to P below.
        uint32_t dk[4], dv[4];
        q8x8_to_bf16(rk[i].x, rk[i].y, dk);
        q8x8_to_bf16(rv[i].x, rv[i].y, dv);
        xk = make_uint4(dk[0], dk[1], dk[2], dk[3]);
        xv = make_uint4(dv[0], dv[1], dv[2], dv[3]);
        // Keys past k_end keep the initial scale of 1 (see above).
        if (d8 == 0 && k0 + key < k_end) {
          s_ks[key] = rks[i];
          s_vs[key] = rvs[i];
        }
      } else {
        xk = rk[i];
        xv = rv[i];
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
        for (int e = 0; e < 4; ++e)
          sdot += qr[j * 8 + 2 * e] * bf16_lo(ws[e]) +
                  qr[j * 8 + 2 * e + 1] * bf16_hi(ws[e]);
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
    // lane shuffles. It was one thread per head looping over the KC keys
    // while the rest of the block waited at the barrier - the fixed cost per
    // chunk that capped int8 decode (half the bytes per chunk) at ~150 GB/s.
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
    const float alpha = s_alpha[o_h];
#pragma unroll
    for (int i = 0; i < OPT; ++i) acc[i] *= alpha;
    for (int k = 0; k < KC; ++k) {
      const float p = s_p[o_h][k];
      const uint16_t* vr = s_v + k * D + o_d;
#pragma unroll
      for (int i = 0; i < OPT; i += 2) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(vr + i);
        acc[i] += p * bf16_lo(w);
        acc[i + 1] += p * bf16_hi(w);
      }
    }
    barrier();  // the next chunk overwrites s_k, s_v, s_p
  }

  const int h = h0 + o_h;
  if (nseg > 1) {
    const int hq = gridDim.y * G;
    const int64_t r = (int64_t(n) * hq + h) * nseg + seg;
#pragma unroll
    for (int i = 0; i < OPT; ++i) o_part[r * D + o_d + i] = acc[i];
    if (o_d == 0) {
      ml_part[2 * r] = s_m[o_h];
      ml_part[2 * r + 1] = s_l[o_h];
    }
  } else {
    const float inv = 1.f / s_l[o_h];
    uint16_t* orow = out + n * so_t + h * so_h + o_d;
#pragma unroll
    for (int i = 0; i < OPT; i += 2)
      *reinterpret_cast<uint32_t*>(orow + i) = pack_bf16x2(acc[i] * inv, acc[i + 1] * inv);
  }
}

// Combines split-KV partials: one block per (token, head), one thread per dim.
template <int D>
__global__ __launch_bounds__(D) void attn_decode_reduce_kernel(
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
    acc += w * o_part[(row * nseg + s) * D + threadIdx.x];
  }
  out[blockIdx.x * so_t + blockIdx.y * so_h + threadIdx.x] =
      static_cast<uint16_t>(pack_bf16x2(acc / l, 0.f) & 0xffffu);
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
                      const std::optional<torch::Tensor>& v_scale) {
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
      static_cast<int>(window_left), static_cast<int>(num_splits), op, mp)
  // Gemma-4 sliding (256, GQA 2) plus (256, 8) for the 35B and 27B bf16,
  // which otherwise fall back to Triton decode. GQA 6 (the served 27B INT8)
  // cannot use this kernel: its layout needs G*KC to divide 256 threads and
  // the loader needs KC*D/8 divisible by the thread count, neither of which
  // admits a factor of 3.
  if (q8) {
    if (d == 256 && g == 2) STRIX_DECODE(256, 2, 32, true, 256);
    else if (d == 256 && g == 8) STRIX_DECODE(256, 8, 32, true, 256);
    else if (d == 256 && g == 6) STRIX_DECODE(256, 6, 32, true, 192);
    else if (d == 512 && g == 8) STRIX_DECODE(512, 8, 16, true, 256);
    else TORCH_CHECK(false, "attn_decode_bf16 int8: head_dim ", d, " GQA ", g,
                     " not built (256: 2, 6, 8; 512: 8)");
  } else {
    if (d == 256 && g == 2) STRIX_DECODE(256, 2, 32, false, 256);
    else if (d == 256 && g == 8) STRIX_DECODE(256, 8, 32, false, 256);
    else if (d == 256 && g == 6) STRIX_DECODE(256, 6, 32, false, 192);
    else if (d == 512 && g == 8) STRIX_DECODE(512, 8, 16, false, 256);
    else TORCH_CHECK(false, "attn_decode_bf16: head_dim ", d, " GQA ", g,
                     " not built (256: 2, 6, 8; 512: 8)");
  }
#undef STRIX_DECODE
  if (num_splits > 1) {
    auto reduce = d == 256 ? attn_decode_reduce_kernel<256> : attn_decode_reduce_kernel<512>;
    reduce<<<dim3(nseq, hq), d, 0, stream>>>(
        reinterpret_cast<uint16_t*>(out.data_ptr()), op, mp,
        static_cast<int>(num_splits), hq, out.stride(0), out.stride(1));
  }
}

}  // namespace strix
