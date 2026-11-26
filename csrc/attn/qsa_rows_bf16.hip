// SPDX-License-Identifier: MIT
// Qwen3.8-Flash-Next QSA sparse attention, one query row per wave pair: the
// row's 12 query heads (GQA 12) share its selected tokens, so they are the 16
// columns of the WMMAs (12 used) and every score computed is one the row
// asked for. vLLM's Triton kernel does the same per row with 16-key steps and
// little latency hiding; the union kernel (qsa_union_bf16.hip) shares one
// gather between 16 rows but scores each row against the whole union, 4-5x
// the needed work on real selections (a 16-row union is 4.3-5.5x one row's
// 2051 tokens). Neighbouring rows select mostly the same tokens and run at the
// same time, so their gathers hit L2 / MALL.
//
// Layout as qsa_union_attn_kernel: S^T = K Q^T (keys as WMMA rows, heads as
// columns) over two waves that split head_dim, O^T = V^T P^T with V^T staged
// in LDS. K fragments load straight from the cache (a lane's 16 elements are
// contiguous); V is transposed through LDS. Any batch: token_to_req maps each
// row to its block-table row.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

// RB rows x HB KV heads per block, one wave pair each. One row with both of
// its KV heads (a token's K / V for the two heads are neighbours) and the
// 16-key V^T pitch measured best: 117 ms per layer on a real 15k-row chunk at
// 135k context, against 124 (pitch 24), 130 (2 rows) and 174 (1 row, 1 head).
template <int D, int G, int RB, int HB, int PV = 16>
__global__ __launch_bounds__(RB * HB * (D / 128) * kWave)
__attribute__((amdgpu_waves_per_eu(4))) void qsa_rows_attn_kernel(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    const int32_t* __restrict__ idx, int64_t ld_idx, int width,
    const int32_t* __restrict__ bt, int64_t ld_bt,
    const int32_t* __restrict__ tok2req, int rows, int block_size,
    int64_t sq_t, int64_t sq_h, int64_t so_t, int64_t so_h, int64_t sk_b,
    int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s, int64_t sv_h,
    float scale_log2) {
  constexpr int BN = 16, kHalf = 128, W = D / kHalf;
  constexpr int kPV = PV;  // V^T tile [d][key] pitch (>= BN, a multiple of 8)
  static_assert(kPV >= BN && kPV % 8 == 0, "V^T rows hold BN keys, 16-byte aligned");
  static_assert(G <= 16, "query heads per KV head must fit the 16 columns");
  __shared__ __attribute__((aligned(16))) uint16_t s_vt[RB * HB][D * kPV];
  __shared__ _Float16 s_x[RB * HB * W][kWave][8];

  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int pair = wave / W, half = wave % W, lo = lane % 16, hi = lane / 16;
  const int kvh = HB == 1 ? blockIdx.y : pair % HB;
  const int ptid = tid % (W * kWave);  // thread within the pair
  const int row = blockIdx.x * RB + pair / HB;
  const bool row_ok = row < rows;
  const int safe_row = row_ok ? row : rows - 1;
  const int req = tok2req ? tok2req[safe_row] : 0;
  const int32_t* brow = bt + int64_t(req) * ld_bt;
  const int32_t* irow = idx + int64_t(safe_row) * ld_idx;
  // Column lo is query head lo of this KV head; columns >= G repeat head 0
  // and are never stored.
  const int h = kvh * G + (lo < G ? lo : 0);

  bf16x16 qf[kHalf / 16];
  {
    const uint16_t* qrow = q + int64_t(safe_row) * sq_t + h * sq_h + half * kHalf;
#pragma unroll
    for (int kk = 0; kk < kHalf / 16; ++kk) qf[kk] = load_frag_bf16(qrow + kk * 16, 0, 0);
  }
  f32x8 o_acc[kHalf / 16];
#pragma unroll
  for (int i = 0; i < kHalf / 16; ++i) o_acc[i] = f32x8{};
  float m_run = -INFINITY, l_run = 0.f;
  const uint32_t p_sel = hi ? 0x03020706u : 0x07060302u;

  // V: 16 keys x 256 d = 512 16-byte chunks over the pair's 64 threads; a
  // thread's chunks are 8 pieces of one key (one page per thread: spreading a
  // thread over 8 keys was 25 % slower).
  constexpr int kChunks = BN * D / 8;
  constexpr int kLoads = kChunks / (W * kWave);
  uint4 rv[kLoads];
  auto token = [&](int k) { return k < width ? irow[k] : -1; };
  auto page_of = [&](int t) { return t >= 0 ? brow[t / block_size] : 0; };
  // A thread's V chunks are all one key (ch % BN = ptid % BN). Each key's
  // index and page are loaded a step ahead of its data, so a step waits on
  // one memory round trip instead of three (index -> page -> data).
  const int vkey = ptid % BN;
  auto fetch_v = [&](int t, int pg) {
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      const int ch = ptid + i * W * kWave;
      const int vd = (ch / BN) * 8;
      uint4 x = {};
      if (t >= 0)
        x = *reinterpret_cast<const uint4*>(vc + int64_t(pg) * sv_b +
                                            int64_t(t % block_size) * sv_s + kvh * sv_h + vd);
      rv[i] = x;
    }
  };

  // The row's selection ends at its last valid entry: rows early in a prompt
  // select fewer tokens than the width and are -1 padded. Found from the end,
  // 32 entries per ballot; every wave of a one-row block gets the same count.
  int n_valid = 0;
  if (row_ok) {
    for (int base = ((width - 1) / kWave) * kWave; base >= 0; base -= kWave) {
      const int k = base + lane;
      const uint64_t m = __ballot(k < width && irow[k] >= 0);
      if (m) {
        n_valid = base + 64 - __builtin_clzll(m);
        break;
      }
    }
  }
  const int n_steps = (n_valid + BN - 1) / BN;
  // All pairs of the block run the same number of steps (barriers): one row
  // per block, so its count (RB > 1 would need the block's maximum).
  static_assert(RB == 1, "the step count is the row's own");
  const int max_steps = n_steps;
  int vt = token(vkey), vpg = page_of(vt);
  if (n_steps > 0) fetch_v(vt, vpg);
  vt = token(BN + vkey);
  vpg = page_of(vt);
  // K fragments load at the start of each step (loading them a step earlier
  // keeps them live through the softmax: 145 against 127 ms); their index and
  // page a step ahead, issued before the step's math (after it: 140 ms).
  bf16x16 kf[kHalf / 16];
  auto load_k = [&](int t, int pg) {
    const uint16_t* kp = t >= 0 ? kc + int64_t(pg) * sk_b +
                                                     int64_t(t % block_size) * sk_s +
                                                     kvh * sk_h + half * kHalf
                                               : nullptr;
#pragma unroll
    for (int kk = 0; kk < kHalf / 16; ++kk) {
      if (kp) {
        kf[kk] = load_frag_bf16(kp + kk * 16, 0, 0);
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) kf[kk][i] = 0;
      }
    }
  };
  int kt = token(lo), kpg = page_of(kt);  // this step's key of this lane
  for (int step = 0; step < max_steps; ++step) {
    const int k0 = step * BN;
    load_k(kt, kpg);
    barrier();  // the previous step is done with s_vt / s_x
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      const int ch = ptid + i * W * kWave;
      const int vkey = ch % BN, vd = (ch / BN) * 8;
      const uint32_t w[4] = {rv[i].x, rv[i].y, rv[i].z, rv[i].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        s_vt[pair][(vd + 2 * e) * kPV + vkey] = static_cast<uint16_t>(w[e] & 0xffffu);
        s_vt[pair][(vd + 2 * e + 1) * kPV + vkey] = static_cast<uint16_t>(w[e] >> 16);
      }
    }
    // The next step's K index and page, issued before this step's math.
    kt = token(k0 + BN + lo);
    kpg = page_of(kt);
    if (step + 1 < n_steps) {
      fetch_v(vt, vpg);
      vt = token(k0 + 2 * BN + vkey);
      vpg = page_of(vt);
    }

    f32x8 s = f32x8{};
#pragma unroll
    for (int kk = 0; kk < kHalf / 16; ++kk) s = wmma_bf16(kf[kk], qf[kk], s);

#pragma unroll
    for (int e = 0; e < 8; ++e) {
      s[e] *= scale_log2;
      s_x[wave][lane][e] = static_cast<_Float16>(s[e]);
    }
    barrier();

    // Element e is key k0 + 2 e + hi of this row, column lo.
    float m_loc = -INFINITY;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float v = s[e];
#pragma unroll
      for (int o = 1; o < W; ++o)
        v += static_cast<float>(s_x[pair * W + (half + o) % W][lane][e]);
      const bool seen = step < n_steps && token(k0 + 2 * e + hi) >= 0;
      s[e] = seen ? v : -INFINITY;
      m_loc = fmaxf(m_loc, s[e]);
    }
    m_loc = fmaxf(m_loc, xor_shuffle_f32(m_loc, 16));
    const float m_new = fmaxf(m_run, m_loc);
    const float m_ref = m_new == -INFINITY ? 0.f : m_new;
    const float alpha = __builtin_amdgcn_exp2f(m_run - m_ref);
    if (__any(alpha != 1.f)) {
#pragma unroll
      for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
        for (int e = 0; e < 8; ++e) o_acc[dt][e] *= alpha;
    }
    float l_loc = 0.f;
    bf16x16 pf;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float p = __builtin_amdgcn_exp2f(s[e] - m_ref);
      l_loc += p;
      const uint32_t mine = round_bf16_hi(p);
      const uint32_t other = __float_as_uint(xor_shuffle_f32(__uint_as_float(mine), 16));
      const uint32_t pk = __builtin_amdgcn_perm(other, mine, p_sel);
      pf[2 * e] = static_cast<short>(pk & 0xffffu);
      pf[2 * e + 1] = static_cast<short>(pk >> 16);
    }
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt) {
      const bf16x16 vf =
          load_frag_bf16(&s_vt[pair][(half * kHalf + dt * 16) * kPV], kPV, lane);
      o_acc[dt] = wmma_bf16(vf, pf, o_acc[dt]);
    }
    l_loc += xor_shuffle_f32(l_loc, 16);
    l_run = l_run * alpha + l_loc;
    m_run = m_new;
  }

  if (row_ok && lo < G) {
    const float inv = l_run > 0.f ? 1.f / l_run : 0.f;
    uint16_t* orow = out + int64_t(row) * so_t + h * so_h + half * kHalf;
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
      for (int e = 0; e < 8; ++e)
        orow[dt * 16 + 2 * e + hi] =
            static_cast<uint16_t>(pack_bf16x2(o_acc[dt][e] * inv, 0.f) & 0xffffu);
  }
}

}  // namespace

// out, q: [rows, Hq, 256] bf16 (last dim contiguous), Hq = 12 * Hkv;
// key_cache / value_cache: [num_blocks, block_size, Hkv, 256] bf16 views;
// indices: [rows, width] int32 request-relative token positions, -1 padded;
// block_table: [reqs, max_blocks] int32; token_to_req: [rows] int32 (or None
// for one request).
void qsa_rows_attn_bf16(torch::Tensor& out, const torch::Tensor& q,
                        const torch::Tensor& key_cache, const torch::Tensor& value_cache,
                        const torch::Tensor& indices, const torch::Tensor& block_table,
                        const std::optional<torch::Tensor>& token_to_req) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 &&
                  key_cache.scalar_type() == at::kBFloat16 &&
                  value_cache.scalar_type() == at::kBFloat16,
              "qsa_rows_attn_bf16: bf16 q, out and caches");
  TORCH_CHECK(indices.scalar_type() == at::kInt && block_table.scalar_type() == at::kInt,
              "int32 indices and block table");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 256 && q.stride(2) == 1 && out.stride(2) == 1,
              "q / out must be [rows, Hq, 256] with contiguous head_dim");
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.size(3) == 256 &&
                  key_cache.stride(3) == 1 && value_cache.stride(3) == 1,
              "caches must be [blocks, block_size, Hkv, 256]");
  const int64_t rows = q.size(0), hkv = key_cache.size(2);
  TORCH_CHECK(q.size(1) == 12 * hkv, "GQA 12 only, got ", q.size(1), "/", hkv);
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == rows && indices.stride(1) == 1,
              "indices must be [rows, width]");
  TORCH_CHECK(block_table.dim() == 2 && block_table.stride(1) == 1,
              "block table must be [reqs, max_blocks]");
  const int32_t* t2r = nullptr;
  if (token_to_req.has_value()) {
    TORCH_CHECK(token_to_req->scalar_type() == at::kInt && token_to_req->dim() == 1 &&
                    token_to_req->size(0) >= rows && token_to_req->stride(0) == 1,
                "token_to_req must be [rows] int32");
    t2r = token_to_req->data_ptr<int32_t>();
  } else {
    TORCH_CHECK(block_table.size(0) == 1, "several requests need token_to_req");
  }
  if (rows == 0) return;
  constexpr int D = 256, G = 12;
  const float scale_log2 = 1.4426950408889634f / 16.f;  // log2(e) / sqrt(256)
  auto stream = at::hip::getCurrentHIPStream();
  constexpr int RB = 1;
  const int HB = hkv == 2 ? 2 : 1;
  auto kernel = HB == 2 ? qsa_rows_attn_kernel<D, G, RB, 2> : qsa_rows_attn_kernel<D, G, RB, 1>;
  kernel<<<dim3((rows + RB - 1) / RB, hkv / HB), RB * HB * (D / 128) * kWave, 0, stream>>>(
          reinterpret_cast<uint16_t*>(out.data_ptr()),
          reinterpret_cast<const uint16_t*>(q.data_ptr()),
          reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),
          reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),
          indices.data_ptr<int32_t>(), indices.stride(0), int(indices.size(1)),
          block_table.data_ptr<int32_t>(), block_table.stride(0), t2r, int(rows),
          int(key_cache.size(1)), q.stride(0), q.stride(1), out.stride(0), out.stride(1),
          key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
          value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), scale_log2);
}

}  // namespace strix
