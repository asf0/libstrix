// SPDX-License-Identifier: MIT
// Qwen3.8-Flash-Next QSA sparse attention for a single-request prefill, over
// row unions: every query row attends to its own TOPK selected tokens (the
// indexer's choice, request-relative positions, -1 padded), and consecutive
// rows select mostly the same tokens. A block owns ROWS rows (16 by default):
// it gathers the union of their selections once (3.38x one row's size for 16
// rows of a real 26k-token prompt, so 4.7x fewer K/V bytes than 16 separate
// rows) and masks each score by whether that row selected the key. The math
// per (row, head) is the same softmax over the row's own selection; the cost
// is the masked-out scores, so the kernel is compute-bound (~24 TF).
//
// qsa_union_build_kernel: per group, a bitmap of the selected positions in
// LDS, a prefix popcount, then the sorted union list and one bit per row for
// every union entry.
//
// qsa_union_attn_kernel: attn_prefill_kernel's layout (bf16, head_dim 256,
// two waves per head splitting head_dim, S^T = K Q^T and O^T = V^T P^T so a
// lane owns one query column) with 16 columns = 4 rows x 4 heads per wave
// pair, 3 head groups for GQA 12 times ROWS / 4 row sets; keys come from the
// union list, the mask from the membership bits.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <cstdlib>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kMaxWords = 4096;   // bitmap words: positions < 131072

// ROWS query rows per group (4, 8 or 16; one membership bit each, uint16).
template <int T, int ROWS>
__global__ __launch_bounds__(T) void qsa_union_build_kernel(
    const int32_t* __restrict__ idx, int64_t ld_idx, int rows, int width,
    int32_t* __restrict__ uni, uint32_t* __restrict__ mem32,
    int32_t* __restrict__ ulen, int64_t u_cap, int words) {
  __shared__ uint32_t s_bits[kMaxWords];
  __shared__ int32_t s_pre[kMaxWords];
  __shared__ int32_t s_scan[T];
  const int group = blockIdx.x, tid = threadIdx.x;
  const int r0 = group * ROWS;
  uint32_t* gmem = mem32 + int64_t(group) * (u_cap / 2);
  for (int w = tid; w < words; w += T) s_bits[w] = 0;
  for (int w = tid; w < u_cap / 2; w += T) gmem[w] = 0;
  __syncthreads();
  const int limit = words * 32;
  for (int i = tid; i < ROWS * width; i += T) {
    const int r = i / width, row = r0 + r;
    if (row >= rows) continue;
    const int t = idx[int64_t(row) * ld_idx + i % width];
    if (t >= 0 && t < limit) atomicOr(&s_bits[t >> 5], 1u << (t & 31));
  }
  __syncthreads();
  // Prefix popcount: thread tid owns words [w0, w1).
  const int per = (words + T - 1) / T;
  const int w0 = min(words, tid * per), w1 = min(words, w0 + per);
  int local = 0;
  for (int w = w0; w < w1; ++w) local += __popc(s_bits[w]);
  s_scan[tid] = local;
  __syncthreads();
  for (int off = 1; off < T; off <<= 1) {
    const int v = tid >= off ? s_scan[tid - off] : 0;
    __syncthreads();
    s_scan[tid] += v;
    __syncthreads();
  }
  int run = s_scan[tid] - local;  // exclusive
  for (int w = w0; w < w1; ++w) {
    s_pre[w] = run;
    uint32_t b = s_bits[w];
    int p = run;
    while (b) {
      uni[int64_t(group) * u_cap + p++] = w * 32 + __ffs(b) - 1;
      b &= b - 1;
    }
    run = p;
  }
  if (tid == T - 1) ulen[group] = s_scan[T - 1];
  __syncthreads();
  for (int i = tid; i < ROWS * width; i += T) {
    const int r = i / width, row = r0 + r;
    if (row >= rows) continue;
    const int t = idx[int64_t(row) * ld_idx + i % width];
    if (t < 0 || t >= limit) continue;
    const int w = t >> 5;
    const int rank = s_pre[w] + __popc(s_bits[w] & ((1u << (t & 31)) - 1u));
    atomicOr(&gmem[rank >> 1], (1u << r) << (16 * (rank & 1)));
  }
}

// Wave pair p covers head group p % (G / 4) of rows 4 * (p / (G / 4)) ..+3.
// VGPRs capped at 192 (8 waves per SIMD), as in attn_prefill_kernel: at the
// ~250 the compiler picks on its own, one block fills a CU and every barrier
// bubble stays empty.
template <int D, int G, int BN, int ROWS>
__global__ __launch_bounds__((G / 4) * (ROWS / 4) * (D / 128) * kWave)
__attribute__((amdgpu_waves_per_eu(8))) void qsa_union_attn_kernel(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    const int32_t* __restrict__ uni, const uint16_t* __restrict__ mem,
    const int32_t* __restrict__ ulen, const int32_t* __restrict__ bt, int rows,
    int64_t u_cap, int block_size, int64_t sq_t, int64_t sq_h, int64_t so_t,
    int64_t so_h, int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b,
    int64_t sv_s, int64_t sv_h, float scale_log2) {
  constexpr int kHalf = 128, W = D / kHalf;  // waves per head group
  constexpr int HG = G / 4;                  // head groups of 4 heads
  constexpr int HPB = HG * (ROWS / 4);       // wave pairs
  constexpr int kThreads = HPB * W * kWave;
  constexpr int kPK = D + 8;    // K tile [key][d] pitch
  constexpr int kPV = BN + 8;   // V^T tile [d][key] pitch
  constexpr int NS = BN / 16;
  __shared__ __attribute__((aligned(16))) uint16_t s_k[BN * kPK];
  __shared__ __attribute__((aligned(16))) uint16_t s_vt[D * kPV];
  __shared__ float s_x[W * HPB][NS][kWave][8];
  __shared__ uint16_t s_mem[BN];

  const int kvh = blockIdx.x, group = blockIdx.y;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int pair = wave / W, half = wave % W, lo = lane % 16, hi = lane / 16;
  // Column lo: row 4 * (pair / HG) + lo / 4 of the group, head lo % 4 of
  // head group pair % HG.
  const int r = 4 * (pair / HG) + (lo >> 2);
  const int row = group * ROWS + r;
  const bool row_ok = row < rows;
  const int safe_row = row_ok ? row : rows - 1;
  const int h = kvh * G + (pair % HG) * 4 + (lo & 3);
  const int n_keys = ulen[group];
  const int32_t* guni = uni + int64_t(group) * u_cap;
  const uint16_t* gmem = mem + int64_t(group) * u_cap;

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

  constexpr int kChunks = BN * D / 8;
  constexpr int kLoads = (kChunks + kThreads - 1) / kThreads;
  uint4 rk[kLoads], rv[kLoads];
  uint16_t rm[kLoads];
  auto fetch = [&](int k0) {
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      const int ch = tid + i * kThreads;
      uint4 xk = {}, xv = {};
      uint16_t xm = 0;
      if (ch < kChunks) {
        const int koff = ch / (D / 8), kd = (ch % (D / 8)) * 8;
        const int voff = ch % BN, vd = (ch / BN) * 8;
        const int ki = k0 + koff, vi = k0 + voff;
        if (ki < n_keys) {
          const int t = guni[ki];
          const int64_t a = int64_t(bt[t / block_size]) * sk_b +
                            int64_t(t % block_size) * sk_s + kvh * sk_h + kd;
          xk = *reinterpret_cast<const uint4*>(kc + a);
          if (kd == 0) xm = gmem[ki];
        }
        if (vi < n_keys) {
          const int t = guni[vi];
          const int64_t a = int64_t(bt[t / block_size]) * sv_b +
                            int64_t(t % block_size) * sv_s + kvh * sv_h + vd;
          xv = *reinterpret_cast<const uint4*>(vc + a);
        }
      }
      rk[i] = xk;
      rv[i] = xv;
      rm[i] = xm;
    }
  };

  const int n_steps = (n_keys + BN - 1) / BN;
  if (n_steps > 0) fetch(0);
  for (int step = 0; step < n_steps; ++step) {
    const int k0 = step * BN;
    barrier();  // previous step is done with s_k / s_vt / s_x / s_mem
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      const int ch = tid + i * kThreads;
      if (ch < kChunks) {
        const int kkey = ch / (D / 8), kd = (ch % (D / 8)) * 8;
        const int vkey = ch % BN, vd = (ch / BN) * 8;
        *reinterpret_cast<uint4*>(s_k + kkey * kPK + kd) = rk[i];
        if (kd == 0) s_mem[kkey] = rm[i];
        const uint32_t w[4] = {rv[i].x, rv[i].y, rv[i].z, rv[i].w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          s_vt[(vd + 2 * e) * kPV + vkey] = static_cast<uint16_t>(w[e] & 0xffffu);
          s_vt[(vd + 2 * e + 1) * kPV + vkey] = static_cast<uint16_t>(w[e] >> 16);
        }
      }
    }
    barrier();
    if (step + 1 < n_steps) fetch(k0 + BN);

    f32x8 s[NS];
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      s[j] = f32x8{};
#pragma unroll
      for (int kk = 0; kk < kHalf / 16; ++kk) {
        const bf16x16 kf =
            load_frag_bf16(s_k + j * 16 * kPK + half * kHalf + kk * 16, kPK, lane);
        s[j] = wmma_bf16(kf, qf[kk], s[j]);
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) s[j][e] *= scale_log2;
    }
#pragma unroll
    for (int j = 0; j < NS; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) s_x[wave][j][lane][e] = s[j][e];
    barrier();

    // Element e of subtile j is union entry k0 + 16 j + 2 e + hi.
    float m_loc = -INFINITY;
#pragma unroll
    for (int j = 0; j < NS; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int kloc = 16 * j + 2 * e + hi;
        float v = s[j][e];
#pragma unroll
        for (int o = 1; o < W; ++o) v += s_x[pair * W + (half + o) % W][j][lane][e];
        const bool seen =
            row_ok && k0 + kloc < n_keys && ((s_mem[kloc] >> r) & 1u);
        s[j][e] = seen ? v : -INFINITY;
        m_loc = fmaxf(m_loc, s[j][e]);
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
#pragma unroll
    for (int j = 0; j < NS; ++j) {
      bf16x16 pf;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float p = __builtin_amdgcn_exp2f(s[j][e] - m_ref);
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
            load_frag_bf16(s_vt + (half * kHalf + dt * 16) * kPV + j * 16, kPV, lane);
        o_acc[dt] = wmma_bf16(vf, pf, o_acc[dt]);
      }
    }
    l_loc += xor_shuffle_f32(l_loc, 16);
    l_run = l_run * alpha + l_loc;
    m_run = m_new;
  }

  // O^T[d][column lo]: element e of tile dt is d = dt*16 + 2e + hi. A row
  // that selected nothing gets zeros, like the Triton kernel.
  if (row_ok) {
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
// block_table: [max_blocks] int32 of the one request. Positions must be below
// 131072 (the union bitmap); the caller keeps longer requests on Triton.
void qsa_union_attn_bf16(torch::Tensor& out, const torch::Tensor& q,
                         const torch::Tensor& key_cache,
                         const torch::Tensor& value_cache,
                         const torch::Tensor& indices,
                         const torch::Tensor& block_table) {
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 &&
                  key_cache.scalar_type() == at::kBFloat16 &&
                  value_cache.scalar_type() == at::kBFloat16,
              "qsa_union_attn_bf16: bf16 q, out and caches");
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
  TORCH_CHECK(block_table.dim() == 1 && block_table.stride(0) == 1, "block table 1-D");
  const int block_size = int(key_cache.size(1));
  const int64_t capacity = block_table.size(0) * int64_t(block_size);
  const int words = int((std::min<int64_t>(capacity, kMaxWords * 32) + 31) / 32);
  if (rows == 0) return;
  const int width = int(indices.size(1));
  // Larger groups gather a union that grows slower than their row count
  // (one row's selection x1.75 for 4 rows, x2.34 for 8, x3.38 for 16), and
  // more waves share each staged K/V tile.
  // 16-row groups: 1.39x the Triton per-row kernel on real 26k-token
  // selections against 1.34x (8 rows) and 1.22x (4); STRIX_QSA_UNION_ROWS
  // picks another for A/B runs.
  static const int rows_per = [] {
    const char* e = std::getenv("STRIX_QSA_UNION_ROWS");
    const int v = e ? std::atoi(e) : 16;
    return v == 4 || v == 8 ? v : 16;
  }();
  const int groups = int((rows + rows_per - 1) / rows_per);
  const int64_t u_cap = (int64_t(rows_per) * width + 3) / 4 * 4;
  auto opts = q.options();
  auto uni = at::empty({groups, u_cap}, opts.dtype(at::kInt));
  auto mem = at::empty({groups, u_cap}, opts.dtype(at::kShort));
  auto ulen = at::empty({groups}, opts.dtype(at::kInt));
  auto stream = at::hip::getCurrentHIPStream();
  constexpr int TB = 256;
  auto build = rows_per == 4   ? qsa_union_build_kernel<TB, 4>
               : rows_per == 8 ? qsa_union_build_kernel<TB, 8>
                               : qsa_union_build_kernel<TB, 16>;
  build<<<groups, TB, 0, stream>>>(
      indices.data_ptr<int32_t>(), indices.stride(0), int(rows), width,
      uni.data_ptr<int32_t>(), reinterpret_cast<uint32_t*>(mem.data_ptr()),
      ulen.data_ptr<int32_t>(), u_cap, words);
  constexpr int D = 256, G = 12;
  const float scale_log2 = 1.4426950408889634f / 16.f;  // log2(e) / sqrt(256)
  static const int bn = [] {
    const char* e = std::getenv("STRIX_QSA_UNION_BN");
    return e ? std::atoi(e) : 16;
  }();
  auto kernel = rows_per == 4    ? qsa_union_attn_kernel<D, G, 16, 4>
                : rows_per == 16 ? qsa_union_attn_kernel<D, G, 16, 16>
                : bn == 32       ? qsa_union_attn_kernel<D, G, 32, 8>
                                 : qsa_union_attn_kernel<D, G, 16, 8>;
  kernel<<<dim3(hkv, groups), (G / 4) * (rows_per / 4) * (D / 128) * kWave, 0, stream>>>(
          reinterpret_cast<uint16_t*>(out.data_ptr()),
          reinterpret_cast<const uint16_t*>(q.data_ptr()),
          reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),
          reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),
          uni.data_ptr<int32_t>(), reinterpret_cast<const uint16_t*>(mem.data_ptr()),
          ulen.data_ptr<int32_t>(),
          block_table.data_ptr<int32_t>(), int(rows), u_cap, block_size, q.stride(0),
          q.stride(1), out.stride(0), out.stride(1), key_cache.stride(0),
          key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),
          value_cache.stride(1), value_cache.stride(2), scale_log2);
}

}  // namespace strix
