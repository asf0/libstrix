// SPDX-License-Identifier: MIT
// Qwen3.8-Flash-Next QSA indexer scores for a single-request prefill:
//   logits[r, c] = sum_h relu(q[r, h] . k[c]) / divisor   for c < visible[r]
//   visible[r]   = min((pos[r] + 1) / ratio, seq_len / ratio)
// and -inf past it, over the paged compressed-key cache (one KV head,
// head_dim 128). The work is quadratic in the context (at 96k tokens ~2.4
// TFLOP per 16k-row chunk), where vLLM's Triton tl.dot version runs ~5 TF.
//
// A block owns 64 rows x 64 keys; the key tile is staged in LDS once and
// serves all heads, Q fragments are read from global (reused across key
// tiles through L2). 1.7x vLLM's Triton tl.dot kernel from 4k to 128k. Tiles past the last row's
// visible keys store -inf only.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kD = 128;           // indexer head_dim
constexpr int kPitch = kD + 8;    // LDS key-tile pitch

template <int BM, int BN, int WM, int WN, int H>
__global__ __launch_bounds__(WM* WN* kWave) void qsa_indexer_kernel(
    float* __restrict__ logits, int32_t* __restrict__ visible_out,
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ kc,
    const int32_t* __restrict__ bt, const int64_t* __restrict__ pos,
    const int32_t* __restrict__ seq_len_ptr, int rows, int columns, int ratio, int block_size, int64_t sq_r,
    int64_t sq_h, int64_t sk_b, int64_t sk_s, int64_t ld_logits, float inv_div) {
  constexpr int kThreads = WM * WN * kWave;
  constexpr int TM = BM / WM, TN = BN / WN, FM = TM / 16, FN = TN / 16;
  constexpr int kChunks = BN * kD / 8;
  constexpr int kLoads = (kChunks + kThreads - 1) / kThreads;
  __shared__ __attribute__((aligned(16))) uint16_t s_k[BN * kPitch];

  const int r0 = blockIdx.x * BM, c0 = blockIdx.y * BN;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int wm = wave % WM, wn = wave / WM;
  const int seq_vis = *seq_len_ptr / ratio;
  auto visible_of = [&](int r) -> int {
    return int(min<int64_t>((pos[r] + 1) / ratio, seq_vis));
  };
  if (blockIdx.y == 0)
    for (int i = tid; i < BM; i += kThreads)
      if (r0 + i < rows) visible_out[r0 + i] = visible_of(r0 + i);
  const int last = min(rows, r0 + BM) - 1;
  // Positions grow with the row within a request: the last row sees most.
  int max_vis = 0;
  for (int i = r0; i <= last; ++i) max_vis = max(max_vis, visible_of(i));
  if (c0 >= max_vis) {
    for (int i = tid; i < BM * BN; i += kThreads) {
      const int r = r0 + i / BN, c = c0 + i % BN;
      if (r < rows && c < columns) logits[int64_t(r) * ld_logits + c] = -INFINITY;
    }
    return;
  }

#pragma unroll
  for (int i = 0; i < kLoads; ++i) {
    const int ch = tid + i * kThreads;
    if (ch < kChunks) {
      const int key = ch / (kD / 8), d = (ch % (kD / 8)) * 8;
      const int c = c0 + key;
      uint4 v = {};
      if (c < max_vis) {
        const int64_t page = bt[c / block_size];
        v = *reinterpret_cast<const uint4*>(kc + page * sk_b + int64_t(c % block_size) * sk_s + d);
      }
      *reinterpret_cast<uint4*>(s_k + key * kPitch + d) = v;
    }
  }
  __syncthreads();

  f32x8 score[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) score[i][j] = f32x8{};
  // Rows past the batch read the last valid row (never stored).
  int qr[FM];
#pragma unroll
  for (int i = 0; i < FM; ++i) qr[i] = min(r0 + wm * TM + i * 16, rows - 1);
#pragma unroll
  for (int h = 0; h < H; ++h) {
    f32x8 acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
      for (int j = 0; j < FN; ++j) acc[i][j] = f32x8{};
#pragma unroll
    for (int kk = 0; kk < kD; kk += 16) {
      bf16x16 fb[FN];
#pragma unroll
      for (int j = 0; j < FN; ++j)
        fb[j] = load_frag_bf16(s_k + (wn * TN + j * 16) * kPitch + kk, kPitch, lane);
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        // A fragment from global: lane row lo of the 16-row slab, clamped.
        // (Staging Q in LDS per head measured slower: 5.1 vs 3.9 ms at 128k.)
        const int r = min(qr[i] + lane % 16, rows - 1);
        const bf16x16 fa = load_frag_bf16(q + int64_t(r) * sq_r + h * sq_h + kk, 0, 0);
#pragma unroll
        for (int j = 0; j < FN; ++j) acc[i][j] = wmma_bf16(fa, fb[j], acc[i][j]);
      }
    }
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
      for (int j = 0; j < FN; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) score[i][j][e] += fmaxf(acc[i][j][e], 0.f);
  }

#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = r0 + wm * TM + i * 16 + acc_row(e, lane);
      if (r >= rows) continue;
      const int vis = visible_of(r);
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        const int c = c0 + wn * TN + j * 16 + acc_col(lane);
        if (c < columns)
          logits[int64_t(r) * ld_logits + c] = c < vis ? score[i][j][e] * inv_div : -INFINITY;
      }
    }
}

}  // namespace

// logits: [rows, columns] fp32 (row stride any); visible: [rows] int32;
// q: [rows, H, 128] bf16; k_cache: [num_blocks, block_size, 1, 128] bf16;
// block_table: [max_blocks] int32 of the one request; positions: [rows]
// int64 request-relative query positions; seq_lens: [1] int32 (device).
void qsa_indexer_bf16(torch::Tensor& logits, torch::Tensor& visible,
                      const torch::Tensor& q, const torch::Tensor& k_cache,
                      const torch::Tensor& block_table,
                      const torch::Tensor& positions, const torch::Tensor& seq_lens,
                      int64_t compress_ratio, double score_divisor) {
  TORCH_CHECK(logits.scalar_type() == at::kFloat && visible.scalar_type() == at::kInt &&
                  q.scalar_type() == at::kBFloat16 && k_cache.scalar_type() == at::kBFloat16 &&
                  block_table.scalar_type() == at::kInt &&
                  positions.scalar_type() == at::kLong &&
                  seq_lens.scalar_type() == at::kInt,
              "qsa_indexer_bf16: dtypes");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kD && q.stride(2) == 1, "q [rows, H, 128]");
  TORCH_CHECK(q.size(1) == 4, "4 indexer heads, got ", q.size(1));
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(2) == 1 && k_cache.size(3) == kD &&
                  k_cache.stride(3) == 1,
              "k_cache [blocks, block_size, 1, 128]");
  TORCH_CHECK(logits.dim() == 2 && logits.stride(1) == 1 && logits.size(0) == q.size(0),
              "logits [rows, columns]");
  TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0, "16-byte aligned q rows");
  const int rows = int(q.size(0)), columns = int(logits.size(1));
  if (rows == 0 || columns == 0) return;
  // 64 x 64, 4 waves: larger tiles (64 x 128, 128 x 64, 32 x 128) were slower.
  constexpr int BM = 64, BN = 64, WM = 2, WN = 2;
  qsa_indexer_kernel<BM, BN, WM, WN, 4>
      <<<dim3((rows + BM - 1) / BM, (columns + BN - 1) / BN), WM * WN * kWave, 0,
         at::hip::getCurrentHIPStream()>>>(
          logits.data_ptr<float>(), visible.data_ptr<int32_t>(),
          reinterpret_cast<const uint16_t*>(q.data_ptr()),
          reinterpret_cast<const uint16_t*>(k_cache.data_ptr()),
          block_table.data_ptr<int32_t>(), positions.data_ptr<int64_t>(),
          seq_lens.data_ptr<int32_t>(), rows, columns, int(compress_ratio),
          int(k_cache.size(1)), q.stride(0), q.stride(1), k_cache.stride(0),
          k_cache.stride(1), logits.stride(0), float(1.0 / score_divisor));
}

}  // namespace strix
