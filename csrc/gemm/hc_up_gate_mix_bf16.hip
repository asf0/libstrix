// SPDX-License-Identifier: MIT
// Qwen3.8-Flash-Next hyper-connection mix: the input_mix_weight_up GEMM fused
// with hc_gate_mix,
//   y[m, c] = (1/4) sum_s sigmoid(bf16(a[m] . w[s * C + c])) * x[m, s * C + c]
// for hc_count = 4 streams of C columns. Unfused, the up GEMM writes the
// [M, 4C] gate (335 MB at 16k rows) and hc_gate_mix reads it straight back.
//
// A block owns BM rows x BN columns of y and computes the matching BN columns
// of all four streams: wave (wm, s) holds rows wm*TM.. of stream s, so A
// fragments are shared by the four stream waves through LDS. K (the low rank,
// 320) runs in 64-wide steps. The gate is rounded to bf16 before the sigmoid
// like the unfused GEMM's output, and the four streams are added into an fp32
// LDS tile in stream order 0..3, the order of hc_gate_mix's loop.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kHC = 4;
constexpr int kBK = 64;
constexpr int kPitch = kBK + 8;

template <int BM, int BN, int WAVES_M>
__global__ __launch_bounds__(WAVES_M* kHC* kWave) void hc_up_gate_mix_kernel(
    uint16_t* __restrict__ y, const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ w, const uint16_t* __restrict__ x, int64_t rows,
    int k_dim, int cols, int64_t lda, int64_t ldw, int64_t ldx, int64_t ldy) {
  constexpr int kThreads = WAVES_M * kHC * kWave;
  constexpr int TM = BM / WAVES_M, FM = TM / 16, FN = BN / 16;
  constexpr int kChunksA = BM * kBK / 8, kChunksB = kHC * BN * kBK / 8;
  constexpr int kLoadsA = (kChunksA + kThreads - 1) / kThreads;
  constexpr int kLoadsB = (kChunksB + kThreads - 1) / kThreads;
  constexpr int kTileBytes = (BM + kHC * BN) * kPitch * 2;
  static_assert(BM * BN * 4 <= kTileBytes, "the fp32 y tile reuses the A/B tiles");
  __shared__ __attribute__((aligned(16))) uint8_t s_buf[kTileBytes];
  uint16_t* s_a = reinterpret_cast<uint16_t*>(s_buf);
  uint16_t* s_b = s_a + BM * kPitch;  // [stream][BN] rows of w

  const int64_t m0 = int64_t(blockIdx.y) * BM;
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int wm = wave % WAVES_M, s = wave / WAVES_M;

  uint4 ra[kLoadsA], rb[kLoadsB];
  auto load = [&](int k0) {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads;
      uint4 v = {};
      if (ch < kChunksA) {
        const int64_t m = m0 + ch / (kBK / 8);
        if (m < rows)
          v = *reinterpret_cast<const uint4*>(a + m * lda + k0 + (ch % (kBK / 8)) * 8);
      }
      ra[i] = v;
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads;
      uint4 v = {};
      if (ch < kChunksB) {
        const int r = ch / (kBK / 8), st = r / BN, c = n0 + r % BN;
        if (c < cols)
          v = *reinterpret_cast<const uint4*>(w + int64_t(st * cols + c) * ldw + k0 +
                                              (ch % (kBK / 8)) * 8);
      }
      rb[i] = v;
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads;
      if (ch < kChunksA)
        *reinterpret_cast<uint4*>(s_a + (ch / (kBK / 8)) * kPitch + (ch % (kBK / 8)) * 8) =
            ra[i];
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads;
      if (ch < kChunksB)
        *reinterpret_cast<uint4*>(s_b + (ch / (kBK / 8)) * kPitch + (ch % (kBK / 8)) * 8) =
            rb[i];
    }
  };

  f32x8 acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) acc[i][j] = f32x8{};
  load(0);
  store();
  __syncthreads();
  for (int k0 = 0; k0 < k_dim; k0 += kBK) {
    const bool more = k0 + kBK < k_dim;
    if (more) load(k0 + kBK);
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      bf16x16 fb[FN];
#pragma unroll
      for (int j = 0; j < FN; ++j)
        fb[j] = load_frag_bf16(s_b + (s * BN + j * 16) * kPitch + kk, kPitch, lane);
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        const bf16x16 fa = load_frag_bf16(s_a + (wm * TM + i * 16) * kPitch + kk, kPitch, lane);
#pragma unroll
        for (int j = 0; j < FN; ++j) acc[i][j] = wmma_bf16(fa, fb[j], acc[i][j]);
      }
    }
    __syncthreads();
    if (more) store();
    __syncthreads();
  }

  // acc[i][j][e] = gate[row wm*TM + i*16 + acc_row(e)][stream s, col j*16 + acc_col].
  // Every wave turns its gates into sigmoid(gate) * x at once (the x loads of
  // all four streams in flight together); only the adds are ordered.
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int64_t m = m0 + wm * TM + i * 16 + acc_row(e, lane);
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        const int c = n0 + j * 16 + acc_col(lane);
        float v = 0.f;
        if (m < rows && c < cols) {
          const float g = __uint_as_float(round_bf16_hi(acc[i][j][e]) & 0xffff0000u);
          const float xv = __uint_as_float(uint32_t(x[m * ldx + int64_t(s) * cols + c]) << 16);
          v = xv / (1.f + __expf(-g));
        }
        acc[i][j][e] = v;
      }
    }
  float* s_y = reinterpret_cast<float*>(s_buf);  // [BM][BN]
  for (int st = 0; st < kHC; ++st) {
    if (s == st) {
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int r = wm * TM + i * 16 + acc_row(e, lane);
#pragma unroll
          for (int j = 0; j < FN; ++j) {
            const int c = j * 16 + acc_col(lane);
            if (st == 0)
              s_y[r * BN + c] = acc[i][j][e];
            else
              s_y[r * BN + c] += acc[i][j][e];
          }
        }
    }
    __syncthreads();
  }
  for (int i = tid; i < BM * BN; i += kThreads) {
    const int r = i / BN, c = i % BN;
    const int64_t m = m0 + r;
    if (m < rows && n0 + c < cols)
      y[m * ldy + n0 + c] =
          static_cast<uint16_t>(pack_bf16x2(s_y[i] * (1.f / kHC), 0.f) & 0xffffu);
  }
}

}  // namespace

// x: [M, 4C] bf16 (the normed multi-stream state), a: [M, K] bf16 (the
// silu'd low-rank projection), w: [4C, K] bf16 (input_mix_weight_up);
// y: [M, C]. Rows of a, w, x and y need 16-byte alignment, K % 64 == 0.
void hc_up_gate_mix_bf16(torch::Tensor& y, const torch::Tensor& x,
                         const torch::Tensor& a, const torch::Tensor& w) {
  TORCH_CHECK(y.scalar_type() == at::kBFloat16 && x.scalar_type() == at::kBFloat16 &&
                  a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
              "hc_up_gate_mix_bf16: bf16 tensors");
  TORCH_CHECK(y.dim() == 2 && x.dim() == 2 && a.dim() == 2 && w.dim() == 2, "2-D tensors");
  TORCH_CHECK(y.stride(1) == 1 && x.stride(1) == 1 && a.stride(1) == 1 && w.stride(1) == 1,
              "contiguous last dims");
  const int64_t rows = x.size(0), k = a.size(1), cols = y.size(1);
  TORCH_CHECK(x.size(1) == kHC * cols && w.size(0) == kHC * cols && w.size(1) == k &&
                  a.size(0) == rows && y.size(0) == rows,
              "shapes: x [M, 4C], a [M, K], w [4C, K], y [M, C]");
  TORCH_CHECK(k % kBK == 0, "K must be a multiple of ", kBK);
  TORCH_CHECK(a.stride(0) % 8 == 0 && w.stride(0) % 8 == 0, "16-byte aligned a / w rows");
  if (rows == 0) return;
  // 128 x 32 (8 waves): 5.2 ms at 16k rows against 6.6 for 128 x 64, 6.6
  // for 64 x 32, 5.3 for 256 x 32 and 128 x 16; the unfused pair is 6.7.
  constexpr int BM = 128, BN = 32, WM = 2;
  hc_up_gate_mix_kernel<BM, BN, WM>
      <<<dim3((cols + BN - 1) / BN, (rows + BM - 1) / BM), WM * kHC * kWave, 0,
         at::hip::getCurrentHIPStream()>>>(
          reinterpret_cast<uint16_t*>(y.data_ptr()),
          reinterpret_cast<const uint16_t*>(a.data_ptr()),
          reinterpret_cast<const uint16_t*>(w.data_ptr()),
          reinterpret_cast<const uint16_t*>(x.data_ptr()), rows, int(k), int(cols),
          a.stride(0), w.stride(0), x.stride(0), y.stride(0));
}

}  // namespace strix
