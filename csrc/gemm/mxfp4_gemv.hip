// SPDX-License-Identifier: MIT
// MXFP4 W4A16 GEMV for M <= 8: one warp per output row, eight rows per block,
// A staged in LDS per K chunk and shared by the eight warps. Weights are
// decoded in registers straight from the packed stream (no LDS staging): each
// lane owns 32-K blocks (16 weight bytes + one e8m0 scale byte), accumulates
// all M dot products, and the warp reduces over lanes.
//
// This exists because the WMMA decode path stages 16 A rows for M=1 and
// decodes weights through LDS; its 127.5 ms/step on gemma-4-31B-MXFP4 is
// ~114 GB/s effective, and libr3d's equivalent register-decoding GEMV measured
// 143 GB/s absolute. The bench decides whether that transfers.
//
// Layout (Quark): w uint8 [N, K/2] (low nibble = even k), scales uint8
// [N, K/32] e8m0. Decode is the same one-add-per-value trick as
// mxfp4_wmma_gemm.hip: base = 0x3F00 + (s - 127) << 7, keep = 0 for an s == 0
// zero block.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kBlock = 32;    // elements per e8m0 scale byte
constexpr int kRows = 8;      // output rows per block (one warp each)
constexpr int kThreads = kRows * kWave;
constexpr int kChunk = 1024;  // K staged per A step; K need not divide it

__device__ __forceinline__ uint32_t nibble(uint32_t v, uint32_t base,
                                           uint32_t keep) {
  const uint32_t t = v & 7u;
  const uint32_t r = base + ((t >= 2u) ? (t << 6) : 0u);
  return t ? (((r | ((v & 8u) << 12)) & keep)) : 0u;
}

template <int M>
__global__ __launch_bounds__(kThreads) void mxfp4_gemv_kernel(
    uint16_t* __restrict__ c, const uint16_t* __restrict__ a,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ scales,
    int N, int K, int64_t lda, int64_t ldw, int64_t lds_n, int64_t ldc) {
  __shared__ __attribute__((aligned(16))) uint16_t s_a[M][kChunk];
  const int lane = threadIdx.x % kWave, warp = threadIdx.x / kWave;
  const int n = blockIdx.x * kRows + warp;
  const bool live = n < N;
  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  const uint8_t* wrow = w + int64_t(live ? n : 0) * ldw;
  const uint8_t* srow = scales + int64_t(live ? n : 0) * lds_n;

  for (int k0 = 0; k0 < K; k0 += kChunk) {
    const int kc = min(kChunk, K - k0);
    // Stage A rows [M][k0, k0+kc) cooperatively.
    for (int i = threadIdx.x; i < M * (kc / 8); i += kThreads) {
      const int m = i / (kc / 8), kk = (i % (kc / 8)) * 8;
      *reinterpret_cast<uint4*>(&s_a[m][kk]) = *reinterpret_cast<const uint4*>(
          a + int64_t(m) * lda + k0 + kk);
    }
    barrier();
    for (int kb = lane; live && kb * kBlock < kc; kb += kWave) {
      const int kk = kb * kBlock;                 // multiple of 32 within the chunk
      const uint32_t s = srow[(k0 + kk) / kBlock];
      const uint32_t base = 0x3F00u + ((s - 127u) << 7);
      const uint32_t keep = s ? 0xFFFFFFFFu : 0u;
      const uint4 q =
          *reinterpret_cast<const uint4*>(wrow + ((k0 + kk) >> 1));
      const uint32_t words[4] = {q.x, q.y, q.z, q.w};
      uint32_t bf[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        bf[4 * i] = nibble(words[i] & 15u, base, keep) |
                    (nibble((words[i] >> 4) & 15u, base, keep) << 16);
        bf[4 * i + 1] = nibble((words[i] >> 8) & 15u, base, keep) |
                        (nibble((words[i] >> 12) & 15u, base, keep) << 16);
        bf[4 * i + 2] = nibble((words[i] >> 16) & 15u, base, keep) |
                        (nibble((words[i] >> 20) & 15u, base, keep) << 16);
        bf[4 * i + 3] = nibble((words[i] >> 24) & 15u, base, keep) |
                        (nibble((words[i] >> 28) & 15u, base, keep) << 16);
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const float w0 = bf16_lo(bf[i]), w1 = bf16_hi(bf[i]);
          const uint32_t av =
              *reinterpret_cast<const uint32_t*>(&s_a[m][kk + 2 * i]);
          s += bf16_lo(av) * w0 + bf16_hi(av) * w1;
        }
        acc[m] += s;
      }
    }
    barrier();
  }

#pragma unroll
  for (int m = 0; m < M; ++m) {
    acc[m] = group_sum<kWave>(acc[m]);
    if (live && lane == 0)
      c[int64_t(m) * ldc + n] =
          static_cast<uint16_t>(pack_bf16x2(acc[m], 0.f) & 0xffffu);
  }
}

}  // namespace

// x: [M, K] bf16, M <= 8, K % 32 == 0; w: uint8 [N, K/2] (rows 16-byte
// aligned); scales: uint8 [N, K/32]. Returns [M, N] bf16.
torch::Tensor mxfp4_gemv(const torch::Tensor& x, const torch::Tensor& w,
                         const torch::Tensor& scales) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
                  x.stride(1) == 1 && x.stride(0) % 8 == 0,
              "x must be bf16 [M, K] with 16-byte aligned rows");
  TORCH_CHECK(w.scalar_type() == at::kByte && w.dim() == 2 &&
                  w.stride(1) == 1 && w.stride(0) % 16 == 0,
              "w must be uint8 [N, K/2] with 16-byte aligned rows");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(m >= 1 && m <= kRows, "mxfp4_gemv is for M <= ", kRows);
  TORCH_CHECK(w.size(1) * 2 == k && k % kBlock == 0, "K mismatch or K % 32");
  TORCH_CHECK(scales.scalar_type() == at::kByte && scales.dim() == 2 &&
                  scales.size(0) == n && scales.size(1) == k / kBlock &&
                  scales.stride(1) == 1,
              "scales must be uint8 [N, K/32]");
  auto c = torch::empty({m, n}, x.options());
  if (m == 0 || n == 0) return c;
  auto stream = at::hip::getCurrentHIPStream();
#define STRIX_MX_GEMV(M)                                                     \
  mxfp4_gemv_kernel<M><<<dim3((n + kRows - 1) / kRows), kThreads, 0,         \
                         stream>>>(                                          \
      reinterpret_cast<uint16_t*>(c.data_ptr()),                             \
      reinterpret_cast<const uint16_t*>(x.data_ptr()),                       \
      reinterpret_cast<const uint8_t*>(w.data_ptr()),                        \
      reinterpret_cast<const uint8_t*>(scales.data_ptr()), int(n), int(k),    \
      x.stride(0), w.stride(0), scales.stride(0), c.stride(0))
  if (m == 1) STRIX_MX_GEMV(1);
  else if (m == 2) STRIX_MX_GEMV(2);
  else if (m == 3) STRIX_MX_GEMV(3);
  else if (m == 4) STRIX_MX_GEMV(4);
  else if (m == 5) STRIX_MX_GEMV(5);
  else if (m == 6) STRIX_MX_GEMV(6);
  else if (m == 7) STRIX_MX_GEMV(7);
  else STRIX_MX_GEMV(8);
#undef STRIX_MX_GEMV
  return c;
}

}  // namespace strix
