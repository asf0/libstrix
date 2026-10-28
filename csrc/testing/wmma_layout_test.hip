// SPDX-License-Identifier: MIT
// Pins the WMMA fragment layout in strix_common.h: one wave computes
// D[16, 16] = A[16, K] @ B[16, K]^T with the shared helpers.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

__global__ __launch_bounds__(kWave) void wmma_tile(float* __restrict__ d,
                                                   const uint16_t* __restrict__ a,
                                                   const uint16_t* __restrict__ b,
                                                   int k) {
  const int lane = threadIdx.x;
  f32x8 acc = {};
  for (int k0 = 0; k0 < k; k0 += 16)
    acc = wmma_bf16(load_frag_bf16(a + k0, k, lane),
                    load_frag_bf16(b + k0, k, lane), acc);
#pragma unroll
  for (int e = 0; e < 8; ++e) d[acc_row(e, lane) * 16 + acc_col(lane)] = acc[e];
}

}  // namespace

torch::Tensor _wmma_bf16_tile(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16);
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && a.sizes() == b.sizes());
  TORCH_CHECK(a.dim() == 2 && a.size(0) == 16 && a.size(1) % 16 == 0);
  auto d = torch::empty({16, 16}, a.options().dtype(at::kFloat));
  wmma_tile<<<1, kWave, 0, at::hip::getCurrentHIPStream()>>>(
      d.data_ptr<float>(), reinterpret_cast<const uint16_t*>(a.data_ptr()),
      reinterpret_cast<const uint16_t*>(b.data_ptr()), a.size(1));
  return d;
}

}  // namespace strix
