// SPDX-License-Identifier: MIT
// float8_e4m3fn [N, K] + 128x128 block scales (fp32 [ceil(N/128), K/128]) ->
// bf16 [N, K], or with per-output-channel scales (fp32 [N]). For prefill M the plugin decodes a block-scaled FP8 weight once
// and runs hipBLASLt on the bf16 matrix (gfx1151 has no fp8 WMMA).
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace strix {
namespace {

constexpr int kBlock = 128;
constexpr int kChunk = 16;  // bytes per thread

// Same bit trick as the skinny GEMM (w8a16_skinny_gemm.hip): the e4m3 byte as
// an fp16 holds value / 2^8.
__device__ __forceinline__ uint32_t fp8_pair_bits(uint32_t q) {
  return ((q & 0x7Fu) << 7) | ((q & 0x80u) << 8) | ((q & 0x7F00u) << 15) |
         ((q & 0x8000u) << 16);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const uint32_t a = __bfloat16_as_ushort(__float2bfloat16(lo));
  const uint32_t b = __bfloat16_as_ushort(__float2bfloat16(hi));
  return a | (b << 16);
}

__global__ __launch_bounds__(256) void fp8_dequant_kernel(
    uint16_t* __restrict__ out, const uint8_t* __restrict__ w,
    const float* __restrict__ scales, int64_t N, int64_t K, int64_t ldw,
    int64_t lds_n, bool channel) {
  const int64_t ch = int64_t(blockIdx.x) * 256 + threadIdx.x;
  const int64_t per_row = K / kChunk;
  if (ch >= N * per_row) return;
  const int64_t r = ch / per_row, c = ch % per_row;
  const uint4 v = *reinterpret_cast<const uint4*>(w + r * ldw + c * kChunk);
  const float s =
      (channel ? scales[r] : scales[(r / kBlock) * lds_n + (c * kChunk) / kBlock]) *
      256.0f;
  const uint32_t word[4] = {v.x, v.y, v.z, v.w};
  uint32_t o[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t lo = fp8_pair_bits(word[i]);
    const uint32_t hi = fp8_pair_bits(word[i] >> 16);
    const half2 a = *reinterpret_cast<const half2*>(&lo);
    const half2 b = *reinterpret_cast<const half2*>(&hi);
    o[2 * i] = pack_bf16(__half2float(a.x) * s, __half2float(a.y) * s);
    o[2 * i + 1] = pack_bf16(__half2float(b.x) * s, __half2float(b.y) * s);
  }
  uint4* dst = reinterpret_cast<uint4*>(out + r * K + c * kChunk);
  dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
  dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

}  // namespace

// w: float8_e4m3fn [N, K] (rows may be padded, 16-byte aligned); scales: fp32
// [ceil(N / 128), K / 128], or [N] per channel. Returns the bf16 [N, K] matrix.
torch::Tensor fp8_dequant(const torch::Tensor& w, const torch::Tensor& scales) {
  TORCH_CHECK(w.scalar_type() == at::kFloat8_e4m3fn && w.dim() == 2 &&
                  w.stride(1) == 1 && w.stride(0) % 16 == 0,
              "w must be float8_e4m3fn [N, K] with 16-byte aligned rows");
  const int64_t n = w.size(0), k = w.size(1);
  const bool channel = scales.dim() == 1;
  TORCH_CHECK(scales.scalar_type() == at::kFloat, "scales must be float32");
  if (channel) {
    TORCH_CHECK(k % kChunk == 0, "K must be a multiple of ", kChunk);
    TORCH_CHECK(scales.size(0) == n && scales.stride(0) == 1,
                "per-channel scale must be contiguous float32 [N]");
  } else {
    TORCH_CHECK(k % kBlock == 0, "K must be a multiple of ", kBlock);
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == (n + kBlock - 1) / kBlock &&
                    scales.size(1) == k / kBlock && scales.stride(1) == 1,
                "scale must be float32 [ceil(N/128), K/128] or [N]");
  }
  auto out = torch::empty({n, k}, w.options().dtype(at::kBFloat16));
  const int64_t chunks = n * (k / kChunk);
  if (chunks == 0) return out;
  fp8_dequant_kernel<<<dim3(unsigned((chunks + 255) / 256)), 256, 0,
                       at::hip::getCurrentHIPStream()>>>(
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      scales.data_ptr<float>(), n, k, w.stride(0),
      channel ? 0 : scales.stride(0), channel);
  return out;
}

}  // namespace strix
