// SPDX-License-Identifier: MIT
// Qwen3.8-Flash-Next QSA: rewrite one layer's visible K/V from the paged bf16
// cache into compact fp16 arrays indexed by token,
//   k16[token][kvh * 256 + d], v16[token][kvh * 256 + d]
// so the sparse attention gather reads dense, contiguous rows with no
// page-table arithmetic, at half the bytes (bf16 -> fp16 is exact for the
// values in a K/V cache: fp16 has three more mantissa bits and the exponent
// range only matters past 65504).
//
// The cost is one pass over the visible cache per layer: at 16k tokens and
// two KV heads that is 66 MB read and 33 MB written, ~0.5 ms.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

__global__ __launch_bounds__(256) void qsa_kv_to_f16_kernel(
    uint16_t* __restrict__ k16, uint16_t* __restrict__ v16,
    const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    const int32_t* __restrict__ bt, int tokens, int width, int block_size,
    int64_t sk_b, int64_t sk_s, int64_t sv_b, int64_t sv_s) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = int64_t(tokens) * width;
  if (i >= total) return;
  const int token = int(i / width), c = int(i % width);
  const int64_t page = bt[token / block_size];
  const int slot = token % block_size;
  const int64_t ksrc = page * sk_b + int64_t(slot) * sk_s + c;
  const int64_t vsrc = page * sv_b + int64_t(slot) * sv_s + c;
  // bf16 -> fp16: widen the bf16 into fp32 and narrow.
  const float kf = __uint_as_float(uint32_t(kc[ksrc]) << 16);
  const float vf = __uint_as_float(uint32_t(vc[vsrc]) << 16);
  k16[i] = static_cast<uint16_t>(
      __builtin_bit_cast(uint16_t, static_cast<_Float16>(kf)));
  v16[i] = static_cast<uint16_t>(
      __builtin_bit_cast(uint16_t, static_cast<_Float16>(vf)));
}

}  // namespace

// k16, v16: [tokens, kv_heads * 256] fp16 (bf16 bit-holding uint16 view);
// key_cache / value_cache: [blocks, block_size, kv_heads, 256] bf16 views;
// block_table: [max_blocks] int32 of the request; tokens = its length.
void qsa_kv_to_f16(torch::Tensor& k16, torch::Tensor& v16,
                   const torch::Tensor& key_cache, const torch::Tensor& value_cache,
                   const torch::Tensor& block_table, int64_t tokens) {
  TORCH_CHECK(k16.scalar_type() == at::kHalf && v16.scalar_type() == at::kHalf,
              "k16 / v16 must be fp16");
  const int width = int(key_cache.size(2) * key_cache.size(3));
  TORCH_CHECK(key_cache.size(3) == 256 && key_cache.stride(3) == 1 &&
                  value_cache.stride(3) == 1,
              "caches must be [blocks, block_size, kv_heads, 256]");
  TORCH_CHECK(k16.is_contiguous() && v16.is_contiguous() &&
                  k16.numel() >= tokens * width && v16.numel() >= tokens * width,
              "k16 / v16 must hold tokens * kv_heads * 256 elements");
  if (tokens == 0) return;
  const int64_t total = tokens * width;
  const int threads = 256;
  qsa_kv_to_f16_kernel<<<dim3((total + threads - 1) / threads), threads, 0,
                         at::hip::getCurrentHIPStream()>>>(
      reinterpret_cast<uint16_t*>(k16.data_ptr()),
      reinterpret_cast<uint16_t*>(v16.data_ptr()),
      reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),
      reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),
      block_table.data_ptr<int32_t>(), int(tokens), width,
      int(key_cache.size(1)), key_cache.stride(0), key_cache.stride(1),
      value_cache.stride(0), value_cache.stride(1));
}

}  // namespace strix
