// SPDX-License-Identifier: MIT
// Qwen3.8-Flash-Next QSA: rewrite one layer's visible K/V from the paged cache
// into compact fp16 arrays indexed by token,
//   k16[token][kvh * 256 + d], v16[token][kvh * 256 + d]
// so the sparse attention gather reads dense, contiguous rows with no
// page-table arithmetic.
//
// bf16 cache: bf16 -> fp16 is exact for the values in a K/V cache (fp16 has
// three more mantissa bits and the exponent range only matters past 65504),
// at half the bytes of the gather.
// int8_per_token_head cache: q * scale per (token, head), rounded once to
// fp16 (11 bits, finer than the int8 step).
//
// The cost is one pass over the visible cache per layer: at 16k tokens and
// two KV heads that is 66 MB read (33 MB for int8) and 33 MB written, ~0.5 ms.
// The caches are views of vLLM's packed layout, where each head's K is
// followed by its V (and, for int8, their scales), so the head stride is
// explicit.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kD = 256;

__global__ __launch_bounds__(256) void qsa_kv_to_f16_kernel(
    uint16_t* __restrict__ k16, uint16_t* __restrict__ v16,
    const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc,
    const int32_t* __restrict__ bt, int tokens, int width, int block_size,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s,
    int64_t sv_h) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = int64_t(tokens) * width;
  if (i >= total) return;
  const int token = int(i / width), c = int(i % width);
  const int h = c / kD, d = c % kD;
  const int64_t page = bt[token / block_size];
  const int slot = token % block_size;
  const int64_t ksrc = page * sk_b + int64_t(slot) * sk_s + h * sk_h + d;
  const int64_t vsrc = page * sv_b + int64_t(slot) * sv_s + h * sv_h + d;
  // bf16 -> fp16: widen the bf16 into fp32 and narrow.
  const float kf = __uint_as_float(uint32_t(kc[ksrc]) << 16);
  const float vf = __uint_as_float(uint32_t(vc[vsrc]) << 16);
  k16[i] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(kf));
  v16[i] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(vf));
}

// Scales: fp32 views [blocks, block_size, kv_heads] with their own strides
// (in floats).
__global__ __launch_bounds__(256) void qsa_kv_i8_to_f16_kernel(
    uint16_t* __restrict__ k16, uint16_t* __restrict__ v16,
    const int8_t* __restrict__ kc, const int8_t* __restrict__ vc,
    const float* __restrict__ ks, const float* __restrict__ vs,
    const int32_t* __restrict__ bt, int tokens, int width, int block_size,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s,
    int64_t sv_h, int64_t sks_b, int64_t sks_s, int64_t sks_h, int64_t svs_b,
    int64_t svs_s, int64_t svs_h) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = int64_t(tokens) * width;
  if (i >= total) return;
  const int token = int(i / width), c = int(i % width);
  const int h = c / kD, d = c % kD;
  const int64_t page = bt[token / block_size];
  const int slot = token % block_size;
  const float kscale = ks[page * sks_b + int64_t(slot) * sks_s + h * sks_h];
  const float vscale = vs[page * svs_b + int64_t(slot) * svs_s + h * svs_h];
  const float kf = float(kc[page * sk_b + int64_t(slot) * sk_s + h * sk_h + d]) * kscale;
  const float vf = float(vc[page * sv_b + int64_t(slot) * sv_s + h * sv_h + d]) * vscale;
  k16[i] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(kf));
  v16[i] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(vf));
}

}  // namespace

// k16, v16: [tokens, kv_heads * 256] fp16;
// key_cache / value_cache: [blocks, block_size, kv_heads, 256] bf16 or int8
// views (any head stride); k_scale / v_scale: fp32 [blocks, block_size,
// kv_heads] for an int8 cache, None otherwise;
// block_table: [max_blocks] int32 of the request; tokens = its length.
void qsa_kv_to_f16(torch::Tensor& k16, torch::Tensor& v16,
                   const torch::Tensor& key_cache, const torch::Tensor& value_cache,
                   const torch::Tensor& block_table, int64_t tokens,
                   const std::optional<torch::Tensor>& k_scale_opt,
                   const std::optional<torch::Tensor>& v_scale_opt) {
  TORCH_CHECK(k16.scalar_type() == at::kHalf && v16.scalar_type() == at::kHalf,
              "k16 / v16 must be fp16");
  const int width = int(key_cache.size(2) * key_cache.size(3));
  TORCH_CHECK(key_cache.size(3) == kD && key_cache.stride(3) == 1 &&
                  value_cache.stride(3) == 1,
              "caches must be [blocks, block_size, kv_heads, 256]");
  TORCH_CHECK(k16.is_contiguous() && v16.is_contiguous() &&
                  k16.numel() >= tokens * width && v16.numel() >= tokens * width,
              "k16 / v16 must hold tokens * kv_heads * 256 elements");
  const bool int8 = key_cache.scalar_type() == at::kChar;
  TORCH_CHECK(value_cache.scalar_type() == key_cache.scalar_type() &&
                  (int8 || key_cache.scalar_type() == at::kBFloat16),
              "caches must both be bf16 or both int8");
  if (tokens == 0) return;
  const int64_t total = tokens * width;
  const int threads = 256;
  const dim3 grid((total + threads - 1) / threads);
  auto stream = at::hip::getCurrentHIPStream();
  if (!int8) {
    qsa_kv_to_f16_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<uint16_t*>(k16.data_ptr()),
        reinterpret_cast<uint16_t*>(v16.data_ptr()),
        reinterpret_cast<const uint16_t*>(key_cache.data_ptr()),
        reinterpret_cast<const uint16_t*>(value_cache.data_ptr()),
        block_table.data_ptr<int32_t>(), int(tokens), width, int(key_cache.size(1)),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2));
    return;
  }
  TORCH_CHECK(k_scale_opt.has_value() && v_scale_opt.has_value(),
              "int8 caches need k_scale / v_scale");
  const torch::Tensor& k_scale = *k_scale_opt;
  const torch::Tensor& v_scale = *v_scale_opt;
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat && v_scale.scalar_type() == at::kFloat &&
                  k_scale.dim() == 3 && v_scale.dim() == 3 &&
                  k_scale.size(2) == key_cache.size(2),
              "int8 caches need fp32 scales [blocks, block_size, kv_heads]");
  qsa_kv_i8_to_f16_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<uint16_t*>(k16.data_ptr()),
      reinterpret_cast<uint16_t*>(v16.data_ptr()),
      key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(),
      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
      block_table.data_ptr<int32_t>(), int(tokens), width, int(key_cache.size(1)),
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
      value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
      k_scale.stride(0), k_scale.stride(1), k_scale.stride(2), v_scale.stride(0),
      v_scale.stride(1), v_scale.stride(2));
}

}  // namespace strix
