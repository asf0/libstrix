// SPDX-License-Identifier: MIT
// The KV cache arguments of the Inkling 2 attention ops.
#pragma once

#include <torch/all.h>

#include <optional>

namespace strix {

struct InklingKV {
  bool q8 = false;
  // Q8 only: the fp32 scales and their (block, slot, head) strides.
  const float* k_scale = nullptr;
  const float* v_scale = nullptr;
  int64_t sc_b = 0, sc_s = 0, sc_h = 0;
};

// Checks the [blocks, block_size, Hkv, 128] caches, bf16 or int8; int8 ones
// come with fp32 [blocks, block_size, Hkv] scales. The kernels read bf16 rows
// 16 bytes at a time, int8 rows (V's start 132 bytes after K's) 4.
inline InklingKV inkling_kv(const torch::Tensor& key_cache, const torch::Tensor& value_cache,
                            const std::optional<torch::Tensor>& k_scale,
                            const std::optional<torch::Tensor>& v_scale) {
  InklingKV kv;
  kv.q8 = key_cache.scalar_type() != at::kBFloat16;
  const int64_t align = kv.q8 ? 4 : 16, elem = kv.q8 ? 1 : 2;
  for (const auto* t : {&key_cache, &value_cache}) {
    TORCH_CHECK(t->scalar_type() == (kv.q8 ? at::kChar : at::kBFloat16) && t->dim() == 4 &&
                    t->size(3) >= 128 && t->stride(3) == 1 &&
                    t->size(1) * t->stride(1) < (int64_t(1) << 31),
                "KV caches must be bf16 or int8 [blocks, block_size, Hkv, 128]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % align == 0 &&
                    t->stride(0) * elem % align == 0 && t->stride(1) * elem % align == 0 &&
                    t->stride(2) * elem % align == 0,
                "KV cache rows must be 16-byte (int8: 4-byte) aligned");
  }
  if (!kv.q8) return kv;
  TORCH_CHECK(k_scale.has_value() && v_scale.has_value(), "int8 KV needs k_scale and v_scale");
  for (const auto* t : {&*k_scale, &*v_scale})
    TORCH_CHECK(t->scalar_type() == at::kFloat && t->sizes() == key_cache.sizes().slice(0, 3) &&
                    t->strides() == k_scale->strides() &&
                    t->size(1) * t->stride(1) < (int64_t(1) << 31),
                "KV scales must be fp32 [blocks, block_size, Hkv] with equal strides");
  kv.k_scale = k_scale->data_ptr<float>();
  kv.v_scale = v_scale->data_ptr<float>();
  kv.sc_b = k_scale->stride(0), kv.sc_s = k_scale->stride(1), kv.sc_h = k_scale->stride(2);
  return kv;
}

}  // namespace strix
