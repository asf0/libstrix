// SPDX-License-Identifier: MIT
// Split-KV reduce shared by attn_decode_bf16 and attn_prefill_bf16.
#pragma once

#include "../strix_common.h"

namespace strix {
// Internal linkage: every including .hip gets its own kernel and host stub.
namespace {

// Combines split-KV partials: one block per (token, head), one thread per dim.
// o_part [T][Hq][nseg][D] holds each segment's unnormalized output, ml_part
// [T][Hq][nseg][2] its (max, sum) in log2 units. sinks (optional, [Hq]): one
// more unscaled natural-log logit per head, folded in once, here, after the
// segments are combined.
template <int D>
__global__ __launch_bounds__(D) void attn_split_reduce_kernel(
    uint16_t* __restrict__ out, const float* __restrict__ o_part,
    const float* __restrict__ ml_part, int nseg, int hq, int64_t so_t,
    int64_t so_h, const float* __restrict__ sinks) {
  const int64_t row = int64_t(blockIdx.x) * hq + blockIdx.y;
  const float* ml = ml_part + row * nseg * 2;
  // kBatch segments' loads are in flight together: one at a time, every segment
  // would wait out a memory latency.
  constexpr int kBatch = 8;
  float m = -INFINITY;
  for (int s0 = 0; s0 < nseg; s0 += kBatch) {
    float mx[kBatch];
#pragma unroll
    for (int j = 0; j < kBatch; ++j) mx[j] = ml[2 * min(s0 + j, nseg - 1)];
#pragma unroll
    for (int j = 0; j < kBatch; ++j) m = fmaxf(m, mx[j]);
  }
  float l = 0.f, acc = 0.f;
  for (int s0 = 0; s0 < nseg; s0 += kBatch) {
    float mx[kBatch], sum[kBatch], o[kBatch];
#pragma unroll
    for (int j = 0; j < kBatch; ++j) {
      const int s = min(s0 + j, nseg - 1);
      mx[j] = ml[2 * s];
      sum[j] = ml[2 * s + 1];
      o[j] = o_part[(row * nseg + s) * D + threadIdx.x];
    }
#pragma unroll
    for (int j = 0; j < kBatch; ++j) {
      if (s0 + j < nseg) {
        const float w = mx[j] == -INFINITY ? 0.f : exp2f(mx[j] - m);
        l += w * sum[j];
        acc += w * o[j];
      }
    }
  }
  if (sinks != nullptr) {
    const float s2 = sinks[blockIdx.y] * 1.4426950408889634f;
    const float m_fin = fmaxf(m, s2);
    const float a = m == -INFINITY ? 0.f : exp2f(m - m_fin);
    acc *= a;
    l = l * a + exp2f(s2 - m_fin);
  }
  out[blockIdx.x * so_t + blockIdx.y * so_h + threadIdx.x] =
      static_cast<uint16_t>(pack_bf16x2(acc / l, 0.f) & 0xffffu);
}

inline void launch_attn_split_reduce(int64_t d, dim3 grid, hipStream_t stream,
                                     uint16_t* out, const float* o_part,
                                     const float* ml_part, int nseg, int hq,
                                     int64_t so_t, int64_t so_h,
                                     const float* sinks) {
  auto reduce = d == 64    ? attn_split_reduce_kernel<64>
                : d == 128 ? attn_split_reduce_kernel<128>
                : d == 256 ? attn_split_reduce_kernel<256>
                           : attn_split_reduce_kernel<512>;
  reduce<<<grid, static_cast<uint32_t>(d), 0, stream>>>(
      out, o_part, ml_part, nseg, hq, so_t, so_h, sinks);
}

}  // namespace
}  // namespace strix
