// SPDX-License-Identifier: MIT
// NVFP4 expert GEMV over libstrix's routing tiles (Inkling 2 Mini decode,
// verify and small batches): a workgroup takes one expert tile (up to 16
// routed rows) and 128 output columns, and streams the expert's weight rows
// once for all of the tile's rows.
//
// Weights are pre-tiled (`pretile_nvfp4`): for 128 weight rows and 64 columns
// the rows' 32 code bytes (byte j: column 2j low, 2j + 1 high) and 4 e4m3
// scale bytes are contiguous, so 128 threads (one per weight row) read a K
// step as one 4 KB block. Two thread halves take alternating K steps.
//
// Codes become bf16 pairs with two v_perm_b32 table lookups per byte (low and
// high bytes of the bf16 e2m1 values) plus the sign bits, and meet the bf16
// activations (staged in LDS) in v_dot2_f32_bf16; each 16-column group's
// partial sum takes its e4m3 scale in fp32, and the expert's amax /
// denominator scales the result.

#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <cstdint>

namespace strix {
namespace {

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

constexpr int kMaxRows = 16;   // routing tile rows (decode-sized tiles)
constexpr int kChunk = 1152;   // activation columns staged per pass (18 K steps)
constexpr int kThreads = 256;  // 2 halves x 128 weight rows

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
  bf16x2 x, y;
  __builtin_memcpy(&x, &a, 4);
  __builtin_memcpy(&y, &b, 4);
  return __builtin_amdgcn_fdot2_f32_bf16(x, y, c, false);
}

// One code byte (column 2j low nibble, 2j + 1 high) -> bf16 pair.
__device__ __forceinline__ uint32_t e2m1x2_bf16(uint32_t p) {
  const uint32_t m0 = p & 7u, m1 = (p >> 4) & 7u;
  // bf16 of {0, .5, 1, 1.5, 2, 3, 4, 6}: low bytes {00 00 80 C0 00 40 80 C0},
  // high bytes {00 3F 3F 3F 40 40 40 40}; selector 0x0C is the zero byte.
  uint32_t lo, hi;
  asm("v_perm_b32 %0, %1, %2, %3"
      : "=v"(lo)
      : "v"(0xC0804000u), "v"(0xC0800000u), "v"(m0 | (m1 << 16) | 0x0C000C00u));
  asm("v_perm_b32 %0, %1, %2, %3"
      : "=v"(hi)
      : "v"(0x40404040u), "v"(0x3F3F3F00u), "v"((m0 << 8) | (m1 << 24) | 0x000C000Cu));
  return lo | hi | ((p & 8u) << 12) | ((p & 0x80u) << 24);
}

__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
  const uint32_t e = (b >> 3) & 15u, m = b & 7u;
  const float normal = __uint_as_float(((e + 120u) << 23) | (m << 20));
  const float v = e ? normal : float(m) * 0.001953125f;  // m * 2^-9
  return (b & 0x80u) ? -v : v;
}

__global__ __launch_bounds__(kThreads) void nvfp4_moe_gemv_kernel(
    const uint16_t* __restrict__ act,     // bf16 [rows, K]
    const uint4* __restrict__ packed,     // [E, N/128, K/64, 128, 2] uint4
    const uint32_t* __restrict__ scales,  // [E, N/128, K/64, 128] u32
    const float* __restrict__ amax, const float* __restrict__ denominator,
    const int32_t* __restrict__ tiles, uint16_t* __restrict__ out,  // bf16
    int max_ctas, int tile_rows, int n, int k) {
  const int tile = blockIdx.x;
  if (tile >= min(tiles[2 * max_ctas], max_ctas)) return;
  const int row0 = tile * tile_rows;
  const int num_rows = min(tiles[max_ctas + tile] - row0, kMaxRows);
  if (num_rows <= 0) return;
  const int expert = tiles[tile];
  const int n_blk = blockIdx.y;
  const int tid = threadIdx.x;
  const int wrow = tid & 127;
  const int half = tid >> 7;
  const int n_tiles = n / 128;
  const int k_tiles = k / 64;
  const int64_t tile0 = ((int64_t)expert * n_tiles + n_blk) * k_tiles;

  __shared__ uint4 a_lds[kMaxRows * kChunk / 8];  // bf16 rows, 8 per uint4
  __shared__ float red[kMaxRows][128];

  float acc[kMaxRows];
#pragma unroll
  for (int r = 0; r < kMaxRows; ++r) acc[r] = 0.f;

  for (int c0 = 0; c0 < k; c0 += kChunk) {
    const int cols = min(kChunk, k - c0);
    const int vec_per_row = cols / 8;
    __syncthreads();
    for (int i = tid; i < num_rows * vec_per_row; i += kThreads) {
      const int r = i / vec_per_row, v = i % vec_per_row;
      a_lds[r * (kChunk / 8) + v] =
          *reinterpret_cast<const uint4*>(act + (int64_t)(row0 + r) * k + c0 + v * 8);
    }
    __syncthreads();
    const int kt0 = c0 / 64;
    const int kt_end = kt0 + cols / 64;
    for (int kt = kt0 + half; kt < kt_end; kt += 2) {
      const int lds_col = (kt - kt0) * 8;  // uint4 index of the K step's 64 columns
      const int64_t base = (tile0 + kt) * 128 + wrow;
      const uint4 w0 = packed[base * 2];
      const uint4 w1 = packed[base * 2 + 1];
      const uint32_t sc = scales[base];
      const uint32_t words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        uint32_t wb[8];  // 16 columns as bf16 pairs
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          wb[j] = e2m1x2_bf16((words[2 * g + j / 4] >> ((j % 4) * 8)) & 0xFFu);
        }
        const float s = e4m3_to_f32((sc >> (8 * g)) & 0xFFu);
#pragma unroll
        for (int r = 0; r < kMaxRows; ++r) {
          if (r < num_rows) {
            const uint4 a0 = a_lds[r * (kChunk / 8) + lds_col + 2 * g];
            const uint4 a1 = a_lds[r * (kChunk / 8) + lds_col + 2 * g + 1];
            float p = dot2(wb[0], a0.x, 0.f);
            p = dot2(wb[1], a0.y, p);
            p = dot2(wb[2], a0.z, p);
            p = dot2(wb[3], a0.w, p);
            p = dot2(wb[4], a1.x, p);
            p = dot2(wb[5], a1.y, p);
            p = dot2(wb[6], a1.z, p);
            p = dot2(wb[7], a1.w, p);
            acc[r] = fmaf(p, s, acc[r]);
          }
        }
      }
    }
  }

  // Combine the two K halves.
  __syncthreads();
  if (half == 1) {
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r)
      if (r < num_rows) red[r][wrow] = acc[r];
  }
  __syncthreads();
  if (half == 0) {
    const float outer = amax[expert] / denominator[0];
    const int col = n_blk * 128 + wrow;
#pragma unroll
    for (int r = 0; r < kMaxRows; ++r) {
      if (r < num_rows) {
        const float v = (acc[r] + red[r][wrow]) * outer;
        uint32_t u = __float_as_uint(v);
        u += 0x7FFFu + ((u >> 16) & 1u);
        out[(int64_t)(row0 + r) * n + col] = (uint16_t)(u >> 16);
      }
    }
  }
}

}  // namespace

void nvfp4_moe_gemv(torch::Tensor& out, const torch::Tensor& act, const torch::Tensor& packed,
                    const torch::Tensor& scales, const torch::Tensor& amax,
                    const torch::Tensor& denominator, const torch::Tensor& tiles,
                    int64_t max_ctas, int64_t tile_rows) {
  TORCH_CHECK(act.scalar_type() == at::kBFloat16 && act.is_contiguous(), "act: contiguous bf16");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous(), "out: contiguous bf16");
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.dim() == 3 && packed.is_contiguous(),
              "packed: contiguous uint8 [E, N, K/2] (pre-tiled)");
  TORCH_CHECK(scales.element_size() == 1 && scales.is_contiguous(), "scales: e4m3 [E, N, K/16]");
  TORCH_CHECK(amax.scalar_type() == at::kFloat && denominator.scalar_type() == at::kFloat,
              "amax / denominator: fp32");
  TORCH_CHECK(tiles.scalar_type() == at::kInt, "tiles: int32");
  TORCH_CHECK(tile_rows <= kMaxRows, "tile_rows must be <= 16");
  const int64_t n = packed.size(1), k = packed.size(2) * 2;
  TORCH_CHECK(n % 128 == 0 && k % 64 == 0, "N % 128 and K % 64 must be 0");
  TORCH_CHECK(act.size(1) == k && out.size(1) == n, "shape mismatch");
  TORCH_CHECK(act.size(0) >= max_ctas * tile_rows && out.size(0) >= max_ctas * tile_rows,
              "act / out rows < max_ctas * tile_rows");
  auto stream = at::hip::getCurrentHIPStream();
  hipLaunchKernelGGL(nvfp4_moe_gemv_kernel, dim3(max_ctas, n / 128), dim3(kThreads), 0, stream,
                     reinterpret_cast<const uint16_t*>(act.data_ptr()),
                     reinterpret_cast<const uint4*>(packed.data_ptr()),
                     reinterpret_cast<const uint32_t*>(scales.data_ptr()),
                     amax.data_ptr<float>(), denominator.data_ptr<float>(),
                     tiles.data_ptr<int32_t>(), reinterpret_cast<uint16_t*>(out.data_ptr()),
                     int(max_ctas), int(tile_rows), int(n), int(k));
}

}  // namespace strix
