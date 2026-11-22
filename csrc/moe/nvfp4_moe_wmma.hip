// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 CIRU AI. Adapted for libstrix.
//
// NVFP4 grouped expert GEMM (Inkling 2 Mini prefill) on gfx1151 fp16 WMMA,
// derived from the paired-M128 / M64-tail K64 kernel of
// https://github.com/ciru-ai/Inkling-2-Mini-CIRU-Halo-Agent-gfx1151
// (overlay/vllm/models/inkling2/amd/ops/hip_moe_prefill.hip, Apache-2.0).
//
// Weights stay NVFP4 in CIRU's pre-tiled layout (`pretile_nvfp4`):
//   packed [E, N/128, K/64, 128 rows, 32 B]   e2m1 pairs, byte j = k 2j low,
//                                             2j+1 high
//   scales [E, N/128, K/64, 128 rows, 4 B]    e4m3, one per 16 columns
// and are decoded to fp16 as they are staged into LDS: one v_perm_b32 maps
// four nibble codes to the fp16 high bytes of 2 x e2m1, the e4m3 scale becomes
// fp16 with a shift (halved, offsetting the 2x), and one v_pk_mul_f16 scales a
// pair. The global scale amax[e] / denominator is applied to the fp32 sums.
//
// Rows come from libstrix's routing layout: expert tile t owns rows
// [t * TILE, row_end[t]) of `act` / `out` (tiles = [expert..., row_end...,
// count]). TILE >= 128: 128-row blocks with more than 64 valid rows run as
// M128, the rest as M64. TILE == 64: every tile runs as M64.
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>

namespace strix {
namespace {

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int kNTile = 128;
constexpr int kKTile = 64;
constexpr int kKStride = 72;  // 64 + 8 fp16 pad: conflict-free b128 LDS reads

__device__ __forceinline__ uint16_t f32_to_bf16_rtne(float x) {
  uint32_t u = __float_as_uint(x);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

// 8 bf16 -> 8 fp16 (exact in fp16 range; activations here are bounded).
__device__ __forceinline__ uint4 bf16x8_to_f16x8(uint4 v) {
  auto cvt2 = [](uint32_t u) -> uint32_t {
    float f0 = __uint_as_float(u << 16);
    float f1 = __uint_as_float(u & 0xFFFF0000u);
    uint32_t pk;
    asm("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(pk) : "v"(f0), "v"(f1));
    return pk;
  };
  return make_uint4(cvt2(v.x), cvt2(v.y), cvt2(v.z), cvt2(v.w));
}

// 8 packed bytes (16 e2m1 codes) and their e4m3 scale -> 16 fp16.
__device__ __forceinline__ void unpack_nvfp4_16(uint32_t p0, uint32_t p1,
                                                uint32_t scale_byte, uint4& o0,
                                                uint4& o1) {
  // e4m3 -> fp16 bits, exponent rebiased by 7 (not 8): half the scale.
  const uint32_t scale_f16 = (scale_byte << 7) + 0x1C00u;
  const uint32_t scale_pk = scale_f16 | (scale_f16 << 16);
  uint32_t dst[8];
  const uint32_t words[2] = {p0, p1};
#pragma unroll
  for (int w = 0; w < 2; ++w) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t packed = (words[w] >> (b * 8)) & 0xFFu;
      // fp16 high bytes of 2 x {0, .5, 1, 1.5, 2, 3, 4, 6}, low byte zero.
      const uint32_t sel =
          0x000C000Cu | ((packed & 7u) << 8) | ((packed & 0x70u) << 20);
      uint32_t mags;
      asm("v_perm_b32 %0, %1, %2, %3"
          : "=v"(mags)
          : "v"(0x4A484644u), "v"(0x42403C00u), "v"(sel));
      const uint32_t sgn = mags | ((packed & 8u) << 12) | ((packed & 128u) << 24);
      uint32_t scaled;
      asm("v_pk_mul_f16 %0, %1, %2" : "=v"(scaled) : "v"(sgn), "v"(scale_pk));
      dst[w * 4 + b] = scaled;
    }
  }
  o0 = make_uint4(dst[0], dst[1], dst[2], dst[3]);
  o1 = make_uint4(dst[4], dst[5], dst[6], dst[7]);
}

// One program: M_TILE rows of one expert tile x 128 output columns.
template <int M_TILE, int W_M, int W_N>
__global__ __launch_bounds__(32 * W_M * W_N, 2) void nvfp4_moe_wmma_kernel(
    const uint16_t* __restrict__ act,      // bf16 [rows, K]
    const uint32_t* __restrict__ packed,   // [E, N/128, K/64, 128, 8] u32
    const uint32_t* __restrict__ scales,   // [E, N/128, K/64, 128] u32
    const float* __restrict__ amax, const float* __restrict__ denominator,
    const int32_t* __restrict__ tiles, uint16_t* __restrict__ out,  // bf16
    int max_ctas, int tile_rows, int n, int k) {
  constexpr int kThreads = 32 * W_M * W_N;
  constexpr int kWmRows = M_TILE / W_M;
  constexpr int kWnCols = kNTile / W_N;
  constexpr int kBm = kWmRows / 16;
  constexpr int kBn = kWnCols / 16;

  // Blocks of M_TILE rows, grouped by 4 along M per column sweep (CIRU's
  // order: neighbouring blocks share the expert's weight tile in cache).
  const int n_tiles = n / kNTile;
  const int pid = blockIdx.x;
  const int block = (pid / (n_tiles * 4)) * 4 + (pid & 3);
  const int n_blk = (pid % (n_tiles * 4)) / 4;

  // Block -> expert tile and rows. M128 blocks are the 128-row halves of a
  // tile; M64 blocks either the first 64 rows of a 128-row block holding at
  // most 64 valid rows (TILE >= 128) or a whole 64-row tile.
  const int block_rows = (M_TILE == 64 && tile_rows >= 128) ? 128 : M_TILE;
  const int per_tile = tile_rows / block_rows;
  const int tile = block / per_tile;
  if (tile >= min(tiles[2 * max_ctas], max_ctas)) return;
  const int row0 = tile * tile_rows + (block % per_tile) * block_rows;
  const int valid = min(tiles[max_ctas + tile] - row0, block_rows);
  if (valid <= 0) return;
  if (tile_rows >= 128) {
    if (M_TILE == 128 && valid <= 64) return;
    if (M_TILE == 64 && valid > 64) return;
  }
  const int expert = tiles[tile];

  __shared__ uint16_t smem_a[M_TILE * kKStride];
  __shared__ uint16_t smem_b[kNTile * kKStride];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / W_N;
  const int wn = warp % W_N;
  const int lane16 = lane & 15;
  const int half_lane = lane >> 4;

  // A: 8 x 16-byte chunks per row per K tile.
  constexpr int kAIters = (M_TILE * 8) / kThreads;
  const uint16_t* a_ptr[kAIters];
  int a_smem[kAIters];
  bool a_valid[kAIters];
#pragma unroll
  for (int it = 0; it < kAIters; ++it) {
    const int idx = it * kThreads + tid;
    const int r = idx / 8;
    const int c = (idx % 8) * 8;
    a_valid[it] = r < valid;
    a_ptr[it] = act + (int64_t)(row0 + (a_valid[it] ? r : 0)) * k + c;
    a_smem[it] = r * kKStride + c;
  }
  // B: 4 x 16-column scale groups per weight row per K tile.
  constexpr int kBIters = (kNTile * 4) / kThreads;
  const int k_tiles = k / kKTile;
  const int64_t tile0 = ((int64_t)expert * n_tiles + n_blk) * k_tiles;
  const uint32_t* b_pack[kBIters];
  const uint32_t* b_scale[kBIters];
  int b_kk[kBIters];
  int b_smem[kBIters];
#pragma unroll
  for (int it = 0; it < kBIters; ++it) {
    const int idx = it * kThreads + tid;
    const int r = idx / 4;
    const int kk = idx % 4;
    b_scale[it] = scales + tile0 * kNTile + r;
    b_pack[it] = packed + (tile0 * kNTile + r) * 8 + kk * 2;
    b_kk[it] = kk;
    b_smem[it] = r * kKStride + kk * 16;
  }

  float8 acc[kBm][kBn];
#pragma unroll
  for (int im = 0; im < kBm; ++im)
#pragma unroll
    for (int in = 0; in < kBn; ++in) acc[im][in] = float8{0, 0, 0, 0, 0, 0, 0, 0};

  uint4 reg_a[kAIters];
  uint2 reg_p[kBIters];
  uint32_t reg_s[kBIters];
  auto prefetch = [&](int kt) {
#pragma unroll
    for (int it = 0; it < kAIters; ++it)
      reg_a[it] = a_valid[it]
                      ? *reinterpret_cast<const uint4*>(a_ptr[it] + kt * kKTile)
                      : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int it = 0; it < kBIters; ++it) {
      reg_s[it] = b_scale[it][(int64_t)kt * kNTile];
      reg_p[it] = *reinterpret_cast<const uint2*>(b_pack[it] + (int64_t)kt * (kNTile * 8));
    }
  };

  prefetch(0);
  for (int kt = 0; kt < k_tiles; ++kt) {
#pragma unroll
    for (int it = 0; it < kAIters; ++it)
      *reinterpret_cast<uint4*>(smem_a + a_smem[it]) = bf16x8_to_f16x8(reg_a[it]);
#pragma unroll
    for (int it = 0; it < kBIters; ++it) {
      uint4 o0, o1;
      unpack_nvfp4_16(reg_p[it].x, reg_p[it].y, (reg_s[it] >> (b_kk[it] * 8)) & 0xFFu,
                      o0, o1);
      uint4* sp = reinterpret_cast<uint4*>(smem_b + b_smem[it]);
      sp[0] = o0;
      sp[1] = o1;
    }
    __syncthreads();
    if (kt + 1 < k_tiles) prefetch(kt + 1);
#pragma unroll
    for (int kk = 0; kk < kKTile; kk += 16) {
      half16 a_frag[kBm];
#pragma unroll
      for (int im = 0; im < kBm; ++im) {
        const int row = wm * kWmRows + im * 16 + lane16;
        const uint4* ap = reinterpret_cast<const uint4*>(smem_a + row * kKStride + kk);
        uint4 pair[2] = {ap[0], ap[1]};
        __builtin_memcpy(&a_frag[im], pair, sizeof(half16));
      }
#pragma unroll
      for (int in = 0; in < kBn; ++in) {
        const int col = wn * kWnCols + in * 16 + lane16;
        const uint4* bp = reinterpret_cast<const uint4*>(smem_b + col * kKStride + kk);
        uint4 pair[2] = {bp[0], bp[1]};
        half16 b_frag;
        __builtin_memcpy(&b_frag, pair, sizeof(half16));
#pragma unroll
        for (int im = 0; im < kBm; ++im)
          acc[im][in] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag[im], b_frag, acc[im][in]);
      }
    }
    __syncthreads();
  }

  // WMMA C layout: element i of a lane is row half_lane + 2 i, column lane16.
  const float outer = amax[expert] / denominator[0];
  const int col_base = n_blk * kNTile + wn * kWnCols;
#pragma unroll
  for (int im = 0; im < kBm; ++im) {
    const int m_base = wm * kWmRows + im * 16;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = m_base + half_lane + 2 * i;
#pragma unroll
      for (int in = 0; in < kBn; ++in) {
        const uint16_t h0 = f32_to_bf16_rtne(acc[im][in][i] * outer);
        const uint32_t h1 = __shfl_down((uint32_t)h0, 1, 16);
        if ((lane16 & 1) == 0 && r < valid) {
          const int col = col_base + in * 16 + lane16;
          *reinterpret_cast<uint32_t*>(out + (int64_t)(row0 + r) * n + col) =
              (uint32_t)h0 | (h1 << 16);
        }
      }
    }
  }
}

}  // namespace

void nvfp4_moe_wmma(torch::Tensor& out, const torch::Tensor& act,
                    const torch::Tensor& packed, const torch::Tensor& scales,
                    const torch::Tensor& amax, const torch::Tensor& denominator,
                    const torch::Tensor& tiles, int64_t max_ctas, int64_t tile_rows) {
  TORCH_CHECK(act.scalar_type() == at::kBFloat16 && act.is_contiguous(), "act: contiguous bf16");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous(), "out: contiguous bf16");
  TORCH_CHECK(packed.scalar_type() == at::kByte && packed.dim() == 3 && packed.is_contiguous(),
              "packed: contiguous uint8 [E, N, K/2] (pre-tiled)");
  TORCH_CHECK(scales.element_size() == 1 && scales.is_contiguous(), "scales: e4m3 [E, N, K/16]");
  TORCH_CHECK(tiles.scalar_type() == at::kInt, "tiles: int32");
  TORCH_CHECK(amax.scalar_type() == at::kFloat && denominator.scalar_type() == at::kFloat,
              "amax / denominator: fp32");
  const int64_t n = packed.size(1);
  const int64_t k = packed.size(2) * 2;
  TORCH_CHECK(n % kNTile == 0 && k % kKTile == 0, "N % 128 and K % 64 must be 0");
  TORCH_CHECK(act.size(1) == k && out.size(1) == n, "shape mismatch");
  TORCH_CHECK(tile_rows == 64 || (tile_rows >= 128 && tile_rows % 128 == 0),
              "tile_rows: 64 or a multiple of 128");
  TORCH_CHECK(act.size(0) >= max_ctas * tile_rows && out.size(0) >= max_ctas * tile_rows,
              "act / out rows < max_ctas * tile_rows");

  auto stream = at::hip::getCurrentHIPStream();
  const int n_tiles = int(n / kNTile);
  auto args = [&](auto kernel, int blocks, int threads) {
    const int grid = ((blocks + 3) / 4) * 4 * n_tiles;
    hipLaunchKernelGGL(kernel, dim3(grid), dim3(threads), 0, stream,
                       reinterpret_cast<const uint16_t*>(act.data_ptr()),
                       reinterpret_cast<const uint32_t*>(packed.data_ptr()),
                       reinterpret_cast<const uint32_t*>(scales.data_ptr()),
                       amax.data_ptr<float>(), denominator.data_ptr<float>(),
                       tiles.data_ptr<int32_t>(),
                       reinterpret_cast<uint16_t*>(out.data_ptr()), int(max_ctas),
                       int(tile_rows), int(n), int(k));
  };
  if (tile_rows >= 128) {
    const int blocks = int(max_ctas * (tile_rows / 128));
    args(nvfp4_moe_wmma_kernel<128, 2, 4>, blocks, 256);
    args(nvfp4_moe_wmma_kernel<64, 1, 4>, blocks, 128);
  } else {
    args(nvfp4_moe_wmma_kernel<64, 1, 4>, int(max_ctas), 128);
  }
}

}  // namespace strix
