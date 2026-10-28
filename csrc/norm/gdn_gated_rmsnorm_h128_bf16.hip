// SPDX-License-Identifier: MIT
// out = rms_norm(x) * w * act(z) over 128-channel rows (GDN output norm,
// RMSNormGated with norm_before_gate=True, group_size=None).
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kD = 128;
// 16 lanes per row, 8 channels (one 16-byte load) per lane.
constexpr int kLanesPerRow = 16;
constexpr int kRowsPerBlock = 16;

enum class Act { kSilu = 0, kSigmoid = 1 };

template <typename WT, Act A>
__global__ __launch_bounds__(kRowsPerBlock * kLanesPerRow) void gated_rmsnorm_h128(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ z, const WT* __restrict__ w, int64_t rows,
    int heads, int64_t x_st, int64_t x_sh, int64_t z_st, int64_t z_sh,
    float eps) {
  const int lane = threadIdx.x % kLanesPerRow;
  const int64_t row =
      int64_t(blockIdx.x) * kRowsPerBlock + threadIdx.x / kLanesPerRow;
  if (row >= rows) return;
  const int64_t t = row / heads, h = row % heads;

  const uint4 xv =
      *reinterpret_cast<const uint4*>(x + t * x_st + h * x_sh + lane * 8);
  const uint4 zv =
      *reinterpret_cast<const uint4*>(z + t * z_st + h * z_sh + lane * 8);
  const uint32_t xp[4] = {xv.x, xv.y, xv.z, xv.w};
  const uint32_t zp[4] = {zv.x, zv.y, zv.z, zv.w};
  float xf[8], zf[8], wf[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    xf[2 * i] = bf16_lo(xp[i]), xf[2 * i + 1] = bf16_hi(xp[i]);
    zf[2 * i] = bf16_lo(zp[i]), zf[2 * i + 1] = bf16_hi(zp[i]);
  }

  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += xf[i] * xf[i];
  const float inv = rsqrtf(group_sum<kLanesPerRow>(ss) * (1.f / kD) + eps);

  if constexpr (sizeof(WT) == 4) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float4 wv = *reinterpret_cast<const float4*>(w + lane * 8 + 4 * i);
      wf[4 * i] = wv.x, wf[4 * i + 1] = wv.y;
      wf[4 * i + 2] = wv.z, wf[4 * i + 3] = wv.w;
    }
  } else {
    const uint4 wv = *reinterpret_cast<const uint4*>(w + lane * 8);
    const uint32_t wp[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      wf[2 * i] = bf16_lo(wp[i]), wf[2 * i + 1] = bf16_hi(wp[i]);
  }

  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float r[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int k = 2 * i + j;
      const float g = 1.f / (1.f + __expf(-zf[k]));
      r[j] = xf[k] * inv * wf[k] * (A == Act::kSilu ? zf[k] * g : g);
    }
    o[i] = pack_bf16x2(r[0], r[1]);
  }
  *reinterpret_cast<uint4*>(out + row * kD + lane * 8) =
      make_uint4(o[0], o[1], o[2], o[3]);
}

void check_rows(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.dim() == 2 || t.dim() == 3, name, " must be 2D or 3D");
  TORCH_CHECK(t.size(-1) == kD && t.stride(-1) == 1, name,
              " needs a contiguous last dim of ", kD);
  TORCH_CHECK(t.scalar_type() == at::kBFloat16, name, " must be bf16");
  for (int i = 0; i < t.dim() - 1; ++i)
    TORCH_CHECK(t.stride(i) % 8 == 0, name, " strides must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
              " must be 16-byte aligned");
}

}  // namespace

// x, z: [T, D] or [T, H, D], bf16, last dim contiguous. Returns a contiguous
// tensor of x's shape.
torch::Tensor gdn_gated_rmsnorm_h128_bf16(const torch::Tensor& x,
                                          const torch::Tensor& z,
                                          const torch::Tensor& weight,
                                          double eps, int64_t activation) {
  check_rows(x, "x");
  check_rows(z, "z");
  TORCH_CHECK(x.sizes() == z.sizes(), "x and z shapes differ");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == kD && weight.is_contiguous(),
              "weight must be contiguous [", kD, "]");
  TORCH_CHECK(activation == 0 || activation == 1,
              "activation must be 0 (silu) or 1 (sigmoid)");

  auto out = torch::empty(x.sizes(), x.options().memory_format(
                                         at::MemoryFormat::Contiguous));
  const bool three = x.dim() == 3;
  const int heads = three ? x.size(1) : 1;
  const int64_t rows = x.numel() / kD;
  if (rows == 0) return out;
  const int64_t x_sh = three ? x.stride(1) : 0, z_sh = three ? z.stride(1) : 0;

  const dim3 grid((rows + kRowsPerBlock - 1) / kRowsPerBlock);
  const dim3 block(kRowsPerBlock * kLanesPerRow);
  auto stream = at::hip::getCurrentHIPStream();
  auto o = reinterpret_cast<uint16_t*>(out.data_ptr());
  auto xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  auto zp = reinterpret_cast<const uint16_t*>(z.data_ptr());

#define STRIX_LAUNCH(WT, A)                                                  \
  gated_rmsnorm_h128<WT, A><<<grid, block, 0, stream>>>(                     \
      o, xp, zp, reinterpret_cast<const WT*>(weight.data_ptr()), rows, heads, \
      x.stride(0), x_sh, z.stride(0), z_sh, static_cast<float>(eps))

  if (weight.scalar_type() == at::kFloat) {
    if (activation == 0) STRIX_LAUNCH(float, Act::kSilu);
    else STRIX_LAUNCH(float, Act::kSigmoid);
  } else {
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16,
                "weight must be bf16 or fp32");
    if (activation == 0) STRIX_LAUNCH(uint16_t, Act::kSilu);
    else STRIX_LAUNCH(uint16_t, Act::kSigmoid);
  }
#undef STRIX_LAUNCH
  return out;
}

}  // namespace strix
