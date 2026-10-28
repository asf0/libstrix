// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// HIP port of vLLM's csrc/gdn/fused_gdn_decode_kernel.cu (CUDA only upstream):
// GDN speculative decode (MTP) for one request x value head per block: L2
// norm of q/k, gating, the delta-rule recurrence over the request's draft
// tokens from the last accepted state, a state write per token slot, and the
// gated RMS norm of the output. Same math and thread layout; cp.async becomes
// direct global loads, warp shuffles use wave32 sums, bf16 is raw uint16.
//
// strix_vllm.gdn exposes it as torch.ops._C.fused_gdn_decode_post_conv_mtp
// (vLLM's schema) when vLLM's own build lacks that op:
// QwenGatedDeltaNetAttention takes the fused path when the op exists, and
// otherwise falls back to the Triton spec-decode kernels.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>
#include <torch/library.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kDimK = 128, kDimV = 128;
constexpr int kThreads = 256, kWarps = kThreads / kWave;
constexpr int kChunkV = 32, kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxMtpTokens = 8;
constexpr int kDtF32 = 0, kDtBf16 = 1, kDtF16 = 2;
static_assert(kRowsPerWarp == 4, "row bookkeeping below assumes 4 rows per wave");

struct Strides {
  int64_t mixed_row, a_row, b_row, gate_row, state_slot;
};

__device__ __forceinline__ float bf(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ __forceinline__ uint16_t to_bf(float x) {
  return static_cast<uint16_t>(pack_bf16x2(x, 0.f) & 0xffffu);
}
__device__ __forceinline__ float sigmoid_fast(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.f ? x : log1pf(__expf(x));
}

template <typename StateT, int VPK, bool SigmoidGate>
__global__ __launch_bounds__(kThreads, 2) void gdn_decode_mtp_kernel(
    const uint16_t* __restrict__ mixed_qkv, const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b, const float* __restrict__ a_log,
    const void* __restrict__ dt_bias, const int* __restrict__ state_indices,
    const int* __restrict__ cu_seqlens, const int* __restrict__ num_accepted,
    StateT* __restrict__ state, const uint16_t* __restrict__ output_gate,
    const void* __restrict__ norm_weight, uint16_t* __restrict__ out, int H, int HV,
    int si_width, int dt_type, bool norm_w_bf16, float scale, float norm_eps,
    Strides st) {
  const int request = blockIdx.x, vh = blockIdx.y;
  const int tid = threadIdx.x, lane = tid % kWave, warp = tid / kWave;
  const int bos = cu_seqlens[request], eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) return;

  const int accepted = num_accepted[request];
  const int source_slot = accepted > 0 && accepted <= si_width
                              ? state_indices[request * si_width + accepted - 1]
                              : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int i = tid; i < num_tokens * kDimV; i += kThreads) {
      const int token = bos + i / kDimV, v = i % kDimV;
      out[(int64_t(token) * HV + vh) * kDimV + v] = 0;
    }
    return;
  }

  const int kh = vh / VPK;
  __shared__ __attribute__((aligned(16))) float s_q[kMaxMtpTokens][kDimK];
  __shared__ __attribute__((aligned(16))) float s_k[kMaxMtpTokens][kDimK];
  __shared__ uint16_t s_v[kMaxMtpTokens][kDimV];
  __shared__ uint16_t s_out[kMaxMtpTokens][kDimV];
  __shared__ float s_decay[kMaxMtpTokens], s_beta[kMaxMtpTokens];

  // Each thread reads its 4 rows x 4 columns of a chunk straight from global
  // (a wave covers 512 contiguous bytes per row): staging the chunk in LDS,
  // as the CUDA version does with cp.async, cost occupancy (45 KB per block)
  // and measured 11% slower at 16 requests.
  const StateT* src = state + int64_t(source_slot) * st.state_slot + vh * kDimV * kDimK;

  if (warp < num_tokens) {
    const int t = warp, token = bos + t;
    const int64_t base = int64_t(token) * st.mixed_row;
    float qv[4], kv[4], qsq = 0.f, ksq = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      qv[i] = bf(mixed_qkv[base + kh * kDimK + d]);
      kv[i] = bf(mixed_qkv[base + H * kDimK + kh * kDimK + d]);
      s_v[t][d] = mixed_qkv[base + 2 * H * kDimK + vh * kDimV + d];
      qsq += qv[i] * qv[i];
      ksq += kv[i] * kv[i];
    }
    const float q_scale = rsqrtf(group_sum<32>(qsq) + 1e-6f) * scale;
    const float k_scale = rsqrtf(group_sum<32>(ksq) + 1e-6f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      s_q[t][lane + i * 32] = qv[i] * q_scale;
      s_k[t][lane + i * 32] = kv[i] * k_scale;
    }
    if (lane == 0) {
      const float av = bf(a[int64_t(token) * st.a_row + vh]);
      const float bv = bf(b[int64_t(token) * st.b_row + vh]);
      const float dtb = dt_type == kDtBf16   ? bf(static_cast<const uint16_t*>(dt_bias)[vh])
                        : dt_type == kDtF16 ? __half2float(static_cast<const __half*>(dt_bias)[vh])
                                            : static_cast<const float*>(dt_bias)[vh];
      const float g = -__expf(a_log[vh]) * softplus_fast(av + dtb);
      s_decay[t] = __expf(g);
      s_beta[t] = sigmoid_fast(bv);
    }
  }
  barrier();

  const int k_base = lane * 4;
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    float h[kRowsPerWarp][4];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int v = chunk * kChunkV + warp + r * kWarps;
      const float4 sv = load_state4(src + v * kDimK + k_base);
      h[r][0] = sv.x, h[r][1] = sv.y, h[r][2] = sv.z, h[r][3] = sv.w;
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&s_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&s_k[t][k_base]);
      const float qv[4] = {q4.x, q4.y, q4.z, q4.w};
      const float kv[4] = {k4.x, k4.y, k4.z, k4.w};
      const float decay = s_decay[t], beta = s_beta[t];

      float hk[kRowsPerWarp] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[r][i] *= decay;
          hk[r] += h[r][i] * kv[i];
        }
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) hk[r] = group_sum<32>(hk[r]);

      float hq[kRowsPerWarp] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) {
        const int v = chunk * kChunkV + warp + r * kWarps;
        const float delta = (bf(s_v[t][v]) - hk[r]) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[r][i] += kv[i] * delta;
          hq[r] += h[r][i] * qv[i];
        }
      }
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) hq[r] = group_sum<32>(hq[r]);
      if (lane == 0)
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r)
          s_out[t][chunk * kChunkV + warp + r * kWarps] = to_bf(hq[r]);

      const int dst_slot = state_indices[request * si_width + t];
      if (dst_slot > 0) {
        StateT* dst = state + int64_t(dst_slot) * st.state_slot + vh * kDimV * kDimK;
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
          const int v = chunk * kChunkV + warp + r * kWarps;
          store_state4(dst + v * kDimK + k_base,
                       make_float4(h[r][0], h[r][1], h[r][2], h[r][3]));
        }
      }
    }
  }
  barrier();

  if (warp < num_tokens) {
    const int t = warp, token = bos + t;
    float ov[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ov[i] = bf(s_out[t][lane + i * 32]);
      ss += ov[i] * ov[i];
    }
    const float rstd = rsqrtf(group_sum<32>(ss) / float(kDimV) + norm_eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int v = lane + i * 32;
      const float gi = bf(output_gate[int64_t(token) * st.gate_row + vh * kDimV + v]);
      const float gate = SigmoidGate ? sigmoid_fast(gi) : gi * sigmoid_fast(gi);
      const float w = norm_w_bf16 ? bf(static_cast<const uint16_t*>(norm_weight)[v])
                                  : static_cast<const float*>(norm_weight)[v];
      out[(int64_t(token) * HV + vh) * kDimV + v] = to_bf(ov[i] * rstd * w * gate);
    }
  }
}

}  // namespace

void gdn_decode_mtp_bf16(
    const torch::Tensor& mixed_qkv, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_log, const torch::Tensor& dt_bias,
    const torch::Tensor& state_indices, const torch::Tensor& cu_seqlens,
    const torch::Tensor& num_accepted_tokens, torch::Tensor& state,
    const torch::Tensor& output_gate, const torch::Tensor& norm_weight,
    torch::Tensor& out, double scale, double norm_eps,
    c10::string_view output_gate_activation) {
  using at::ScalarType;
  TORCH_CHECK(mixed_qkv.scalar_type() == ScalarType::BFloat16 && mixed_qkv.dim() == 2 &&
                  mixed_qkv.stride(1) == 1,
              "mixed_qkv must be bf16 [L, 2*H*128 + HV*128] with contiguous channels");
  TORCH_CHECK(a.scalar_type() == ScalarType::BFloat16 && b.scalar_type() == ScalarType::BFloat16 &&
                  a.stride(1) == 1 && b.stride(1) == 1,
              "a and b must be bf16 with contiguous heads");
  TORCH_CHECK(a_log.scalar_type() == ScalarType::Float && a_log.is_contiguous(), "A_log fp32");
  const auto dtt = dt_bias.scalar_type();
  TORCH_CHECK(dt_bias.is_contiguous() &&
                  (dtt == ScalarType::Float || dtt == ScalarType::BFloat16 || dtt == ScalarType::Half),
              "dt_bias must be fp32, bf16 or fp16");
  for (const auto* t : {&state_indices, &cu_seqlens, &num_accepted_tokens})
    TORCH_CHECK(t->scalar_type() == ScalarType::Int && t->is_contiguous(), "int32 metadata");
  const auto stt = state.scalar_type();
  TORCH_CHECK((stt == ScalarType::Float || stt == ScalarType::BFloat16) && state.dim() == 4 &&
                  state.size(2) == kDimV && state.size(3) == kDimK &&
                  state.stride(1) == kDimV * kDimK && state.stride(2) == kDimK &&
                  state.stride(3) == 1,
              "state must be fp32/bf16 [slots, HV, 128, 128] with contiguous slot contents");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                  state.stride(0) % (stt == ScalarType::Float ? 4 : 8) == 0,
              "state slots must keep 16-byte alignment");
  TORCH_CHECK(output_gate.scalar_type() == ScalarType::BFloat16 && output_gate.dim() == 3 &&
                  output_gate.stride(2) == 1 && output_gate.stride(1) == kDimV,
              "output_gate must be bf16 [L, HV, 128] with contiguous head rows");
  TORCH_CHECK(norm_weight.is_contiguous() && norm_weight.numel() == kDimV &&
                  (norm_weight.scalar_type() == ScalarType::Float ||
                   norm_weight.scalar_type() == ScalarType::BFloat16),
              "norm_weight must be fp32/bf16 [128]");
  TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16 && out.is_contiguous(), "out bf16");
  TORCH_CHECK(output_gate_activation == "silu" || output_gate_activation == "sigmoid",
              "output_gate_activation must be 'silu' or 'sigmoid'");
  const int num_tokens = mixed_qkv.size(0);
  TORCH_CHECK(num_tokens > 0, "needs at least one token");
  const int hv = state.size(1);
  const int64_t key_width = mixed_qkv.size(1) - int64_t(hv) * kDimV;
  TORCH_CHECK(key_width > 0 && key_width % (2 * kDimK) == 0, "mixed_qkv width vs state");
  const int h = key_width / (2 * kDimK), vpk = hv / h;
  TORCH_CHECK(hv % h == 0 && ((vpk >= 1 && vpk <= 4) || vpk == 8), "HV/H in {1, 2, 3, 4, 8}");
  TORCH_CHECK(state_indices.dim() == 2 && state_indices.size(1) >= 1 &&
                  state_indices.size(1) <= kMaxMtpTokens,
              "state_indices [N, S], 1 <= S <= 8");
  const int n = state_indices.size(0);
  TORCH_CHECK(cu_seqlens.numel() == n + 1 && num_accepted_tokens.numel() == n, "N mismatch");
  TORCH_CHECK(a.size(0) == num_tokens && a.size(1) == hv && b.size(0) == num_tokens &&
                  b.size(1) == hv && a_log.numel() == hv && dt_bias.numel() == hv &&
                  output_gate.size(0) == num_tokens && output_gate.size(1) == hv &&
                  out.size(0) == num_tokens && out.size(1) == hv && out.size(2) == kDimV,
              "shape mismatch");
  if (n == 0) return;
  const Strides st{mixed_qkv.stride(0), a.stride(0), b.stride(0), output_gate.stride(0),
                   state.stride(0)};
  const int dt_type = dtt == ScalarType::Float ? kDtF32 : dtt == ScalarType::BFloat16 ? kDtBf16 : kDtF16;
  const bool sig = output_gate_activation == "sigmoid";
  auto stream = at::hip::getCurrentHIPStream();
  const dim3 grid(n, hv);
#define STRIX_GDN_MTP(ST, VPK, SIG)                                          \
  gdn_decode_mtp_kernel<ST, VPK, SIG><<<grid, kThreads, 0, stream>>>(        \
      reinterpret_cast<const uint16_t*>(mixed_qkv.data_ptr()),               \
      reinterpret_cast<const uint16_t*>(a.data_ptr()),                       \
      reinterpret_cast<const uint16_t*>(b.data_ptr()), a_log.data_ptr<float>(), \
      dt_bias.data_ptr(), state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(), \
      num_accepted_tokens.data_ptr<int>(), reinterpret_cast<ST*>(state.data_ptr()), \
      reinterpret_cast<const uint16_t*>(output_gate.data_ptr()),             \
      norm_weight.data_ptr(), reinterpret_cast<uint16_t*>(out.data_ptr()), h, hv, \
      state_indices.size(1), dt_type, norm_weight.scalar_type() == ScalarType::BFloat16, \
      static_cast<float>(scale), static_cast<float>(norm_eps), st)
#define STRIX_GDN_MTP_VPK(ST, SIG)                                           \
  switch (vpk) {                                                             \
    case 1: STRIX_GDN_MTP(ST, 1, SIG); break;                                \
    case 2: STRIX_GDN_MTP(ST, 2, SIG); break;                                \
    case 3: STRIX_GDN_MTP(ST, 3, SIG); break;                                \
    case 4: STRIX_GDN_MTP(ST, 4, SIG); break;                                \
    default: STRIX_GDN_MTP(ST, 8, SIG); break;                               \
  }
  if (stt == ScalarType::Float) {
    if (sig) { STRIX_GDN_MTP_VPK(float, true) } else { STRIX_GDN_MTP_VPK(float, false) }
  } else {
    if (sig) { STRIX_GDN_MTP_VPK(uint16_t, true) } else { STRIX_GDN_MTP_VPK(uint16_t, false) }
  }
#undef STRIX_GDN_MTP_VPK
#undef STRIX_GDN_MTP
}

}  // namespace strix

