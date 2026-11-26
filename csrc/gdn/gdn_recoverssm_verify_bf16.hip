// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// RecoverSSM speculative verify for Qwen GDN (vLLM's
// gdn_recoverssm_verify): the 1 + k token window of each request from its
// checkpoint, without writing the recurrent state. Records each token's decay,
// normalized key and beta-scaled correction in the checkpoint block; the commit
// replays the accepted prefix from them. One request x value head per block,
// or a quarter of a head when there are few, in the thread layout of
// gdn_decode_mtp_bf16 (whose state update, L2 norm and
// gating it shares); the output norm and gate stay with the caller.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kDimK = 128, kDimV = 128;
constexpr int kThreads = 256, kWarps = kThreads / kWave;
constexpr int kChunkV = 32, kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxTokens = 8;
constexpr int kDtF32 = 0, kDtBf16 = 1, kDtF16 = 2;
static_assert(kRowsPerWarp == 4, "row bookkeeping below assumes 4 rows per wave");

struct VerifyStrides {
  int64_t q_tok, k_tok, v_tok, a_row, b_row, state_block;
  int64_t corr_block, corr_head, corr_pos, key_block, key_head, key_pos;
  int64_t decay_block, decay_head, out_tok, si;
};

__device__ __forceinline__ float bf(uint16_t x) { return __uint_as_float(uint32_t(x) << 16); }
__device__ __forceinline__ uint16_t to_bf(float x) {
  return static_cast<uint16_t>(pack_bf16x2(x, 0.f) & 0xffffu);
}
__device__ __forceinline__ float sigmoid_fast(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.f ? x : log1pf(__expf(x));
}

template <typename StateT, int VPK>
__global__ __launch_bounds__(kThreads, 2) void gdn_recoverssm_verify_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v, const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b, const float* __restrict__ a_log,
    const void* __restrict__ dt_bias, const int* __restrict__ cu_seqlens,
    const int64_t* __restrict__ state_indices, const StateT* __restrict__ state,
    float* __restrict__ correction, float* __restrict__ keys,
    float* __restrict__ decays, uint16_t* __restrict__ out, int dt_type,
    int64_t null_block, float scale, int cpb, int rec_len, VerifyStrides st) {
  // Block z of a head runs its 32-row slices z * cpb .. z * cpb + cpb - 1:
  // cpb = 1 spreads a head over 4 blocks (few requests), cpb = 4 keeps it in one.
  const int request = blockIdx.x, vh = blockIdx.y, chunk0 = blockIdx.z * cpb;
  const int tid = threadIdx.x, lane = tid % kWave, warp = tid / kWave;
  const int bos = cu_seqlens[request], eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) return;
  const int64_t slot = state_indices[int64_t(request) * st.si];
  if (slot <= null_block || num_tokens > rec_len) {
    for (int i = tid; i < num_tokens * kChunkV * cpb; i += kThreads)
      out[int64_t(bos + i / (kChunkV * cpb)) * st.out_tok + vh * kDimV + chunk0 * kChunkV +
          i % (kChunkV * cpb)] = 0;
    return;
  }
  const int kh = vh / VPK;
  __shared__ __attribute__((aligned(16))) float s_q[kMaxTokens][kDimK];
  __shared__ __attribute__((aligned(16))) float s_k[kMaxTokens][kDimK];
  __shared__ float s_v[kMaxTokens][kDimV];
  __shared__ float s_decay[kMaxTokens], s_beta[kMaxTokens];
  const StateT* src = state + slot * st.state_block + vh * kDimV * kDimK;

  if (warp < num_tokens) {
    const int t = warp, token = bos + t;
    float qv[4], kv[4], qsq = 0.f, ksq = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * 32;
      qv[i] = bf(q[int64_t(token) * st.q_tok + kh * kDimK + d]);
      kv[i] = bf(k[int64_t(token) * st.k_tok + kh * kDimK + d]);
      s_v[t][d] = bf(v[int64_t(token) * st.v_tok + vh * kDimV + d]);
      qsq += qv[i] * qv[i];
      ksq += kv[i] * kv[i];
    }
    const float q_scale = rsqrtf(group_sum<32>(qsq) + 1e-6f) * scale;
    const float k_scale = rsqrtf(group_sum<32>(ksq) + 1e-6f);
    const bool write_key = vh % VPK == 0 && chunk0 == 0;
    float* kdst = keys + slot * st.key_block + kh * st.key_head + t * st.key_pos;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      s_q[t][lane + i * 32] = qv[i] * q_scale;
      s_k[t][lane + i * 32] = kv[i] * k_scale;
      if (write_key) kdst[lane + i * 32] = kv[i] * k_scale;
    }
    if (lane == 0) {
      const float av = bf(a[int64_t(token) * st.a_row + vh]);
      const float bv = bf(b[int64_t(token) * st.b_row + vh]);
      const float dtb = dt_type == kDtBf16   ? bf(static_cast<const uint16_t*>(dt_bias)[vh])
                        : dt_type == kDtF16 ? __half2float(static_cast<const __half*>(dt_bias)[vh])
                                            : static_cast<const float*>(dt_bias)[vh];
      s_decay[t] = __expf(-__expf(a_log[vh]) * softplus_fast(av + dtb));
      s_beta[t] = sigmoid_fast(bv);
      if (chunk0 == 0) decays[slot * st.decay_block + vh * st.decay_head + t] = s_decay[t];
    }
  }
  barrier();

  const int k_base = lane * 4;
  float* corr = correction + slot * st.corr_block + vh * st.corr_head;
  for (int chunk = chunk0; chunk < chunk0 + cpb; ++chunk) {
    float h[kRowsPerWarp][4];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      const int vv = chunk * kChunkV + warp + r * kWarps;
      const float4 sv = load_state4(src + vv * kDimK + k_base);
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
          h[r][i] = __fmul_rn(h[r][i], decay);  // rounded, never fused
          hk[r] += h[r][i] * kv[i];
        }
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) hk[r] = group_sum<32>(hk[r]);
      float hq[kRowsPerWarp] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) {
        const int vv = chunk * kChunkV + warp + r * kWarps;
        const float delta = (s_v[t][vv] - hk[r]) * beta;
        if (lane == 0) corr[t * st.corr_pos + vv] = delta;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          // The commit replays fma(correction, key, h * decay): the decayed
          // state stays rounded (it is a separate multiply above) and the
          // rank-1 update is an explicit fma, so -ffp-contract cannot pick
          // another fusion and the replayed state matches this one.
          h[r][i] = __builtin_fmaf(kv[i], delta, h[r][i]);
          hq[r] += h[r][i] * qv[i];
        }
      }
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) hq[r] = group_sum<32>(hq[r]);
      if (lane == 0)
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r)
          out[int64_t(bos + t) * st.out_tok + vh * kDimV + chunk * kChunkV + warp +
              r * kWarps] = to_bf(hq[r]);
    }
  }
}

}  // namespace

void gdn_recoverssm_verify_bf16(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_log,
    const torch::Tensor& dt_bias, const torch::Tensor& state,
    torch::Tensor& correction, torch::Tensor& keys, torch::Tensor& decays,
    const torch::Tensor& cu_seqlens, const torch::Tensor& state_indices,
    torch::Tensor& out, double scale, int64_t null_block) {
  using at::ScalarType;
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(t->scalar_type() == ScalarType::BFloat16 && t->dim() == 3 &&
                    t->size(2) == 128 && t->stride(2) == 1 && t->stride(1) == 128,
                "q, k, v must be bf16 [T, H, 128] with contiguous heads");
  TORCH_CHECK(a.scalar_type() == ScalarType::BFloat16 && b.scalar_type() == ScalarType::BFloat16 &&
                  a.stride(1) == 1 && b.stride(1) == 1,
              "a and b must be bf16 with contiguous heads");
  TORCH_CHECK(a_log.scalar_type() == ScalarType::Float && a_log.is_contiguous(), "A_log fp32");
  const auto dtt = dt_bias.scalar_type();
  TORCH_CHECK(dt_bias.is_contiguous() &&
                  (dtt == ScalarType::Float || dtt == ScalarType::BFloat16 || dtt == ScalarType::Half),
              "dt_bias must be fp32, bf16 or fp16");
  const auto stt = state.scalar_type();
  TORCH_CHECK((stt == ScalarType::Float || stt == ScalarType::BFloat16) && state.dim() == 4 &&
                  state.size(2) == kDimV && state.size(3) == kDimK &&
                  state.stride(1) == kDimV * kDimK && state.stride(2) == kDimK && state.stride(3) == 1,
              "state must be fp32/bf16 [slots, HV, 128, 128] with contiguous slot contents");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                  state.stride(0) % (stt == ScalarType::Float ? 4 : 8) == 0,
              "state slots must keep 16-byte alignment");
  for (const auto* t : {&correction, &keys, &decays})
    TORCH_CHECK(t->scalar_type() == ScalarType::Float && t->stride(-1) == 1,
                "records must be fp32 with contiguous rows");
  TORCH_CHECK(cu_seqlens.scalar_type() == ScalarType::Int && cu_seqlens.is_contiguous() &&
                  state_indices.scalar_type() == ScalarType::Long,
              "cu_seqlens int32, state_indices int64");
  TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16 && out.dim() == 3, "out bf16 [T, HV, 128]");
  const int hv = state.size(1), hk = q.size(1), vpk = hv / hk;
  TORCH_CHECK(hk > 0 && hv % hk == 0 && ((vpk >= 1 && vpk <= 4) || vpk == 8), "HV/H in {1,2,3,4,8}");
  TORCH_CHECK(v.size(1) == hv && k.size(1) == hk && a.size(1) == hv && b.size(1) == hv &&
                  a_log.numel() == hv && dt_bias.numel() == hv && out.size(1) == hv &&
                  correction.size(1) == hv && keys.size(1) == hk && decays.size(1) == hv &&
                  correction.size(2) <= kMaxTokens && keys.size(2) == correction.size(2),
              "shape mismatch");
  const int n = state_indices.size(0);
  TORCH_CHECK(cu_seqlens.numel() == n + 1, "N mismatch");
  if (n == 0) return;
  const VerifyStrides st{q.stride(0),          k.stride(0),         v.stride(0),
                         a.stride(0),          b.stride(0),         state.stride(0),
                         correction.stride(0), correction.stride(1), correction.stride(2),
                         keys.stride(0),       keys.stride(1),      keys.stride(2),
                         decays.stride(0),     decays.stride(1),    out.stride(0),
                         state_indices.stride(0)};
  const int dt_type = dtt == ScalarType::Float ? kDtF32 : dtt == ScalarType::BFloat16 ? kDtBf16 : kDtF16;
  auto stream = at::hip::getCurrentHIPStream();
  // Few blocks (one request): a head over 4 blocks, 1.30x at one 35B request;
  // more: one block per head, which measured faster from 2 requests on.
  const int cpb = int64_t(n) * hv < 64 ? 1 : kNumChunks;
  const dim3 grid(n, hv, kNumChunks / cpb);
#define STRIX_GDN_VERIFY(ST, VPK)                                                          \
  gdn_recoverssm_verify_kernel<ST, VPK><<<grid, kThreads, 0, stream>>>(                    \
      reinterpret_cast<const uint16_t*>(q.data_ptr()),                                     \
      reinterpret_cast<const uint16_t*>(k.data_ptr()),                                     \
      reinterpret_cast<const uint16_t*>(v.data_ptr()),                                     \
      reinterpret_cast<const uint16_t*>(a.data_ptr()),                                     \
      reinterpret_cast<const uint16_t*>(b.data_ptr()), a_log.data_ptr<float>(),            \
      dt_bias.data_ptr(), cu_seqlens.data_ptr<int>(), state_indices.data_ptr<int64_t>(),   \
      reinterpret_cast<const ST*>(state.data_ptr()), correction.data_ptr<float>(),         \
      keys.data_ptr<float>(), decays.data_ptr<float>(),                                    \
      reinterpret_cast<uint16_t*>(out.data_ptr()), dt_type, null_block,                    \
      static_cast<float>(scale), cpb, int(correction.size(2)), st)
#define STRIX_GDN_VERIFY_VPK(ST)                      \
  switch (vpk) {                                      \
    case 1: STRIX_GDN_VERIFY(ST, 1); break;           \
    case 2: STRIX_GDN_VERIFY(ST, 2); break;           \
    case 3: STRIX_GDN_VERIFY(ST, 3); break;           \
    case 4: STRIX_GDN_VERIFY(ST, 4); break;           \
    default: STRIX_GDN_VERIFY(ST, 8); break;          \
  }
  if (stt == ScalarType::Float) {
    STRIX_GDN_VERIFY_VPK(float)
  } else {
    STRIX_GDN_VERIFY_VPK(uint16_t)
  }
#undef STRIX_GDN_VERIFY_VPK
#undef STRIX_GDN_VERIFY
}

}  // namespace strix
