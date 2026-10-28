// SPDX-License-Identifier: MIT
// GDN prefill front end (Qwen3.x): the causal depthwise conv over mixed_qkv
// and the post-conv split into q/k/v with L2 norm and gating.
//
// conv: out[t, f] = act(bias[f] + sum_j w[f, j] x[t - 3 + j, f]), width 4,
// varlen over cu_seqlens; positions before a sequence's first token come from
// its conv state row (when has_initial_state) or are zero. The state row then
// holds the sequence's last 3 inputs (state index 2 = newest).
//
// Feature blocks are the fastest grid index so concurrently running blocks
// read the same token rows (token-fastest grids camp on DRAM channels here).
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kWidth = 4, kState = kWidth - 1;
constexpr int kThreads = 256;
constexpr int kFeatPerBlock = kThreads * 8;
constexpr int kTokPerBlock = 32;
constexpr int kPadSlot = -1, kNullBlock = 0;

__device__ __forceinline__ void unpack8(uint4 v, float* f) {
  const uint32_t p[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) f[2 * i] = bf16_lo(p[i]), f[2 * i + 1] = bf16_hi(p[i]);
}
__device__ __forceinline__ uint4 pack8(const float* f) {
  return make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]),
                    pack_bf16x2(f[4], f[5]), pack_bf16x2(f[6], f[7]));
}

struct ConvState {
  uint16_t* p;
  int64_t s_seq, s_dim, s_tok;
  const int32_t* cache_idx;
  const bool* has_init;

  // Cache line of sequence n, or -1 when it has no usable initial state.
  __device__ int64_t init_line(int n) const {
    const int ci = cache_idx[n];
    if (ci == kPadSlot || ci == kNullBlock) return -1;
    return has_init != nullptr && has_init[n] ? ci : -1;
  }
  __device__ void load8(int64_t line, int idx, int f0, float* f) const {
    const uint16_t* q = p + line * s_seq + idx * s_tok + f0 * s_dim;
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = __uint_as_float(uint32_t(q[i * s_dim]) << 16);
  }
};

// Sequence owning token t: the count of cu[1..nseq-1] <= t, by bisection.
__device__ __forceinline__ int seq_of(const int32_t* cu, int nseq, int t) {
  int lo = 0, hi = nseq - 1;  // answer in [lo, hi]
  for (int it = 0; it < 32 && lo < hi; ++it) {
    const int mid = (lo + hi + 1) / 2;
    if (cu[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

template <bool kSilu>
__global__ __launch_bounds__(kThreads) void conv_fwd_kernel(
    uint16_t* __restrict__ out, int64_t ldo, const uint16_t* __restrict__ x,
    int64_t ldx, const uint16_t* __restrict__ w, const uint16_t* __restrict__ bias,
    ConvState st, const int32_t* __restrict__ cu, int nseq, int ntok, int dim,
    int nfb) {
  const int fb = blockIdx.x % nfb, tb = blockIdx.x / nfb;
  const int f0 = (fb * kThreads + threadIdx.x) * 8;
  if (f0 >= dim) return;
  const int t_begin = tb * kTokPerBlock, t_end = min(ntok, t_begin + kTokPerBlock);

  float wf[kWidth][8], bf[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    // w is [dim, 4]: two 16-byte loads cover 8 features x 4 taps.
    const uint2 wv = *reinterpret_cast<const uint2*>(w + (f0 + i) * kWidth);
    wf[0][i] = bf16_lo(wv.x), wf[1][i] = bf16_hi(wv.x);
    wf[2][i] = bf16_lo(wv.y), wf[3][i] = bf16_hi(wv.y);
    bf[i] = bias != nullptr ? __uint_as_float(uint32_t(bias[f0 + i]) << 16) : 0.f;
  }

  float h[kState][8];  // the 3 inputs before the current token, oldest first
  int n = seq_of(cu, nseq, t_begin);
  int seq_end = cu[n + 1];
  auto load_window = [&](int t) {
    const int s = cu[n];
    const int64_t line = t - s < kState ? st.init_line(n) : -1;
#pragma unroll
    for (int j = 0; j < kState; ++j) {
      const int tt = t - kState + j;
      if (tt >= s) {
        unpack8(*reinterpret_cast<const uint4*>(x + int64_t(tt) * ldx + f0), h[j]);
      } else if (line >= 0) {
        st.load8(line, kState + (tt - s), f0, h[j]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) h[j][i] = 0.f;
      }
    }
  };
  load_window(t_begin);

  for (int t = t_begin; t < t_end; ++t) {
    if (t == seq_end) {  // empty sequences have no tokens to land on
      n = seq_of(cu, nseq, t);
      seq_end = cu[n + 1];
      load_window(t);
    }
    float xv[8], o[8];
    unpack8(*reinterpret_cast<const uint4*>(x + int64_t(t) * ldx + f0), xv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float acc = bf[i] + wf[0][i] * h[0][i] + wf[1][i] * h[1][i] +
                  wf[2][i] * h[2][i] + wf[3][i] * xv[i];
      if constexpr (kSilu) acc = acc / (1.f + __expf(-acc));
      o[i] = acc;
      h[0][i] = h[1][i], h[1][i] = h[2][i], h[2][i] = xv[i];
    }
    *reinterpret_cast<uint4*>(out + int64_t(t) * ldo + f0) = pack8(o);
  }
}

// Runs after the conv so no block reads a state row that is being replaced.
__global__ __launch_bounds__(kThreads) void conv_state_kernel(
    const uint16_t* __restrict__ x, int64_t ldx, ConvState st,
    const int32_t* __restrict__ cu, int dim) {
  const int n = blockIdx.y;
  const int f0 = (blockIdx.x * kThreads + threadIdx.x) * 8;
  const int ci = st.cache_idx[n];
  if (f0 >= dim || ci == kPadSlot || ci == kNullBlock) return;
  const int s = cu[n], len = cu[n + 1] - s;
  const int64_t line = st.init_line(n);
  float nv[kState][8];
#pragma unroll
  for (int j = 0; j < kState; ++j) {
    const int tt = len - kState + j;
    if (tt >= 0) {
      unpack8(*reinterpret_cast<const uint4*>(x + int64_t(s + tt) * ldx + f0), nv[j]);
    } else if (line >= 0) {
      st.load8(line, j + len, f0, nv[j]);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) nv[j][i] = 0.f;
    }
  }
#pragma unroll
  for (int j = 0; j < kState; ++j) {
    uint16_t* q = st.p + int64_t(ci) * st.s_seq + j * st.s_tok + f0 * st.s_dim;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t b = pack_bf16x2(nv[j][i], 0.f);
      q[i * st.s_dim] = static_cast<uint16_t>(b & 0xffffu);
    }
  }
}

// One block per token row of the conv output: q/k heads (L2-normalized,
// 16 lanes x 8 channels per head), v heads (copied), then g and beta.
__global__ __launch_bounds__(kThreads) void post_conv_kernel(
    uint16_t* __restrict__ q, uint16_t* __restrict__ k, uint16_t* __restrict__ v,
    float* __restrict__ g, float* __restrict__ beta,
    const uint16_t* __restrict__ conv, int64_t ldc,
    const uint16_t* __restrict__ a, int64_t lda, const uint16_t* __restrict__ b,
    int64_t ldb, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    int hk, int hv, float eps) {
  constexpr int kD = 128;
  const int64_t t = blockIdx.x;
  const int tid = threadIdx.x;
  const int qk_dim = hk * kD, chunks = (2 * qk_dim + hv * kD) / 8;
  const uint16_t* row = conv + t * ldc;
  // Groups of 16 lanes stay inside one head: heads are 16 chunks wide and
  // kThreads is a multiple of 16.
  for (int c = tid; c < chunks; c += kThreads) {
    const int e = c * 8;
    const uint4 raw = *reinterpret_cast<const uint4*>(row + e);
    if (e < 2 * qk_dim) {
      float f[8], ss = 0.f;
      unpack8(raw, f);
#pragma unroll
      for (int i = 0; i < 8; ++i) ss += f[i] * f[i];
      const float inv = 1.f / sqrtf(group_sum<16>(ss) + eps);
#pragma unroll
      for (int i = 0; i < 8; ++i) f[i] *= inv;
      uint16_t* dst = e < qk_dim ? q + t * qk_dim + e : k + t * qk_dim + (e - qk_dim);
      *reinterpret_cast<uint4*>(dst) = pack8(f);
    } else {
      *reinterpret_cast<uint4*>(v + t * hv * kD + (e - 2 * qk_dim)) = raw;
    }
  }
  for (int h = tid; h < hv; h += kThreads) {
    const float xa = __uint_as_float(uint32_t(a[t * lda + h]) << 16) + dt_bias[h];
    const float sp = xa > 20.f ? xa : log1pf(__expf(xa));
    g[t * hv + h] = -__expf(a_log[h]) * sp;
    const float xb = __uint_as_float(uint32_t(b[t * ldb + h]) << 16);
    beta[t * hv + h] = 1.f / (1.f + __expf(-xb));
  }
}

}  // namespace

// x, out: [T, dim] bf16 with unit feature stride (row strides free);
// weight: [dim, 4] contiguous bf16; bias: [dim] bf16 or none;
// conv_states: [lines, dim, 3] bf16, any strides, updated in place;
// cu_seqlens: [N + 1] int32; cache_indices: [N] int32 (-1 / 0 = no state
// line); has_initial_state: [N] bool or none.
void gdn_causal_conv1d_w4_bf16(torch::Tensor& out, const torch::Tensor& x,
                               const torch::Tensor& weight,
                               const std::optional<torch::Tensor>& bias,
                               torch::Tensor& conv_states,
                               const torch::Tensor& cu_seqlens,
                               const torch::Tensor& cache_indices,
                               const std::optional<torch::Tensor>& has_initial_state,
                               bool silu) {
  const int64_t ntok = x.size(0), dim = x.size(1), nseq = cu_seqlens.numel() - 1;
  for (const torch::Tensor* t : {&x, static_cast<const torch::Tensor*>(&out)})
    TORCH_CHECK(t->dim() == 2 && t->scalar_type() == at::kBFloat16 &&
                    t->stride(1) == 1 && t->stride(0) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "x/out must be 16-byte aligned bf16 [T, dim] rows");
  TORCH_CHECK(out.sizes() == x.sizes() && dim % 8 == 0, "out/x shape mismatch");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16 && weight.is_contiguous() &&
                  weight.dim() == 2 && weight.size(0) == dim &&
                  weight.size(1) == kWidth,
              "weight must be contiguous bf16 [dim, 4]");
  TORCH_CHECK(conv_states.scalar_type() == at::kBFloat16 && conv_states.dim() == 3 &&
                  conv_states.size(1) == dim && conv_states.size(2) >= kState,
              "conv_states must be bf16 [lines, dim, >=3]");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt &&
                  cache_indices.scalar_type() == at::kInt &&
                  cache_indices.is_contiguous() && cache_indices.numel() >= nseq,
              "cu_seqlens / cache_indices must be int32");
  if (bias)
    TORCH_CHECK(bias->scalar_type() == at::kBFloat16 && bias->is_contiguous() &&
                    bias->numel() == dim,
                "bias must be contiguous bf16 [dim]");
  if (has_initial_state)
    TORCH_CHECK(has_initial_state->scalar_type() == at::kBool &&
                    has_initial_state->is_contiguous(),
                "has_initial_state must be contiguous bool");
  if (ntok == 0 || nseq == 0) return;

  ConvState st{reinterpret_cast<uint16_t*>(conv_states.data_ptr()),
               conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
               cache_indices.data_ptr<int32_t>(),
               has_initial_state ? has_initial_state->data_ptr<bool>() : nullptr};
  const auto* xp = reinterpret_cast<const uint16_t*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const uint16_t*>(weight.data_ptr());
  const auto* bp = bias ? reinterpret_cast<const uint16_t*>(bias->data_ptr()) : nullptr;
  const auto* cu = cu_seqlens.data_ptr<int32_t>();
  const int nfb = (dim + kFeatPerBlock - 1) / kFeatPerBlock;
  const int ntb = (ntok + kTokPerBlock - 1) / kTokPerBlock;
  auto stream = at::hip::getCurrentHIPStream();
  auto kernel = silu ? conv_fwd_kernel<true> : conv_fwd_kernel<false>;
  kernel<<<nfb * ntb, kThreads, 0, stream>>>(
      reinterpret_cast<uint16_t*>(out.data_ptr()), out.stride(0), xp, x.stride(0),
      wp, bp, st, cu, nseq, ntok, dim, nfb);
  conv_state_kernel<<<dim3(nfb, nseq), kThreads, 0, stream>>>(xp, x.stride(0), st,
                                                                cu, dim);
}

// conv: [T, 2 * hk * 128 + hv * 128] bf16 (unit feature stride); a, b:
// [T, hv] bf16; a_log, dt_bias: [hv] fp32. Outputs q, k: [T, hk, 128],
// v: [T, hv, 128] (contiguous bf16); g, beta: [T, hv] fp32.
void gdn_post_conv_bf16(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
                        torch::Tensor& g, torch::Tensor& beta,
                        const torch::Tensor& conv, const torch::Tensor& a,
                        const torch::Tensor& b, const torch::Tensor& a_log,
                        const torch::Tensor& dt_bias, double eps) {
  const int64_t ntok = conv.size(0), hk = q.size(1), hv = v.size(1);
  TORCH_CHECK(conv.scalar_type() == at::kBFloat16 && conv.dim() == 2 &&
                  conv.stride(1) == 1 && conv.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(conv.data_ptr()) % 16 == 0 &&
                  conv.size(1) == (2 * hk + hv) * 128,
              "conv must be 16-byte aligned bf16 [T, (2 hk + hv) * 128]");
  for (const auto* t : {&q, &k, &v})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->is_contiguous() &&
                    t->size(0) == ntok && t->size(2) == 128,
                "q/k/v must be contiguous bf16 [T, H, 128]");
  for (const auto* t : {&a, &b})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->dim() == 2 &&
                    t->stride(1) == 1 && t->size(0) == ntok && t->size(1) == hv,
                "a/b must be bf16 [T, hv] with unit head stride");
  for (const torch::Tensor* t : {static_cast<const torch::Tensor*>(&g), static_cast<const torch::Tensor*>(&beta), &a_log, &dt_bias})
    TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous(),
                "g/beta/a_log/dt_bias must be contiguous fp32");
  if (ntok == 0) return;
  post_conv_kernel<<<ntok, kThreads, 0, at::hip::getCurrentHIPStream()>>>(
      reinterpret_cast<uint16_t*>(q.data_ptr()),
      reinterpret_cast<uint16_t*>(k.data_ptr()),
      reinterpret_cast<uint16_t*>(v.data_ptr()), g.data_ptr<float>(),
      beta.data_ptr<float>(), reinterpret_cast<const uint16_t*>(conv.data_ptr()),
      conv.stride(0), reinterpret_cast<const uint16_t*>(a.data_ptr()), a.stride(0),
      reinterpret_cast<const uint16_t*>(b.data_ptr()), b.stride(0),
      a_log.data_ptr<float>(), dt_bias.data_ptr<float>(), hk, hv,
      static_cast<float>(eps));
}

}  // namespace strix
