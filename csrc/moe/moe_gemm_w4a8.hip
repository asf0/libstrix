// SPDX-License-Identifier: MIT
// Grouped int4-weight / int8-activation (W4A8) expert GEMM over vLLM's
// moe_align_block_size layout, on the iu8 WMMA:
//   C[sid, n] = sum_g sa[arow, g] * sw[e, n, g] * sum_{k in g} qa[arow, k] * (w[e, n, k] - 8)
// with arow = sid / top_k, e = expert(sid), g = one 128-element K group.
//
// Same block ownership, fused {silu,gelu_tanh}_and_mul epilogue and topk
// weighting as moe_gemm_w4a16. What changes is the operand path: activations
// are pre-quantized to int8 per (row, 128-element group) by moe_act_quant_i8,
// and the int4 weights become int8 at the LDS commit with two byte permutes
// and an add/xor per 4 values (no float convert, multiply or bf16 rounding per
// weight, which is what bounded moe_gemm_w4a16 at ~26 TF). Each 128-wide K
// step is exactly one quantization group, so the i32 WMMA accumulator is
// folded into fp32 at the end of every step with sa * sw.
//
// W is vLLM's Triton WNA16 layout as uint8 [E, N, K/2]: byte j holds k = 2j
// in its low nibble and k = 2j + 1 in its high nibble; the stored value is
// q + 8 (symmetric uint4b8). Scales are [E, N, K/128] bf16 or fp16.
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kGK = 128;  // K per main-loop step: one quantization group
constexpr int kKT = kGK / 16;  // 16-wide WMMA k-tiles per step

typedef int32_t v4i32 __attribute__((ext_vector_type(4)));
typedef int32_t v8i32 __attribute__((ext_vector_type(8)));

enum Act8 : int { kA8None = 0, kA8Silu = 1, kA8GeluTanh = 2 };

template <int ACT>
__device__ __forceinline__ float moe8_gate(float g) {
  if constexpr (ACT == kA8Silu) {
    return g / (1.f + __expf(-g));
  } else {
    const float z = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    return 0.5f * g * (2.f - 2.f / (__expf(2.f * z) + 1.f));
  }
}

// 8 packed uint4b8 values (plain nibble order, k0 in the low nibble) -> 8
// signed int8 bytes k0..k7 as two words.
__device__ __forceinline__ void w4_to_i8(uint32_t w, uint32_t& o0, uint32_t& o1) {
  const uint32_t lo = w & 0x0F0F0F0Fu;          // k0 k2 k4 k6
  const uint32_t hn = (w >> 4) & 0x0F0F0F0Fu;   // k1 k3 k5 k7
  // perm(hn, lo, sel): selector bytes 0-3 pick lo, 4-7 pick hn.
  // (n + 0x78) ^ 0x80 = n - 8 as a signed byte, no carry between bytes.
  o0 = (__builtin_amdgcn_perm(hn, lo, 0x05010400u) + 0x78787878u) ^ 0x80808080u;
  o1 = (__builtin_amdgcn_perm(hn, lo, 0x07030602u) + 0x78787878u) ^ 0x80808080u;
}

template <int BM, int BN, int WAVES_M, int WAVES_N, int ACT>
__global__ __launch_bounds__(WAVES_M* WAVES_N* kWave) void moe_gemm_w4a8_kernel(
    uint16_t* __restrict__ c, const int8_t* __restrict__ qa,
    const float* __restrict__ sa, const uint8_t* __restrict__ w,
    const uint16_t* __restrict__ scales, const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_post_padded, const float* __restrict__ topk_w,
    int64_t num_valid, int top_k, int k_dim, int s_f16, int64_t ldw,
    int64_t lds_n, int64_t w_expert_stride, int64_t s_expert_stride,
    int64_t ldc, int n_half) {
  constexpr int kThreads = WAVES_M * WAVES_N * kWave;
  constexpr int TM = BM / WAVES_M, TN = BN / WAVES_N;
  constexpr int FM = TM / 16, FN = TN / 16;
  constexpr int kChunksA = BM * kKT;  // 16-byte int8 pieces of the A step
  constexpr int kChunksB = BN * kKT;  // 8-byte int4 pieces of the W step
  constexpr int kLoadsA = (kChunksA + kThreads - 1) / kThreads;
  constexpr int kLoadsB = (kChunksB + kThreads - 1) / kThreads;
  static_assert(FM >= 1 && FN >= 1 && TM % 16 == 0 && TN % 16 == 0);
  static_assert(ACT == kA8None ||
                    (WAVES_N % 2 == 0 && BM * (BN / 2) * 4 <= (BM + BN) * kGK),
                "the fused and-mul epilogue needs an even wave split and LDS "
                "for the up half");
  // [k-tile][row] 16-byte fragments: a lane's 16 k values of one row are one
  // 16-byte LDS read, and the staging stores of consecutive rows are
  // consecutive. The epilogue reuses the buffer for the up half.
  __shared__ __attribute__((aligned(16))) int8_t s_buf[(BM + BN) * kGK];
  int8_t* s_a = s_buf;
  int8_t* s_b = s_buf + BM * kGK;
  __shared__ int32_t s_sid[BM];
  __shared__ int64_t s_qrow[BM];
  __shared__ float s_sa[BM];

  const int mb = blockIdx.y;
  if (mb * BM >= *num_post_padded) return;
  const int expert = expert_ids[mb];
  if (expert < 0) return;
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int wm = wave % WAVES_M, wn = wave / WAVES_M;
  const int groups = k_dim / kGK;

  for (int i = tid; i < BM; i += kThreads) {
    const int32_t sid = sorted_ids[mb * BM + i];
    const bool ok = sid < num_valid;
    s_sid[i] = ok ? sid : -1;
    s_qrow[i] = ok ? int64_t(sid / top_k) : -1;
  }
  __syncthreads();

  const int64_t w_base = int64_t(expert) * w_expert_stride;
  const int64_t s_base = int64_t(expert) * s_expert_stride;
  // Tile row brow -> W row (fused and-mul: gate rows, then the matching up rows).
  auto w_row = [&](int brow) -> int64_t {
    const int64_t base = ACT != kA8None ? n0 / 2 : n0;
    const bool up = ACT != kA8None && brow >= BN / 2;
    return base + (up ? int64_t(n_half - BN / 2) : 0) + brow;
  };

  v4i32 ra[kLoadsA];
  uint2 rb[kLoadsB];
  float rsa = 0.f;
  auto load = [&](int g) {
    const int k0 = g * kGK;
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads;
      v4i32 v = {0, 0, 0, 0};
      if (ch < kChunksA) {
        const int row = ch % BM, t = ch / BM;
        const int64_t q = s_qrow[row];
        if (q >= 0)
          v = *reinterpret_cast<const v4i32*>(qa + q * k_dim + k0 + 16 * t);
      }
      ra[i] = v;
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads;
      uint2 v = {0, 0};
      if (ch < kChunksB) {
        const int row = ch % BN, t = ch / BN;
        v = *reinterpret_cast<const uint2*>(w + w_base + w_row(row) * ldw +
                                            (k0 + 16 * t) / 2);
      }
      rb[i] = v;
    }
    if (tid < BM) {
      const int64_t q = s_qrow[tid];
      rsa = q >= 0 ? sa[q * groups + g] : 0.f;
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads;
      if (ch < kChunksA)
        *reinterpret_cast<v4i32*>(s_a + ((ch / BM) * BM + ch % BM) * 16) = ra[i];
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads;
      if (ch < kChunksB) {
        uint32_t o0, o1, o2, o3;
        w4_to_i8(rb[i].x, o0, o1);
        w4_to_i8(rb[i].y, o2, o3);
        *reinterpret_cast<uint4*>(s_b + ((ch / BN) * BN + ch % BN) * 16) =
            make_uint4(o0, o1, o2, o3);
      }
    }
    if (tid < BM) s_sa[tid] = rsa;
  };
  auto load_sw = [&](int brow, int g) -> float {
    const uint16_t sraw = scales[s_base + w_row(brow) * lds_n + g];
    return s_f16 ? __half2float(*reinterpret_cast<const __half*>(&sraw))
                 : __uint_as_float(uint32_t(sraw) << 16);
  };

  f32x8 acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) acc[i][j] = f32x8{};
  const v8i32 kZero = {0, 0, 0, 0, 0, 0, 0, 0};

  load(0);
  store();
  __syncthreads();
  for (int g = 0; g < groups; ++g) {
    const bool more = g + 1 < groups;
    if (more) load(g + 1);
    float sw[FN];
#pragma unroll
    for (int j = 0; j < FN; ++j) sw[j] = load_sw(wn * TN + j * 16 + acc_col(lane), g);
    v8i32 acci[FM][FN];
#pragma unroll
    for (int t = 0; t < kKT; ++t) {
      v4i32 fa[FM];
#pragma unroll
      for (int i = 0; i < FM; ++i)
        fa[i] = *reinterpret_cast<const v4i32*>(
            s_a + (t * BM + wm * TM + i * 16 + lane % 16) * 16);
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        const v4i32 fb = *reinterpret_cast<const v4i32*>(
            s_b + (t * BN + wn * TN + j * 16 + lane % 16) * 16);
#pragma unroll
        for (int i = 0; i < FM; ++i)
          acci[i][j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
              true, fa[i], true, fb, t == 0 ? kZero : acci[i][j], false);
      }
    }
    // Fold the group: rows' activation scale times the column's weight scale.
#pragma unroll
    for (int i = 0; i < FM; ++i) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float s_row = s_sa[wm * TM + i * 16 + acc_row(e, lane)];
#pragma unroll
        for (int j = 0; j < FN; ++j)
          acc[i][j][e] += float(acci[i][j][e]) * (s_row * sw[j]);
      }
    }
    __syncthreads();
    if (more) store();
    __syncthreads();
  }

  // acc[i][j][e] = C[token row wm*TM + i*16 + acc_row(e)][n wn*TN + j*16 + col]
  if constexpr (ACT != kA8None) {
    constexpr int kHalf = BN / 2;
    float* s_up = reinterpret_cast<float*>(s_buf);  // [BM][BN/2]
    const bool up = wn >= WAVES_N / 2;
    const int col0 = (up ? wn - WAVES_N / 2 : wn) * TN;
    if (up) {
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e)
#pragma unroll
          for (int j = 0; j < FN; ++j)
            s_up[(wm * TM + i * 16 + acc_row(e, lane)) * kHalf + col0 + j * 16 +
                 acc_col(lane)] = acc[i][j][e];
    }
    __syncthreads();
    if (!up) {
#pragma unroll
      for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int r = wm * TM + i * 16 + acc_row(e, lane);
          const int32_t sid = s_sid[r];
          if (sid < 0) continue;
          uint16_t* crow = c + int64_t(sid) * ldc + n0 / 2 + col0 + acc_col(lane);
#pragma unroll
          for (int j = 0; j < FN; ++j) {
            const float gt = acc[i][j][e];
            const float u = s_up[r * kHalf + col0 + j * 16 + acc_col(lane)];
            const uint32_t packed = pack_bf16x2(moe8_gate<ACT>(gt) * u, 0.f);
            crow[j * 16] = static_cast<uint16_t>(packed & 0xffffu);
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < FM; ++i) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int32_t sid = s_sid[wm * TM + i * 16 + acc_row(e, lane)];
        if (sid < 0) continue;
        const float scale = topk_w ? topk_w[sid] : 1.f;
        uint16_t* crow = c + int64_t(sid) * ldc + n0 + wn * TN + acc_col(lane);
#pragma unroll
        for (int j = 0; j < FN; ++j) {
          const uint32_t packed = pack_bf16x2(acc[i][j][e] * scale, 0.f);
          crow[j * 16] = static_cast<uint16_t>(packed & 0xffffu);
        }
      }
    }
  }
}

// Per (row, 128-element group): s = max|x| / 127 (1 for an all-zero group),
// q = clamp(rint(x / s), -127, 127). One 16-lane team per group, 8 values a
// lane. Same convention as w8a8_wmma_gemm's act_quant_kernel, plain layout.
template <int T>
__global__ __launch_bounds__(T) void moe_act_quant_i8_kernel(
    const uint16_t* __restrict__ x, int64_t ldx, int8_t* __restrict__ q,
    float* __restrict__ sa, int64_t rows, int k_dim) {
  const int groups = k_dim / kGK;
  const int64_t gi = int64_t(blockIdx.x) * (T / 16) + threadIdx.x / 16;
  if (gi >= rows * groups) return;  // whole 16-lane teams leave together
  const int64_t row = gi / groups;
  const int g = int(gi % groups), l = threadIdx.x % 16;
  const uint4 raw =
      *reinterpret_cast<const uint4*>(x + row * ldx + g * kGK + l * 8);
  const uint32_t wd[4] = {raw.x, raw.y, raw.z, raw.w};
  float v[8];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[2 * i] = __uint_as_float(wd[i] << 16);
    v[2 * i + 1] = __uint_as_float(wd[i] & 0xffff0000u);
    amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
  }
#pragma unroll
  for (int off = 8; off >= 1; off >>= 1)
    amax = fmaxf(amax, __shfl_xor(amax, off, 16));
  const float s = amax > 0.f ? amax / 127.f : 1.f;
  const float inv = 1.f / s;
  uint32_t packed[2] = {0, 0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int qi = max(-127, min(127, __float2int_rn(v[i] * inv)));
    packed[i / 4] |= uint32_t(qi & 0xff) << (8 * (i % 4));
  }
  *reinterpret_cast<uint2*>(q + row * k_dim + g * kGK + l * 8) =
      make_uint2(packed[0], packed[1]);
  if (l == 0) sa[row * groups + g] = s;
}

struct MoeW4A8Launch {
  uint16_t* c;
  const int8_t* qa;
  const float* sa;
  const uint8_t* w;
  const uint16_t* scales;
  const int32_t* sorted_ids;
  const int32_t* expert_ids;
  const int32_t* num_post_padded;
  const float* topk_w;
  int64_t num_valid;
  int top_k, k_dim, s_f16;
  int64_t ldw, lds_n, w_expert_stride, s_expert_stride, ldc;
  int n_half;
  dim3 grid;
  hipStream_t stream;
};

template <int BM, int BN, int WM, int WN, int ACT>
inline void launch_moe_w4a8(const MoeW4A8Launch& m) {
  moe_gemm_w4a8_kernel<BM, BN, WM, WN, ACT>
      <<<m.grid, WM * WN * kWave, 0, m.stream>>>(
          m.c, m.qa, m.sa, m.w, m.scales, m.sorted_ids, m.expert_ids,
          m.num_post_padded, m.topk_w, m.num_valid, m.top_k, m.k_dim, m.s_f16,
          m.ldw, m.lds_n, m.w_expert_stride, m.s_expert_stride, m.ldc, m.n_half);
}

}  // namespace

// x: [rows, K] bf16 (unit last stride); q: [rows, K] int8; sa: [rows, K/128] fp32.
void moe_act_quant_i8(const torch::Tensor& x, torch::Tensor& q, torch::Tensor& sa) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && q.scalar_type() == at::kChar &&
                  sa.scalar_type() == at::kFloat,
              "moe_act_quant_i8: x bf16, q int8, sa fp32");
  TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1 && x.stride(0) % 8 == 0,
              "x must be 2-D with 16-byte aligned rows");
  const int64_t rows = x.size(0), k = x.size(1);
  TORCH_CHECK(k % kGK == 0, "K must be a multiple of ", kGK);
  TORCH_CHECK(q.is_contiguous() && q.size(0) == rows && q.size(1) == k,
              "q must be contiguous [rows, K]");
  TORCH_CHECK(sa.is_contiguous() && sa.numel() == rows * (k / kGK),
              "sa must be contiguous [rows, K/128]");
  constexpr int T = 256;
  const int64_t teams = rows * (k / kGK);
  if (teams == 0) return;
  moe_act_quant_i8_kernel<T>
      <<<dim3((teams + T / 16 - 1) / (T / 16)), T, 0,
         at::hip::getCurrentHIPStream()>>>(
          reinterpret_cast<const uint16_t*>(x.data_ptr()), x.stride(0),
          q.data_ptr<int8_t>(), sa.data_ptr<float>(), rows, int(k));
}

// qa: [rows, K] int8 and sa: [rows, K/128] from moe_act_quant_i8 (row =
// sid / top_k); w: uint8 [E, N, K/2] (Triton WNA16 layout); scales:
// [E, N, K/128] bf16 or fp16; c: [num_valid, N], or [num_valid, N/2] with
// silu_mul. block_m / block_n: 64/256, 128/128 or 64/128.
void moe_gemm_w4a8(torch::Tensor& c, const torch::Tensor& qa,
                   const torch::Tensor& sa, const torch::Tensor& w,
                   const torch::Tensor& scales, const torch::Tensor& sorted_ids,
                   const torch::Tensor& expert_ids,
                   const torch::Tensor& num_post_padded,
                   const std::optional<torch::Tensor>& topk_weights,
                   int64_t top_k, int64_t block_m, int64_t block_n,
                   bool silu_mul, c10::string_view activation) {
  const bool s_f16 = scales.scalar_type() == at::kHalf;
  TORCH_CHECK(qa.scalar_type() == at::kChar && sa.scalar_type() == at::kFloat &&
                  w.scalar_type() == at::kByte &&
                  (scales.scalar_type() == at::kBFloat16 || s_f16) &&
                  c.scalar_type() == at::kBFloat16,
              "moe_gemm_w4a8: qa int8, sa fp32, w uint8, scales bf16/fp16, c bf16");
  TORCH_CHECK(qa.dim() == 2 && w.dim() == 3 && scales.dim() == 3 && c.dim() == 2,
              "bad ranks");
  TORCH_CHECK(qa.is_contiguous() && sa.is_contiguous() && w.stride(2) == 1 &&
                  scales.stride(2) == 1 && c.stride(1) == 1,
              "qa and sa contiguous, last dims contiguous");
  const int64_t n = w.size(1), k = w.size(2) * 2;
  TORCH_CHECK(qa.size(1) == k && sa.numel() == qa.size(0) * (k / kGK),
              "qa / sa shape mismatch");
  TORCH_CHECK(c.size(1) == (silu_mul ? n / 2 : n), "c shape mismatch");
  TORCH_CHECK(!silu_mul || (n % 2 == 0 && !topk_weights),
              "silu_mul needs even N and no topk weights");
  TORCH_CHECK(activation == "silu" || activation == "gelu_tanh",
              "activation must be silu or gelu_tanh, got ", activation);
  TORCH_CHECK(k % kGK == 0 && n % block_n == 0, "K % ", kGK, " and N % block_n");
  TORCH_CHECK(scales.size(1) == n && scales.size(2) == k / kGK,
              "scales must be [E, N, K / 128]");
  TORCH_CHECK(w.stride(1) % 8 == 0 && w.stride(0) % 8 == 0,
              "8-byte aligned weight rows required");
  TORCH_CHECK(sorted_ids.scalar_type() == at::kInt &&
                  expert_ids.scalar_type() == at::kInt &&
                  num_post_padded.scalar_type() == at::kInt,
              "routing tensors must be int32");
  TORCH_CHECK(expert_ids.numel() * block_m >= sorted_ids.numel(),
              "expert_ids too short for block_m ", block_m);
  const float* tw = nullptr;
  if (topk_weights) {
    TORCH_CHECK(topk_weights->scalar_type() == at::kFloat &&
                    topk_weights->is_contiguous() &&
                    topk_weights->numel() == c.size(0),
                "topk_weights must be contiguous fp32 [num_valid]");
    tw = topk_weights->data_ptr<float>();
  }
  const int act = !silu_mul ? kA8None
                  : activation == "gelu_tanh" ? kA8GeluTanh
                                              : kA8Silu;
  const MoeW4A8Launch m{
      reinterpret_cast<uint16_t*>(c.data_ptr()),
      qa.data_ptr<int8_t>(),
      sa.data_ptr<float>(),
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      reinterpret_cast<const uint16_t*>(scales.data_ptr()),
      sorted_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_post_padded.data_ptr<int32_t>(),
      tw,
      c.size(0),
      int(top_k),
      int(k),
      int(s_f16),
      w.stride(1),
      scales.stride(1),
      w.stride(0),
      scales.stride(0),
      c.stride(0),
      int(n / 2),
      dim3(n / block_n, expert_ids.numel()),
      at::hip::getCurrentHIPStream()};
#define STRIX_MOE_W4A8(BM, BN, WM, WN)                                        \
  do {                                                                        \
    if (act == kA8None) {                                                     \
      launch_moe_w4a8<BM, BN, WM, WN, kA8None>(m);                            \
    } else if (act == kA8Silu) {                                              \
      launch_moe_w4a8<BM, BN, WM, WN, kA8Silu>(m);                            \
    } else {                                                                  \
      launch_moe_w4a8<BM, BN, WM, WN, kA8GeluTanh>(m);                        \
    }                                                                         \
  } while (0)
  // 64x256 is 4 % (w13) / 9 % (w2) ahead of 128x128 at 16k Qwen3.8-Flash-Next
  // tokens, with uniform and with skewed (power-law) routing; 16-wave
  // 128x256 / 256x128 tiles were slower still.
  if (block_m == 64 && block_n == 256) STRIX_MOE_W4A8(64, 256, 2, 4);
  else if (block_m == 128 && block_n == 128) STRIX_MOE_W4A8(128, 128, 2, 4);
  else if (block_m == 64 && block_n == 128) STRIX_MOE_W4A8(64, 128, 2, 4);
  else TORCH_CHECK(false, "moe_gemm_w4a8: unsupported block_m/block_n ", block_m,
                   "/", block_n);
#undef STRIX_MOE_W4A8
}

}  // namespace strix
