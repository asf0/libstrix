// SPDX-License-Identifier: MIT
// Grouped int8 (W8A16) expert GEMM over vLLM's moe_align_block_size layout:
//   C[sid, n] = sum_k A[sid / top_k, k] * W[expert(sid), n, k]  (* w[sid])
// for every valid sorted id `sid`. Same block ownership as moe_gemm_bf16: one
// block owns BM sorted ids (one expert) and BN output columns.
//
// W is the compressed-tensors int8 layout that vLLM's Triton int8 MoE keeps
// as-is (convert_to_int8_moe_kernel_format is a no-op for the TRITON backend):
// signed int8 [E, N, K], one scale per output channel ([E, N] or [E, N, 1]).
// Weights are dequantized (int8 * scale) as they are staged into LDS, so the
// main loop is the same bf16 WMMA loop as moe_gemm_bf16/moe_gemm_w4a16 and the
// activation stays bf16 (W8A16: more accurate than the reference's per-token
// int8 activations, and half the weight bytes of the bf16 kernel).
//
// ACT fuses vLLM's {silu,gelu_tanh}_and_mul into the w13 GEMM exactly as the
// other two kernels do: W rows are [gate (N/2); up (N/2)], a block's BN rows
// are BN/2 gate rows and the matching BN/2 up rows, and c is [slots, N/2].
#include <ATen/hip/HIPContext.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kBK = 64;  // K per main-loop step; K must be a multiple of it
constexpr int kPad = 8;  // LDS row pitch kBK + kPad keeps 16-byte alignment
constexpr int kPitch = kBK + kPad;

enum Act : int { kActNone = 0, kActSilu = 1, kActGeluTanh = 2 };

template <int ACT>
__device__ __forceinline__ float moe_gate(float g) {
  if constexpr (ACT == kActSilu) {
    return g / (1.f + __expf(-g));
  } else {
    const float z = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    return 0.5f * g * (2.f - 2.f / (__expf(2.f * z) + 1.f));
  }
}

template <int BM, int BN, int WAVES_M, int WAVES_N, int ACT>
__global__ __launch_bounds__(WAVES_M* WAVES_N* kWave) void moe_gemm_w8a16_kernel(
    uint16_t* __restrict__ c, const uint16_t* __restrict__ a,
    const int8_t* __restrict__ w, const uint16_t* __restrict__ scales,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_post_padded, const float* __restrict__ topk_w,
    int64_t num_valid, int top_k, int k_dim, int s_f16, int s_f32, int64_t lda,
    int64_t ldw, int64_t w_expert_stride, int64_t s_expert_stride,
    int64_t ldc, int n_half) {
  constexpr int kThreads = WAVES_M * WAVES_N * kWave;
  constexpr int TM = BM / WAVES_M, TN = BN / WAVES_N;
  constexpr int FM = TM / 16, FN = TN / 16;
  constexpr int kChunksA = BM * kBK / 8;   // 16-byte bf16 chunks of A
  constexpr int kChunksB = BN * kBK / 16;  // 16-byte int8 chunks (16 k) of W
  constexpr int kLoadsA = (kChunksA + kThreads - 1) / kThreads;
  constexpr int kLoadsB = (kChunksB + kThreads - 1) / kThreads;
  static_assert(FM >= 1 && FN >= 1 && TM % 16 == 0 && TN % 16 == 0);

  static_assert(ACT == kActNone ||
                    (WAVES_N % 2 == 0 &&
                     BM * (BN / 2) * 4 <= (BM + BN) * kPitch * 2),
                "the fused and-mul epilogue needs an even wave split and LDS "
                "for the up half");
  // One buffer: s_a and s_b are carved from the front, and the fused epilogue
  // reuses the whole thing for the up half once the main loop is done.
  __shared__ __attribute__((aligned(16))) uint16_t s_buf[(BM + BN) * kPitch];
  uint16_t* s_a = s_buf;
  uint16_t* s_b = s_buf + BM * kPitch;
  __shared__ int32_t s_sid[BM];
  __shared__ int64_t s_arow[BM];

  const int mb = blockIdx.y;
  if (mb * BM >= *num_post_padded) return;
  const int expert = expert_ids[mb];
  if (expert < 0) return;
  const int n0 = blockIdx.x * BN;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int wm = wave % WAVES_M, wn = wave / WAVES_M;

  for (int i = tid; i < BM; i += kThreads) {
    const int32_t sid = sorted_ids[mb * BM + i];
    const bool ok = sid < num_valid;
    s_sid[i] = ok ? sid : -1;
    s_arow[i] = ok ? int64_t(sid / top_k) * lda : -1;
  }
  __syncthreads();

  const int64_t w_base = int64_t(expert) * w_expert_stride;
  const int64_t s_base = int64_t(expert) * s_expert_stride;

  typedef uint32_t stage_a_t __attribute__((ext_vector_type(4 * kLoadsA)));
  typedef uint32_t stage_b_t __attribute__((ext_vector_type(4 * kLoadsB)));
  stage_a_t ra = {};
  stage_b_t rb = {};
  float rs[kLoadsB];

  auto load = [&](int k0) {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads;
      uint4 v = {};
      if (ch < kChunksA) {
        const int64_t row = s_arow[ch / (kBK / 8)];
        if (row >= 0)
          v = *reinterpret_cast<const uint4*>(a + row + k0 +
                                              (ch % (kBK / 8)) * 8);
      }
      ra[4 * i] = v.x, ra[4 * i + 1] = v.y, ra[4 * i + 2] = v.z,
      ra[4 * i + 3] = v.w;
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads;
      uint4 v = {};
      float s = 0.f;
      if (ch < kChunksB) {
        const int brow = ch / (kBK / 16);
        const int k16 = (ch % (kBK / 16)) * 16;
        // Fused and-mul: tile rows [0, BN/2) are gate rows n0/2.., the rest
        // up rows N/2 + n0/2.. (n_half = N/2); otherwise BN consecutive rows.
        const int64_t base = ACT != kActNone ? n0 / 2 : n0;
        const bool up = ACT != kActNone && brow >= BN / 2;
        const int64_t row =
            base + (up ? int64_t(n_half - BN / 2) : 0) + brow;
        v = *reinterpret_cast<const uint4*>(w + w_base + row * ldw + (k0 + k16));
        // One scale per output channel: the whole row shares it.
        const int64_t s_off = s_base + row;
        if (s_f32) {
          s = reinterpret_cast<const float*>(scales)[s_off];
        } else {
          const uint16_t sraw = scales[s_off];
          s = s_f16 ? __half2float(*reinterpret_cast<const __half*>(&sraw))
                    : __uint_as_float(uint32_t(sraw) << 16);
        }
      }
      rb[4 * i] = v.x, rb[4 * i + 1] = v.y, rb[4 * i + 2] = v.z,
      rb[4 * i + 3] = v.w;
      rs[i] = s;
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads;
      if (ch < kChunksA)
        *reinterpret_cast<uint4*>(s_a + (ch / (kBK / 8)) * kPitch +
                                  (ch % (kBK / 8)) * 8) =
            make_uint4(ra[4 * i], ra[4 * i + 1], ra[4 * i + 2], ra[4 * i + 3]);
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads;
      if (ch < kChunksB) {
        const int brow = ch / (kBK / 16);
        const int k16 = (ch % (kBK / 16)) * 16;
        float f[16];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          // Little-endian bytes: byte t of word q is logical k = 4q + t.
          const uint32_t wd = rb[4 * i + q];
#pragma unroll
          for (int t = 0; t < 4; ++t)
            f[4 * q + t] = float(int8_t((wd >> (8 * t)) & 0xFFu)) * rs[i];
        }
        uint4 lo = make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]),
                              pack_bf16x2(f[4], f[5]), pack_bf16x2(f[6], f[7]));
        uint4 hi = make_uint4(pack_bf16x2(f[8], f[9]), pack_bf16x2(f[10], f[11]),
                              pack_bf16x2(f[12], f[13]), pack_bf16x2(f[14], f[15]));
        uint16_t* dst = s_b + brow * kPitch + k16;
        *reinterpret_cast<uint4*>(dst) = lo;
        *reinterpret_cast<uint4*>(dst + 8) = hi;
      }
    }
  };

  f32x8 acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) acc[i][j] = f32x8{};

  load(0);
  store();
  __syncthreads();
  for (int k0 = 0; k0 < k_dim; k0 += kBK) {
    const bool more = k0 + kBK < k_dim;
    if (more) load(k0 + kBK);
    const uint16_t* sa = s_a;
    const uint16_t* sb = s_b;
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      bf16x16 fb[FN];
#pragma unroll
      for (int j = 0; j < FN; ++j)
        fb[j] = load_frag_bf16(sb + (wn * TN + j * 16) * kPitch + kk, kPitch,
                               lane);
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        const bf16x16 fa =
            load_frag_bf16(sa + (wm * TM + i * 16) * kPitch + kk, kPitch, lane);
#pragma unroll
        for (int j = 0; j < FN; ++j) acc[i][j] = wmma_bf16(fa, fb[j], acc[i][j]);
      }
    }
    __syncthreads();
    if (more) store();
    __syncthreads();
  }

  // acc[i][j][e] = C[token row wm*TM + i*16 + acc_row(e)][n wn*TN + j*16 + col]
  if constexpr (ACT != kActNone) {
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
          uint16_t* crow =
              c + int64_t(sid) * ldc + n0 / 2 + col0 + acc_col(lane);
#pragma unroll
          for (int j = 0; j < FN; ++j) {
            const float g = acc[i][j][e];
            const float u = s_up[r * kHalf + col0 + j * 16 + acc_col(lane)];
            const uint32_t packed = pack_bf16x2(moe_gate<ACT>(g) * u, 0.f);
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

}  // namespace

namespace {

struct MoeW8Launch {
  uint16_t* c;
  const uint16_t* a;
  const int8_t* w;
  const uint16_t* scales;
  const int32_t* sorted_ids;
  const int32_t* expert_ids;
  const int32_t* num_post_padded;
  const float* topk_w;
  int64_t num_valid;
  int top_k, k_dim, s_f16, s_f32;
  int64_t lda, ldw, w_expert_stride, s_expert_stride, ldc;
  int n_half;
  dim3 grid;
  hipStream_t stream;
};

template <int BM, int BN, int WM, int WN, int ACT>
inline void launch_moe_w8(const MoeW8Launch& m) {
  moe_gemm_w8a16_kernel<BM, BN, WM, WN, ACT>
      <<<m.grid, WM * WN * kWave, 0, m.stream>>>(
          m.c, m.a, m.w, m.scales, m.sorted_ids, m.expert_ids,
          m.num_post_padded, m.topk_w, m.num_valid, m.top_k, m.k_dim, m.s_f16,
          m.s_f32, m.lda, m.ldw, m.w_expert_stride, m.s_expert_stride, m.ldc,
          m.n_half);
}

}  // namespace

// a: [rows, K] bf16 (row = sid / top_k); w: int8 [E, N, K] (row pitch in
// bytes); scales: bf16 or fp16 [E, N] / [E, N, 1]; c: [num_valid, N] view
// (row = sid) or [num_valid, N/2] with silu_mul. Same epilogue contract as
// moe_gemm_bf16: `activation` is "silu" or "gelu_tanh".
void moe_gemm_w8a16(torch::Tensor& c, const torch::Tensor& a,
                    const torch::Tensor& w, const torch::Tensor& scales,
                    const torch::Tensor& sorted_ids,
                    const torch::Tensor& expert_ids,
                    const torch::Tensor& num_post_padded,
                    const std::optional<torch::Tensor>& topk_weights,
                    int64_t top_k, int64_t block_m, int64_t block_n,
                    bool silu_mul, c10::string_view activation) {
  const bool s_f16 = scales.scalar_type() == at::kHalf;
  const bool s_f32 = scales.scalar_type() == at::kFloat;
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kChar &&
                  (scales.scalar_type() == at::kBFloat16 || s_f16 || s_f32) &&
                  c.scalar_type() == at::kBFloat16,
              "moe_gemm_w8a16: a, c bf16; scales bf16, fp16 or fp32; w int8");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 3 && c.dim() == 2, "bad ranks");
  TORCH_CHECK(scales.dim() == 2 || scales.dim() == 3, "scales must be [E, N]");
  const int64_t n = w.size(1), k = w.size(2);
  TORCH_CHECK(a.size(1) == k && c.size(1) == (silu_mul ? n / 2 : n),
              "shape mismatch");
  TORCH_CHECK(scales.size(0) == w.size(0) && scales.size(1) == n &&
                  (scales.dim() == 2 || scales.size(2) == 1),
              "scales must be [E, N] or [E, N, 1]");
  TORCH_CHECK(a.stride(1) == 1 && w.stride(2) == 1 && c.stride(1) == 1 &&
                  scales.stride(scales.dim() - 1) == 1,
              "last dims must be contiguous");
  TORCH_CHECK(!silu_mul || (n % 2 == 0 && !topk_weights),
              "silu_mul needs even N and no topk weights");
  TORCH_CHECK(activation == "silu" || activation == "gelu_tanh",
              "activation must be silu or gelu_tanh, got ", activation);
  TORCH_CHECK(k % kBK == 0 && n % block_n == 0, "K % ", kBK,
              " and N % block_n");
  TORCH_CHECK(a.stride(0) % 8 == 0 && w.stride(1) % 8 == 0 &&
                  w.stride(0) % 8 == 0,
              "16-byte aligned rows required");
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

  const dim3 grid(n / block_n, expert_ids.numel());
  auto stream = at::hip::getCurrentHIPStream();
  const int act = !silu_mul ? kActNone
                  : activation == "gelu_tanh" ? kActGeluTanh
                                              : kActSilu;
  const MoeW8Launch m{
      reinterpret_cast<uint16_t*>(c.data_ptr()),
      reinterpret_cast<const uint16_t*>(a.data_ptr()),
      reinterpret_cast<const int8_t*>(w.data_ptr()),
      reinterpret_cast<const uint16_t*>(scales.data_ptr()),
      sorted_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_post_padded.data_ptr<int32_t>(),
      tw,
      c.size(0),
      int(top_k),
      int(k),
      int(s_f16),
      int(s_f32),
      a.stride(0),
      w.stride(1),
      w.stride(0),
      scales.stride(0),
      c.stride(0),
      int(n / 2),
      grid,
      stream};
#define STRIX_MOE_W8(BM, BN, WM, WN)                                          \
  do {                                                                        \
    if (act == kActNone) {                                                    \
      launch_moe_w8<BM, BN, WM, WN, kActNone>(m);                             \
    } else if (act == kActSilu) {                                             \
      launch_moe_w8<BM, BN, WM, WN, kActSilu>(m);                             \
    } else {                                                                  \
      launch_moe_w8<BM, BN, WM, WN, kActGeluTanh>(m);                         \
    }                                                                         \
  } while (0)
  if (block_m == 16 && block_n == 64) STRIX_MOE_W8(16, 64, 1, 4);
  else if (block_m == 32 && block_n == 64) STRIX_MOE_W8(32, 64, 2, 2);
  else if (block_m == 64 && block_n == 64) STRIX_MOE_W8(64, 64, 2, 2);
  else if (block_m == 64 && block_n == 128) STRIX_MOE_W8(64, 128, 2, 4);
  else if (block_m == 128 && block_n == 64) STRIX_MOE_W8(128, 64, 2, 2);
  else if (block_m == 128 && block_n == 128) STRIX_MOE_W8(128, 128, 4, 2);
  else if (block_m == 64 && block_n == 256) STRIX_MOE_W8(64, 256, 2, 4);
  else TORCH_CHECK(false, "unsupported block_m/block_n ", block_m, "/", block_n);
#undef STRIX_MOE_W8
}

}  // namespace strix
