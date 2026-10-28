// SPDX-License-Identifier: MIT
// Grouped expert GEMM over vLLM's moe_align_block_size layout:
//   C[sid, n] = sum_k A[sid / top_k, k] * B[expert(sid), n, k]  (* w[sid])
// for every valid sorted id `sid`. One block owns BM sorted ids (one expert,
// as moe_align_block_size guarantees) and BN output columns.
//
// B rows should not sit a large power of two apart (pad the expert weights'
// row pitch): every row of a tile is read at the same k, and 4 KiB-strided
// rows camp on the same DRAM channels on gfx1151 (4x slower, measured).
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kMinBK = 32;  // smallest K step; K must be a multiple of BK
constexpr int kPad = 8;     // LDS row pitch BK + kPad keeps 16-byte alignment

// STAGES = 2 double-buffers LDS (one barrier per K step); 1 halves LDS for
// tiles that would not fit twice.
//
// ACT fuses vLLM's {silu,gelu_tanh}_and_mul into the w13 GEMM: b rows are
// [gate (N/2); up (N/2)], a block's BN rows are BN/2 gate rows and the matching
// BN/2 up rows, and c is [slots, N/2] = act(gate) * up. The gate half of the
// tile is held by waves wn < WAVES_N / 2, the up half by the rest; the up waves
// pass their accumulators through the (by then idle) tile LDS.
//
// ACT is a *template* parameter, not a runtime `act`: each epilogue is then
// straight-line code, and kActSilu keeps the exact expression (and so the
// codegen and the bits) it had when this was `bool SILU_MUL`, which the
// 27B/35B/LFM numbers were measured on. The cost is one extra instantiation
// per tile config, selected in the host dispatcher below.
enum Act : int { kActNone = 0, kActSilu = 1, kActGeluTanh = 2 };

// Gate nonlinearity. GELU-tanh is HF's gelu_pytorch_tanh:
//   0.5 g (1 + tanh(z)), z = sqrt(2/pi) (g + 0.044715 g^3)
// with tanh(z) = 1 - 2 / (e^{2z} + 1), i.e. one __expf (a single v_exp_f32)
// instead of ocml tanhf's polynomial; the saturating limits fall out of the
// same expression (e^{2z} -> inf gives g, -> 0 gives 0, never inf * 0).
template <int ACT>
__device__ __forceinline__ float moe_gate(float g) {
  if constexpr (ACT == kActSilu) {
    return g / (1.f + __expf(-g));
  } else {
    const float z = 0.7978845608028654f * (g + 0.044715f * g * g * g);
    return 0.5f * g * (2.f - 2.f / (__expf(2.f * z) + 1.f));
  }
}

template <int BM, int BN, int WAVES_M, int WAVES_N, int kBK, int STAGES,
          int ACT>
__global__ __launch_bounds__(WAVES_M* WAVES_N* kWave) void moe_gemm_bf16_kernel(
    uint16_t* __restrict__ c, const uint16_t* __restrict__ a,
    const uint16_t* __restrict__ b, const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_post_padded,
    const float* __restrict__ topk_w, int64_t num_valid, int top_k, int k_dim,
    int64_t lda, int64_t ldb, int64_t b_expert_stride, int64_t ldc,
    int n_half) {
  constexpr int kThreads = WAVES_M * WAVES_N * kWave;
  constexpr int kPitch = kBK + kPad;
  constexpr int TM = BM / WAVES_M, TN = BN / WAVES_N;
  constexpr int FM = TM / 16, FN = TN / 16;
  constexpr int kChunksA = BM * kBK / 8, kChunksB = BN * kBK / 8;
  constexpr int kLoadsA = (kChunksA + kThreads - 1) / kThreads;
  constexpr int kLoadsB = (kChunksB + kThreads - 1) / kThreads;
  static_assert(FM >= 1 && FN >= 1 && TM % 16 == 0 && TN % 16 == 0);

  static_assert(ACT == kActNone ||
                    (WAVES_N % 2 == 0 &&
                     BM * (BN / 2) * 4 <= STAGES * (BM + BN) * kPitch * 2),
                "the fused and-mul epilogue needs an even wave split and LDS "
                "for the up half");
  __shared__ __attribute__((aligned(16))) uint16_t s_buf[STAGES * (BM + BN) * kPitch];
  uint16_t(*s_a)[BM * kPitch] = reinterpret_cast<uint16_t(*)[BM * kPitch]>(s_buf);
  uint16_t(*s_b)[BN * kPitch] =
      reinterpret_cast<uint16_t(*)[BN * kPitch]>(s_buf + STAGES * BM * kPitch);
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

  // Fused and-mul: tile rows [0, BN/2) are gate rows n0/2.., the rest up rows
  // N/2 + n0/2.. (n_half = N/2); otherwise BN consecutive rows from n0.
  const uint16_t* bexp = b + expert * b_expert_stride +
                         int64_t(ACT != kActNone ? n0 / 2 : n0) * ldb;
  // Staging registers as flat vectors: uint4 arrays were kept in scratch.
  typedef uint32_t stage_a_t __attribute__((ext_vector_type(4 * kLoadsA)));
  typedef uint32_t stage_b_t __attribute__((ext_vector_type(4 * kLoadsB)));
  stage_a_t ra = {};
  stage_b_t rb = {};

#define STRIX_LOAD_TILES(K0)                                                 \
  _Pragma("unroll") for (int i = 0; i < kLoadsA; ++i) {                      \
    const int ch = tid + i * kThreads;                                       \
    uint4 v = make_uint4(0, 0, 0, 0);                                        \
    if (ch < kChunksA) {                                                     \
      const int64_t row = s_arow[ch / (kBK / 8)];                            \
      if (row >= 0)                                                          \
        v = *reinterpret_cast<const uint4*>(a + row + (K0) +                 \
                                            (ch % (kBK / 8)) * 8);           \
    }                                                                        \
    ra[4 * i] = v.x, ra[4 * i + 1] = v.y, ra[4 * i + 2] = v.z,               \
    ra[4 * i + 3] = v.w;                                                     \
  }                                                                          \
  _Pragma("unroll") for (int i = 0; i < kLoadsB; ++i) {                      \
    const int ch = tid + i * kThreads;                                       \
    if (ch < kChunksB) {                                                     \
      const int brow = ch / (kBK / 8);                                       \
      const bool bup = ACT != kActNone && brow >= BN / 2;                    \
      const int64_t boff = bup ? int64_t(n_half - BN / 2 + brow) * ldb       \
                               : int64_t(brow) * ldb;                        \
      const uint4 v = *reinterpret_cast<const uint4*>(                       \
          bexp + boff + (K0) + (ch % (kBK / 8)) * 8);                        \
      rb[4 * i] = v.x, rb[4 * i + 1] = v.y, rb[4 * i + 2] = v.z,             \
      rb[4 * i + 3] = v.w;                                                   \
    }                                                                        \
  }
#define STRIX_STORE_TILES(STAGE)                                             \
  _Pragma("unroll") for (int i = 0; i < kLoadsA; ++i) {                      \
    const int ch = tid + i * kThreads;                                       \
    if (ch < kChunksA)                                                       \
      *reinterpret_cast<uint4*>(s_a[STAGE] + (ch / (kBK / 8)) * kPitch +     \
                                (ch % (kBK / 8)) * 8) =                      \
          make_uint4(ra[4 * i], ra[4 * i + 1], ra[4 * i + 2], ra[4 * i + 3]);  \
  }                                                                          \
  _Pragma("unroll") for (int i = 0; i < kLoadsB; ++i) {                      \
    const int ch = tid + i * kThreads;                                       \
    if (ch < kChunksB)                                                       \
      *reinterpret_cast<uint4*>(s_b[STAGE] + (ch / (kBK / 8)) * kPitch +     \
                                (ch % (kBK / 8)) * 8) =                      \
          make_uint4(rb[4 * i], rb[4 * i + 1], rb[4 * i + 2], rb[4 * i + 3]);  \
  }

  f32x8 acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) acc[i][j] = f32x8{};

  STRIX_LOAD_TILES(0)
  STRIX_STORE_TILES(0)
  __syncthreads();
  int stage = 0;
  for (int k0 = 0; k0 < k_dim; k0 += kBK) {
    const bool more = k0 + kBK < k_dim;
    if (more) {
      STRIX_LOAD_TILES(k0 + kBK)
    }
    const uint16_t* sa = s_a[stage];
    const uint16_t* sb = s_b[stage];
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      // B fragments stay resident; A fragments stream one at a time.
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
    if constexpr (STAGES == 2) {
      if (more) {
        STRIX_STORE_TILES(stage ^ 1)
      }
      __syncthreads();
      stage ^= 1;
    } else {
      __syncthreads();
      if (more) {
        STRIX_STORE_TILES(0)
      }
      __syncthreads();
    }
  }

#undef STRIX_LOAD_TILES
#undef STRIX_STORE_TILES

  // acc[i][j][e] = C[token row wm*TM + i*16 + acc_row(e)][n wn*TN + j*16 + col]
  if constexpr (ACT != kActNone) {
    // The loop's last barrier left the tile LDS idle.
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

// Launch arguments as one struct so the tile dispatch below names each config
// once instead of once per activation.
struct MoeLaunch {
  uint16_t* c;
  const uint16_t* a;
  const uint16_t* b;
  const int32_t* sorted_ids;
  const int32_t* expert_ids;
  const int32_t* num_post_padded;
  const float* topk_w;
  int64_t num_valid;
  int top_k, k_dim;
  int64_t lda, ldb, b_expert_stride, ldc;
  int n_half;
  dim3 grid;
  hipStream_t stream;
};

template <int BM, int BN, int WM, int WN, int BK, int ST, int ACT>
inline void launch_moe(const MoeLaunch& m) {
  moe_gemm_bf16_kernel<BM, BN, WM, WN, BK, ST, ACT>
      <<<m.grid, WM * WN * kWave, 0, m.stream>>>(
          m.c, m.a, m.b, m.sorted_ids, m.expert_ids, m.num_post_padded, m.topk_w,
          m.num_valid, m.top_k, m.k_dim, m.lda, m.ldb, m.b_expert_stride,
          m.ldc, m.n_half);
}

}  // namespace

// a: [rows, K] bf16 (row = sid / top_k); b: [E, N, K] bf16, K contiguous;
// c: [num_valid, N] view (row = sid), written only at valid sorted ids;
// [num_valid, N / 2] = act(gate) * up when silu_mul (b rows [gate; up]), with
// `activation` picking act: "silu" (default, the original behaviour) or
// "gelu_tanh" (gemma-4's MoE, hidden_activation gelu_pytorch_tanh). The
// parameter is trailing and defaulted, so existing positional calls are
// unchanged.
void moe_gemm_bf16(torch::Tensor& c, const torch::Tensor& a,
                   const torch::Tensor& b, const torch::Tensor& sorted_ids,
                   const torch::Tensor& expert_ids,
                   const torch::Tensor& num_post_padded,
                   const std::optional<torch::Tensor>& topk_weights,
                   int64_t top_k, int64_t block_m, int64_t block_n,
                   bool silu_mul, c10::string_view activation) {
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
                  c.scalar_type() == at::kBFloat16,
              "moe_gemm_bf16: a, b, c must be bf16");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 3 && c.dim() == 2, "bad ranks");
  TORCH_CHECK(a.stride(1) == 1 && b.stride(2) == 1 && c.stride(1) == 1,
              "last dims must be contiguous");
  const int64_t n = b.size(1), k = b.size(2);
  TORCH_CHECK(a.size(1) == k && c.size(1) == (silu_mul ? n / 2 : n),
              "shape mismatch");
  TORCH_CHECK(!silu_mul || (n % 2 == 0 && !topk_weights),
              "silu_mul needs even N and no topk weights");
  TORCH_CHECK(activation == "silu" || activation == "gelu_tanh",
              "activation must be silu or gelu_tanh, got ", activation);
  TORCH_CHECK(k % kMinBK == 0 && n % block_n == 0, "K % ", kMinBK,
              " and N % block_n");
  TORCH_CHECK(a.stride(0) % 8 == 0 && b.stride(1) % 8 == 0 && b.stride(0) % 8 == 0,
              "16-byte aligned rows required");
  TORCH_CHECK(sorted_ids.scalar_type() == at::kInt &&
                  expert_ids.scalar_type() == at::kInt &&
                  num_post_padded.scalar_type() == at::kInt,
              "routing tensors must be int32");
  // moe_align_block_size sizes sorted_ids as numel + E * (block_m - 1), not
  // a multiple of block_m; blocks past num_post_padded exit before reading it.
  TORCH_CHECK(expert_ids.numel() * block_m >= sorted_ids.numel(),
              "expert_ids too short for block_m ", block_m);
  const float* w = nullptr;
  if (topk_weights) {
    TORCH_CHECK(topk_weights->scalar_type() == at::kFloat &&
                    topk_weights->is_contiguous() &&
                    topk_weights->numel() == c.size(0),
                "topk_weights must be contiguous fp32 [num_valid]");
    w = topk_weights->data_ptr<float>();
  }

  const dim3 grid(n / block_n, expert_ids.numel());
  auto stream = at::hip::getCurrentHIPStream();
  const int act = !silu_mul ? kActNone
                  : activation == "gelu_tanh" ? kActGeluTanh
                                              : kActSilu;
  const MoeLaunch m{
      reinterpret_cast<uint16_t*>(c.data_ptr()),
      reinterpret_cast<const uint16_t*>(a.data_ptr()),
      reinterpret_cast<const uint16_t*>(b.data_ptr()),
      sorted_ids.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(),
      num_post_padded.data_ptr<int32_t>(),
      w,
      c.size(0),
      int(top_k),
      int(k),
      a.stride(0),
      b.stride(1),
      b.stride(0),
      c.stride(0),
      int(n / 2),
      grid,
      stream};
#define STRIX_MOE(BM, BN, WM, WN, BK, ST)                                     \
  do {                                                                        \
    TORCH_CHECK(k % BK == 0, "K % ", BK);                                     \
    if (act == kActNone) {                                                    \
      launch_moe<BM, BN, WM, WN, BK, ST, kActNone>(m);                        \
    } else if (act == kActSilu) {                                             \
      launch_moe<BM, BN, WM, WN, BK, ST, kActSilu>(m);                        \
    } else {                                                                  \
      launch_moe<BM, BN, WM, WN, BK, ST, kActGeluTanh>(m);                    \
    }                                                                         \
  } while (0)
  if (block_m == 16 && block_n == 64) STRIX_MOE(16, 64, 1, 4, 32, 2);
  else if (block_m == 32 && block_n == 64) STRIX_MOE(32, 64, 2, 2, 32, 2);
  else if (block_m == 64 && block_n == 64) STRIX_MOE(64, 64, 2, 2, 32, 2);
  else if (block_m == 64 && block_n == 128) STRIX_MOE(64, 128, 2, 4, 64, 2);
  else if (block_m == 128 && block_n == 64) STRIX_MOE(128, 64, 2, 2, 32, 2);
  else if (block_m == 128 && block_n == 128) STRIX_MOE(128, 128, 4, 2, 64, 1);
  else if (block_m == 64 && block_n == 256) STRIX_MOE(64, 256, 2, 4, 32, 2);
  else TORCH_CHECK(false, "unsupported block_m/block_n ", block_m, "/", block_n);
#undef STRIX_MOE
}

}  // namespace strix
