// SPDX-License-Identifier: Apache-2.0
// W8A16 skinny GEMM for decode (M <= 5): uint8b128 weights [N, K], per-channel
// or per-group (g = 128) scales, bf16/fp16 activations.
// Moved from the asf0/strix-vllm fork (csrc/rocm/skinny_gemms_int8.cu), itself derived
// from vLLM's csrc/rocm/skinny_gemms.cu (Apache-2.0, vLLM contributors).
#include <assert.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>  // std::in_range

namespace strix {



// Combined RDNA macro (gfx11 + gfx12) - both use 32-wide wavefronts
#if defined(__GFX11__) || defined(__GFX12__)
  #define __HIP__GFX1X__
#endif

// The decode tuning and the deep-K routing below were swept on gfx115x and are
// kept to it: the rest of gfx11 and gfx12 also compile this kernel, and keep
// the behaviour and the K * N bound they had.
#if defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || \
    defined(__gfx1153__)
  #define __HIP__GFX115X__
#endif

#define LDS_SIZE 64 * 1024

static constexpr int get_lds_size_int8() { return 64 * 1024; }

// Host-side twin of __HIP__GFX115X__, for the tuple choices the dispatch makes.
static bool on_gfx115x_int8() {
  static const bool result = [] {
    const auto* dprops = at::cuda::getCurrentDeviceProperties();
    return std::string(dprops->gcnArchName).find("gfx115") != std::string::npos;
  }();
  return result;
}

static bool on_gfx1x_int8() {
  static const bool result = [] {
    const std::string arch(
        at::cuda::getCurrentDeviceProperties()->gcnArchName);
    return arch.find("gfx11") != std::string::npos ||
           arch.find("gfx12") != std::string::npos;
  }();
  return result;
}

#if defined(NDEBUG)
  #undef NDEBUG
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

template <typename T>
__device__ __forceinline__ float __s2float8(T v);

template <>
__device__ __forceinline__ float __s2float8(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float __s2float8(__hip_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T __float2s8(float v);

template <>
__device__ __forceinline__ half __float2s8(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __hip_bfloat16 __float2s8(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ T loadnt8(T* addr) {
  return __builtin_nontemporal_load(addr);
}

template <typename T>
struct scalar8 {};

template <>
struct scalar8<c10::Half> {
  using type = half;
};

template <>
struct scalar8<c10::BFloat16> {
  using type = __hip_bfloat16;
};

// fp16 dot2 with fp32 accumulation. gfx11/gfx12 have no bf16 dot2 hardware, so
// activations are staged in LDS as fp16 for both input dtypes: bf16 -> fp16
// conversion is exact, so one instruction family covers both (same approach as
// skinny_gemms.cu fp8 kernels converting to half for dot2).
__device__ __forceinline__ float dot2c8(float acc, uint32_t a, uint32_t w) {
  return __builtin_amdgcn_fdot2(*((half2*)(&a)), *((half2*)(&w)), acc, false);
}

// DPP-based wave32 reduction for GFX1X (matches skinny_gemms_int4.cu).
#if defined(__HIP__GFX1X__)
  #define REDUCE_SUM_DPP_WAVE32_8(val)                                      \
    do {                                                                    \
      val += __builtin_amdgcn_mov_dpp(val, 0x118, 0xf, 0xf, 1); /* shr8  */ \
      val += __builtin_amdgcn_mov_dpp(val, 0x114, 0xf, 0xf, 1); /* shr4  */ \
      val += __builtin_amdgcn_mov_dpp(val, 0x112, 0xf, 0xf, 1); /* shr2  */ \
      val += __builtin_amdgcn_mov_dpp(val, 0x111, 0xf, 0xf, 1); /* shr1  */ \
      val += __shfl_xor(val, 16);                                           \
    } while (0)
#endif

__device__ inline unsigned int min__8(uint32_t a, uint32_t b) {
  return min(a, b);
}

// The 0x6400 magic turns two raw bytes into fp16 lanes (1024 + byte); each
// lane then has 1152 = 1024 + 128 subtracted, recovering the signed int8
// value of uint8b128 storage: raw - 128 in [-128, 127], exact in fp16.
[[maybe_unused]] constexpr uint32_t MAGIC_BIAS_8 = 0x64806480u;  // fp16(1152)

// LDS staging: activations are always fp16, whatever the input dtype. Laid
// out piece-major within a block of THRDS lanes
// (see skinny_gemms_int4.cu ACT_LDS_PIECE) so 16-byte LDS accesses tile the
// banks instead of colliding every fourth lane.
constexpr uint32_t ACT_LDS_PIECE8 = 8;  // halves per 16-byte LDS access
struct alignas(16) lds_piece8 {
  float f[ACT_LDS_PIECE8 / 2];
};

union bigTypeG8 {
  __hip_bfloat16 h[ACT_LDS_PIECE8];  // 16 bytes = 8 bf16 elements
  float f[ACT_LDS_PIECE8 / 2];
  uint32_t u32[ACT_LDS_PIECE8 / 2];
};

// One 16-byte piece (8 elements): fp16 is a straight copy, bf16 converts
// element-wise (exact for |v| < 65504; inference activations are far below).
template <typename scalar_t>
__device__ __forceinline__ void copy_piece8_one(lds_piece8* dst,
                                                const scalar_t* src) {
  if constexpr (std::is_same_v<scalar_t, half>) {
    *dst = *(const lds_piece8*)src;
  } else {
    bigTypeG8 g;
#pragma unroll
    for (uint32_t i = 0; i < ACT_LDS_PIECE8 / 2; i++)
      g.u32[i] = reinterpret_cast<const uint32_t*>(src)[i];
#pragma unroll
    for (uint32_t i = 0; i < ACT_LDS_PIECE8; i++)
      ((half*)dst)[i] = __float2half(__bfloat162float(g.h[i]));
  }
}

// Copies A[kBase, kBase + chunkSpan) of every activation row into a
// KFIT-strided fp16 LDS band. The caller owns the barriers. Requires
// K % A_CHUNK == 0, which the host checks.
template <typename scalar_t, int THRDS, int A_CHUNK, int N, uint32_t KFIT>
__device__ __forceinline__ void load_act_chunk_into_lds8(
    half* s, const scalar_t* __restrict__ A, const int K, const uint32_t kBase,
    const uint32_t chunkSpan, const int _WvPrGrp) {
  const uint32_t span = min__8(chunkSpan, static_cast<uint32_t>(K) - kBase);
  const uint32_t step = THRDS * A_CHUNK * _WvPrGrp;
  const uint32_t off = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK;
  for (uint32_t k = 0; k < span; k += step) {
    uint32_t k_in = k + off;
    if (k_in >= span) break;
    // k is a multiple of THRDS * A_CHUNK, so the block base is
    // k + threadIdx.y * (THRDS * A_CHUNK); this thread owns lane
    // threadIdx.x of it.
    const uint32_t blk = k + threadIdx.y * (THRDS * A_CHUNK);
#pragma unroll
    for (int n = 0; n < N; n++) {
      const scalar_t* src = reinterpret_cast<const scalar_t*>(
          __builtin_assume_aligned(
              &A[static_cast<uint32_t>(K) * n + kBase + k_in], 16));
      lds_piece8* dst = (lds_piece8*)__builtin_assume_aligned(
          &s[n * KFIT + blk + threadIdx.x * ACT_LDS_PIECE8], 16);
#pragma unroll
      for (uint32_t p = 0; p < A_CHUNK / ACT_LDS_PIECE8; p++) {
        // Piece-major within the block (piece p of lane i at p * THRDS + i),
        // matching the compute body's reads.
        copy_piece8_one<scalar_t>(&dst[p * THRDS],
                                  src + p * ACT_LDS_PIECE8);
      }
    }
  }
}

// W8A16 skinny GEMM kernel: uint8 weights (uint8b128 storage, one byte per
// element, [M, K] rows), fp16/bf16 activations, per-channel or per-group
// scales. Targets the "sml" case, where the activation fits in LDS as fp16,
// and with CHUNKED the deep-K case, where it is staged there a chunk at a
// time. A_CHUNK: K-elements per thread per step. GROUP_SIZE: 0 = per-channel
// scale [M], >0 = per-group scale [M, K/GROUP_SIZE]; GROUP_SIZE % A_CHUNK ==
// 0 required when GROUP_SIZE > 0, so one thread chunk never straddles a
// group. B_row_stride_bytes is the per-row byte stride of the weights.
// CHUNKED walks K in LDS-sized chunks so all N activation rows stay in LDS
// even when K * N exceeds it. LDS_LEN is a template parameter so the band
// stride is constant and folds into the LDS address immediate.
#if defined(__HIP__GFX1X__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GROUP_SIZE = 0, bool CHUNKED = false,
          uint32_t LDS_LEN = LDS_SIZE / 2>
__global__ void __launch_bounds__(WvPrGrp* THRDS)
    wvSplitK_int8_hf_sml_(const int K, const int M, const int Bx, const int By,
                          const uint8_t* B_packed,
                          const scalar_t* __restrict__ A,
                          const scalar_t* scale,
                          const scalar_t* __restrict__ BIAS, scalar_t* C,
                          const int _WvPrGrp, const int CuCount,
                          const int group_stride,
                          const int B_row_stride_bytes) {
  constexpr int max_lds_len = LDS_LEN;

  union bigTypeA {
    half h[A_CHUNK];
    half2 h2[A_CHUNK / 2];
    float f[A_CHUNK / 2];
  };

  union bigTypeW {
    uint8_t b[A_CHUNK];
    uint32_t u32[A_CHUNK / 4];
  };

  __shared__ half s[max_lds_len];

  const half2 magic_bias2 = *(const half2*)&MAGIC_BIAS_8;

  // CHUNKED fills s[] per chunk instead, and needs the waves above _WvPrGrp
  // gone before the first barrier: s_barrier is untagged on gfx11, so a wave
  // parked at a barrier it does not own would satisfy a reload barrier.
  if constexpr (!CHUNKED) {
    for (uint32_t k = 0; k < min__8(K * N, max_lds_len);
         k += THRDS * WvPrGrp * A_CHUNK) {
      uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);

      if (k_in >= min__8(K * N, max_lds_len)) break;

#pragma unroll
      for (uint32_t p = 0; p < A_CHUNK / ACT_LDS_PIECE8; p++) {
        lds_piece8 piece;
        copy_piece8_one<scalar_t>(&piece, &A[k_in + p * ACT_LDS_PIECE8]);
        *((lds_piece8*)(&s[k_in + p * ACT_LDS_PIECE8])) = piece;
      }
    }
    __syncthreads();
  }

  if (threadIdx.y >= _WvPrGrp) return;

  uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

  // Per-row LDS band stride for CHUNKED, a whole number of K_STEP steps.
  constexpr uint32_t K_STEP = THRDS * A_CHUNK * UNRL;
  constexpr uint32_t kFit = (LDS_LEN / N) - (LDS_LEN / N) % K_STEP;
  static_assert(!CHUNKED || kFit >= K_STEP,
                "CHUNKED needs at least one K_STEP of LDS per activation row");
  static_assert(!CHUNKED || static_cast<uint64_t>(kFit) * N <= LDS_LEN,
                "CHUNKED LDS band overruns s[]");
  [[maybe_unused]] uint32_t kBase = 0;

  // How much is copied per chunk need not be the band stride; spread K evenly
  // instead, still in whole K_STEP steps so a chunk boundary falls between
  // two k1 iterations.
  [[maybe_unused]] uint32_t chunkSpan = kFit;
  if constexpr (CHUNKED) {
    const uint32_t n_chunks = (static_cast<uint32_t>(K) + kFit - 1) / kFit;
    const uint32_t even = (static_cast<uint32_t>(K) + n_chunks - 1) / n_chunks;
    chunkSpan = (even + K_STEP - 1) / K_STEP * K_STEP;
  }

  // Live waves must run the m loop the same number of times or they stop
  // arriving at the reload barriers. The M % YTILE ragged case is routed to
  // CHUNKED by the host dispatch (or to YTILE = 1), so m tiles never straddle
  // the M boundary when writes happen.
  uint32_t mEnd = M;
  if constexpr (CHUNKED) {
    const uint32_t YW = YTILE * _WvPrGrp;
    mEnd = (M % YW == 0) ? M : (M - M % YW + YW);
  }

  [[maybe_unused]] const int num_groups = (GROUP_SIZE > 0) ? group_stride : 0;

  float sum[N][YTILE];

  while (m < mEnd) {
    for (int i = 0; i < YTILE; i++)
      for (int n = 0; n < N; n++) sum[n][i] = 0;

    bigTypeW bigB[YTILE][UNRL];

    // Whole K_STEP blocks take the unchecked path, where the UNRL weight
    // loads clause together; only the ragged tail pays the check.
    auto k_load = [&](auto CHECK_T, bigTypeW(&bigB)[YTILE][UNRL], uint32_t k1) {
      constexpr bool CHECK = decltype(CHECK_T)::value;
#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if constexpr (CHECK) {
          if (k_ >= K) break;
        }

#pragma unroll
        for (int y = 0; y < YTILE; y++) {
          const uint32_t* src = (const uint32_t*)(
              &B_packed[(m + y) * B_row_stride_bytes + k_]);
#pragma unroll
          for (int i = 0; i < A_CHUNK / 4; i++)
            bigB[y][k2].u32[i] = loadnt8((uint32_t*)(&src[i]));
        }
      }
    };

    auto k_compute = [&](auto CHECK_T, const bigTypeW(&bigB)[YTILE][UNRL],
                         uint32_t k1) {
      constexpr bool CHECK = decltype(CHECK_T)::value;

#pragma unroll
      for (uint32_t k2 = 0; k2 < UNRL; k2++) {
        uint32_t k = k1 + k2 * THRDS * A_CHUNK;
        uint32_t k_ = k + threadIdx.x * A_CHUNK;
        if constexpr (CHECK) {
          if (k_ >= K) break;
        }

        bigTypeA bigA[N];
        for (int n = 0; n < N; n++) {
          // Only the LDS address is chunk-relative; the scale lookup below
          // stays on the absolute k_.
          if constexpr (CHUNKED) {
            const lds_piece8* src =
                (const lds_piece8*)__builtin_assume_aligned(
                    &s[kFit * n + (k - kBase) + threadIdx.x * ACT_LDS_PIECE8],
                    16);
            auto* dst = reinterpret_cast<lds_piece8*>(&bigA[n]);
#pragma unroll
            for (uint32_t p = 0; p < A_CHUNK / ACT_LDS_PIECE8; p++)
              dst[p] = src[p * THRDS];
          } else {
            bigA[n] = *((const bigTypeA*)(&(s[k_ + K * n])));
          }
        }

#pragma unroll
        for (int n = 0; n < N; n++) {
#pragma unroll
          for (int y = 0; y < YTILE; y++) {
            float partial = 0;
#pragma unroll
            for (uint32_t w = 0; w < A_CHUNK / 4; w++) {
              uint32_t qa = bigB[y][k2].u32[w];

              // Per-byte fp16 lanes: lane0 = 0x6400 | byte0 at bits 0-7,
              // lane1 = 0x6400 | byte1 shifted from bits 8-15 into 16-23.
              // One word covers elements k..k+3 in consecutive halves; the
              // group scale is applied at the chunk accumulate below (the
              // int4 sibling's structure: raw values only in the dot).
              uint32_t lo = 0x64006400u | (qa & 0x000000FFu) |
                            ((qa << 8) & 0x00FF0000u);
              uint32_t hi = 0x64006400u | ((qa >> 16) & 0x000000FFu) |
                            ((qa & 0xFF000000u) >> 8);
              half2 w01 = __hsub2(*(half2*)(&lo), magic_bias2);
              half2 w23 = __hsub2(*(half2*)(&hi), magic_bias2);

              partial += __builtin_amdgcn_fdot2(
                  *((half2*)(&bigA[n].f[2 * w])), *((half2*)(&w01)), 0.0f,
                  false);
              partial += __builtin_amdgcn_fdot2(
                  *((half2*)(&bigA[n].f[2 * w + 1])), *((half2*)(&w23)), 0.0f,
                  false);
            }

            if constexpr (GROUP_SIZE > 0) {
              const int group_pos = (m + y) * num_groups + k_ / GROUP_SIZE;
              sum[n][y] +=
                  partial * __s2float8(scale[group_pos]);
            } else {
              sum[n][y] += partial;
            }
          }
        }
      }
    };

    if constexpr (CHUNKED) {
      const uint32_t K_whole = K - (K % K_STEP);
      for (uint32_t kb = 0; kb < static_cast<uint32_t>(K); kb += chunkSpan) {
        __syncthreads();
        load_act_chunk_into_lds8<scalar_t, THRDS, A_CHUNK, N, kFit>(
            s, A, K, kb, chunkSpan, _WvPrGrp);
        __syncthreads();
        kBase = kb;
        // Waves past the real M had to reach the barriers above, but have
        // no rows to reduce.
        if (m >= M) continue;
        const uint32_t kStop = min__8(K_whole, kb + chunkSpan);
        for (uint32_t k1 = kb; k1 < kStop; k1 += K_STEP) {
          k_load(std::false_type{}, bigB, k1);
          k_compute(std::false_type{}, bigB, k1);
        }
        // chunkSpan is a multiple of K_STEP, so the ragged tail falls in
        // the final chunk.
        if (kb + chunkSpan >= static_cast<uint32_t>(K) &&
            K_whole < static_cast<uint32_t>(K)) {
          k_load(std::true_type{}, bigB, K_whole);
          k_compute(std::true_type{}, bigB, K_whole);
        }
      }
    } else {
      const uint32_t K_whole = K - (K % K_STEP);
      for (uint32_t k1 = 0; k1 < K_whole; k1 += K_STEP) {
        k_load(std::false_type{}, bigB, k1);
        k_compute(std::false_type{}, bigB, k1);
      }
      if (K_whole < K) {
        k_load(std::true_type{}, bigB, K_whole);
        k_compute(std::true_type{}, bigB, K_whole);
      }
    }

    if constexpr (CHUNKED) {
      if (m >= M) {
        m += CuCount * _WvPrGrp * YTILE;
        kBase = 0;
        continue;
      }
    }

    for (int n = 0; n < N; n++)
      for (int y = 0; y < YTILE; y++) REDUCE_SUM_DPP_WAVE32_8(sum[n][y]);

    if (threadIdx.x == (THRDS - 1)) {
      for (int n = 0; n < N; n++) {
        for (int i = 0; i < YTILE; i++) {
          if constexpr (GROUP_SIZE == 0) {
            sum[n][i] *= __s2float8(scale[m + i]);
          }
          if (BIAS) sum[n][i] += __s2float8(BIAS[(m + i) % Bx + (n % By) * M]);
          C[m + i + n * M] = __float2s8<scalar_t>(sum[n][i]);
        }
      }
    }
    m += CuCount * _WvPrGrp * YTILE;
  }
}
#else  // !defined(__HIP__GFX1X__)
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK,
          int UNRL, int N, int GROUP_SIZE = 0, bool CHUNKED = false,
          uint32_t LDS_LEN = LDS_SIZE / 2>
__global__ void wvSplitK_int8_hf_sml_(
    const int K, const int M, const int Bx, const int By,
    const uint8_t* B_packed, const scalar_t* __restrict__ A,
    const scalar_t* scale, const scalar_t* __restrict__ BIAS, scalar_t* C,
    const int _WvPrGrp, const int CuCount, const int group_stride,
    const int B_row_stride_bytes) {
  UNREACHABLE_CODE
}
#endif  // defined(__HIP__GFX1X__)

static int mindiv_int8(int N, int div1, int div2) {
  int nPrRnd = div1 * div2;
  int limit = div2 < 13 ? div2 : 13;
  int rnds[16];
  for (int i = 0; i < limit; i++) {
    rnds[i] = (N + nPrRnd - 1) / nPrRnd;
    nPrRnd -= div1;
  }
  for (int i = limit - 1; i >= 0; i--)
    if (rnds[0] == rnds[i]) return (div2 - i);
  return 0;
}

// W8A16 skinny GEMM: uint8 weights (uint8b128, one byte per element) with
// group-wise scales.
// in_a: weights [M, >=K] uint8; rows may be padded past K (stride >= K)
// in_b: activations [N, K] (fp16/bf16), contiguous, N in [1, 5]
// in_scale: group scales [M, K/group_size] or per-channel scales [M]
// in_bias: optional bias [M] or [N, M]
// group_size: 0 (per-channel), 32, 64, or 128. Symmetric (uint8b128)
// dequantization only; there is no zero-point support (that format stores
// raw uint8 zero points per group and is not needed by W8A16 checkpoints
// quantized with llm-compressor symmetric schemes).
torch::Tensor w8a16_skinny_gemm(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size) {
  TORCH_CHECK(on_gfx1x_int8(),
              "w8a16_skinny_gemm targets gfx11/gfx12 wave32 GPUs only");

  auto M_in = in_a.size(0);  // weight rows = output features
  auto K_in = in_b.size(1);
  auto N_in = in_b.size(0);  // activation rows = batch, must be 1..5
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == at::kByte,
              "Weights must be uint8 (uint8b128 layout, one byte per element),"
              " got ",
              in_a.dtype());
  TORCH_CHECK(in_a.numel() >= M_in * K_in,
              "Weight tensor must hold at least M*K bytes, got ",
              in_a.numel(), " expected >= ", M_in * K_in);
  const int64_t b_row_stride_bytes = in_a.stride(0) * in_a.element_size();
  TORCH_CHECK(b_row_stride_bytes >= K_in, "Weight row stride (",
              b_row_stride_bytes, " bytes) must hold at least K=", K_in,
              " bytes per row");
  TORCH_CHECK(std::in_range<int>(b_row_stride_bytes),
              "Weight row stride (", b_row_stride_bytes,
              " bytes) exceeds int range");
  const int b_row_stride_bytes_i32 = static_cast<int>(b_row_stride_bytes);
  TORCH_CHECK(
      in_b.dtype() == torch::kFloat16 || in_b.dtype() == torch::kBFloat16,
      "Activation must be float16 or bfloat16");
  TORCH_CHECK(in_scale.dtype() == in_b.dtype(),
              "Scale dtype must match activation dtype");
  TORCH_CHECK(group_size == 0 || group_size == 32 || group_size == 64 ||
                  group_size == 128,
              "group_size must be 0 (per-channel), 32, 64, or 128, got ",
              group_size);
  if (group_size > 0) {
    TORCH_CHECK(K_in % group_size == 0,
                "K must be divisible by group_size=", group_size);
    TORCH_CHECK(group_size % 16 == 0,
                "group_size must be divisible by the A_CHUNK of 16");
  }
  const int64_t num_groups = (group_size > 0) ? K_in / group_size : M_in;
  if (group_size > 0) {
    TORCH_CHECK(in_scale.dim() == 2,
                "Scale must be 2D [M, K/group_size], got shape ",
                in_scale.sizes());
    TORCH_CHECK(in_scale.size(0) == M_in && in_scale.size(1) == num_groups,
                "Scale must be [M, K/group_size] = [", M_in, ", ", num_groups,
                "] but got [", in_scale.size(0), ", ", in_scale.size(1), "]");
    // Scale rows are indexed by their own stride, so the caller may pad them
    // past K / group_size. Only dim 0 may be padded; dim 1 must stay
    // contiguous because the kernel reads consecutive groups.
    TORCH_CHECK(in_scale.stride(1) == 1, "Scale must be contiguous in dim 1");
    TORCH_CHECK(in_scale.stride(0) >= num_groups,
                "Scale row stride must be at least K/group_size=", num_groups,
                ", got ", in_scale.stride(0));
    TORCH_CHECK(std::in_range<int>(in_scale.stride(0)),
                "Scale row stride overflows int");
  } else {
    TORCH_CHECK(in_scale.dim() == 1 && in_scale.numel() == M_in,
                "Per-channel scale must be [M=", M_in, "], got shape ",
                in_scale.sizes());
    TORCH_CHECK(in_scale.stride(0) == 1, "Scale must be contiguous");
  }
  const int group_stride_i32 = static_cast<int>(in_scale.stride(0));
  TORCH_CHECK(K_in % 16 == 0,
              "K must be divisible by 16, got ", K_in);
  TORCH_CHECK(in_b.is_contiguous(), "Activation must be contiguous");

  // gfx115x walks K in LDS-sized chunks, so K * N is unbounded there.
  // Elsewhere the activation must fit LDS outright.
  const int max_lds_len = get_lds_size_int8() / 2;
  TORCH_CHECK(on_gfx115x_int8() || K_in * N_in <= max_lds_len,
              "K*N exceeds LDS capacity on this GPU (deep-K requires gfx115x);"
              " K=",
              K_in, " N=", N_in);

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_b.dtype()).device(in_b.device()));

  dim3 grid(CuCount);
  const c10::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

// Ragged M (weight rows) must not straddle an m tile when the write happens,
// so YTILE is snapped to the largest of {4, 2, 1} that divides M.
#define WVSPLITK_INT8G_LAUNCH(_YTILE, _UNRL, _N, _GS)                        \
  {                                                                          \
    dim3 block(_THRDS8, 16);                                                 \
    int __wvPrGrp = mindiv_int8(M_in, CuCount * _YTILE, 16);                 \
    if (K_in * N_in <= max_lds_len)                                          \
      wvSplitK_int8_hf_sml_<fptype, _THRDS8, _YTILE, 16, A_CHUNK8, _UNRL,    \
                            _N, _GS><<<grid, block, 0, stream>>>(            \
          K_in, M_in, Bx_in, By_in, wptr, aptr, sptr, biasptr, cptr,         \
          __wvPrGrp, CuCount, group_stride_i32, b_row_stride_bytes_i32);     \
    else                                                                     \
      wvSplitK_int8_hf_sml_<fptype, _THRDS8, _YTILE, 16, A_CHUNK8, _UNRL,    \
                            _N, _GS, /*CHUNKED=*/true, lds_elems8>           \
          <<<grid, block, 0, stream>>>(                                     \
          K_in, M_in, Bx_in, By_in, wptr, aptr, sptr, biasptr, cptr,         \
          __wvPrGrp, CuCount, group_stride_i32, b_row_stride_bytes_i32);     \
  }

#define WVSPLIT_INT8G_GS(_YTILE, _UNRL, _N)                                   \
  if (group_size == 0)                                                        \
    WVSPLITK_INT8G_LAUNCH(_YTILE, _UNRL, _N, 0)                               \
  else if (group_size == 32)                                                  \
    WVSPLITK_INT8G_LAUNCH(_YTILE, _UNRL, _N, 32)                              \
  else if (group_size == 64)                                                  \
    WVSPLITK_INT8G_LAUNCH(_YTILE, _UNRL, _N, 64)                              \
  else                                                                        \
    WVSPLITK_INT8G_LAUNCH(_YTILE, _UNRL, _N, 128)

#define WVSPLIT_INT8G_TILE(_sTY, __N)                                         \
  {                                                                           \
    /* Ragged M (weight rows) must not straddle an m tile at write time, so */\
    /* YTILE is snapped at runtime to the largest of {4, 2, 1} dividing M. */ \
    if (M_in % 4 == 0)                                                        \
      WVSPLIT_INT8G_TILE_YT(_sTY, 4, __N)                                     \
    else if (M_in % 2 == 0)                                                   \
      WVSPLIT_INT8G_TILE_YT(_sTY, 2, __N)                                     \
    else                                                                      \
      WVSPLIT_INT8G_TILE_YT(_sTY, 1, __N)                                     \
  }

#define WVSPLIT_INT8G_TILE_YT(_sYT, _YT, __N)                                 \
  {                                                                           \
    if (K_in * N_in > max_lds_len) {                                          \
      /* gfx115x deep-K: walk K in LDS-sized chunks. */                       \
      if (_sYT < 30)                                                          \
        WVSPLIT_INT8G_GS(_YT, 2, __N)                                         \
      else                                                                    \
        WVSPLIT_INT8G_GS(_YT, 1, __N)                                         \
    } else if (__N == 1 && _sYT >= 40) {                                      \
      /* Many rows per CU: fewer k-steps to amortise. */                      \
      WVSPLIT_INT8G_GS(_YT, 2, __N)                                           \
    } else {                                                                  \
      WVSPLIT_INT8G_GS(_YT, 4, __N)                                           \
    }                                                                         \
  }

  constexpr int A_CHUNK8 = 16;
  constexpr int _THRDS8 = 32;
  constexpr int lds_elems8 = LDS_SIZE / 2;

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      in_b.scalar_type(), "w8a16_skinny_gemm", [&] {
        using fptype = typename scalar8<scalar_t>::type;
        const uint8_t* wptr = reinterpret_cast<const uint8_t*>(in_a.data_ptr());
        const fptype* aptr = reinterpret_cast<const fptype*>(in_b.data_ptr());
        const fptype* sptr =
            reinterpret_cast<const fptype*>(in_scale.data_ptr());
        const fptype* biasptr =
            (in_bias.has_value() && in_bias->numel() > 0)
                ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
                : nullptr;
        fptype* cptr = reinterpret_cast<fptype*>(out_c.data_ptr());

        // Largest of {4, 2, 1} dividing M keeps every m tile inside M: the
        // tile choice happens at runtime inside WVSPLIT_INT8G_TILE.
        int sYT = (M_in + CuCount * 4 - 1) / (CuCount * 4);
        switch (N_in) {
          case 1:
            WVSPLIT_INT8G_TILE(sYT, 1)
            break;
          case 2:
            WVSPLIT_INT8G_TILE(sYT, 2)
            break;
          case 3:
            WVSPLIT_INT8G_TILE(sYT, 3)
            break;
          case 4:
            WVSPLIT_INT8G_TILE(sYT, 4)
            break;
          default:
            WVSPLIT_INT8G_TILE(sYT, 5)
            break;
        }
      });

#undef WVSPLIT_INT8G_TILE_YT
#undef WVSPLIT_INT8G_TILE
#undef WVSPLIT_INT8G_GS
#undef WVSPLITK_INT8G_LAUNCH

  return out_c;
}
}  // namespace strix
