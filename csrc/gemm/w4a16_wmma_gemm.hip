// SPDX-License-Identifier: MIT
// W4A16 GEMM for prefill: C[M, N] = A[M, K] (bf16) x W[N, K]^T, W symmetric
// int4 (uint4b8) in vLLM's RDNAHybridW4A16 layout (int8 [N, K/2] view of
// ExLlama-shuffled int32 words: k order 0 2 4 6 1 3 5 7 per nibble) with
// bf16 group scales [N, K / group]. Weights are dequantized to bf16 as they
// are staged into LDS ((nibble - 8) * scale, one scale per 16-element chunk),
// so the main loop is a plain bf16 WMMA GEMM: no activation quantization and
// no per-group fold, which made w4a8_wmma_gemm with 32-element groups fold
// four times per 128-element chunk.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kBK = 64;      // K per main-loop step
constexpr int kPad = 8;      // LDS row pitch kBK + kPad
constexpr int kPitch = kBK + kPad;

template <int BM, int BN, int WAVES_M, int WAVES_N>
__global__ __launch_bounds__(WAVES_M* WAVES_N* kWave) void w4a16_gemm_kernel(
    uint16_t* __restrict__ c, const uint16_t* __restrict__ a,
    const uint8_t* __restrict__ w, const uint16_t* __restrict__ scales,
    const uint16_t* __restrict__ bias, int M, int N, int K, int group,
    int64_t lda, int64_t ldw, int64_t lds_n, int64_t ldc) {
  constexpr int kThreads = WAVES_M * WAVES_N * kWave;
  constexpr int TM = BM / WAVES_M, TN = BN / WAVES_N;
  constexpr int FM = TM / 16, FN = TN / 16;
  constexpr int kChunksA = BM * kBK / 8;   // 16-byte bf16 chunks of A
  constexpr int kChunksB = BN * kBK / 16;  // 8-byte int4 chunks (16 k) of W
  constexpr int kLoadsA = kChunksA / kThreads, kLoadsB = kChunksB / kThreads;
  static_assert(kChunksA % kThreads == 0 && kChunksB % kThreads == 0);

  __shared__ __attribute__((aligned(16))) uint16_t s_a[BM * kPitch];
  __shared__ __attribute__((aligned(16))) uint16_t s_b[BN * kPitch];

  // N tiles fastest: blocks running together share the A rows of one M tile.
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int wm = wave % WAVES_M, wn = wave / WAVES_M;

  typedef uint32_t stage_a_t __attribute__((ext_vector_type(4 * kLoadsA)));
  typedef uint32_t stage_b_t __attribute__((ext_vector_type(2 * kLoadsB)));
  stage_a_t ra = {};
  stage_b_t rb = {};
  float rs[kLoadsB];

  auto load = [&](int k0) {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads, r = ch / (kBK / 8), c8 = (ch % (kBK / 8)) * 8;
      uint4 v = {};
      if (m0 + r < M) v = *reinterpret_cast<const uint4*>(a + int64_t(m0 + r) * lda + k0 + c8);
      ra[4 * i] = v.x, ra[4 * i + 1] = v.y, ra[4 * i + 2] = v.z, ra[4 * i + 3] = v.w;
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads, r = ch / (kBK / 16), k16 = (ch % (kBK / 16)) * 16;
      uint2 v = {};
      float s = 0.f;
      if (n0 + r < N) {
        v = *reinterpret_cast<const uint2*>(w + int64_t(n0 + r) * ldw + (k0 + k16) / 2);
        s = __uint_as_float(uint32_t(scales[int64_t(n0 + r) * lds_n + (k0 + k16) / group]) << 16);
      }
      rb[2 * i] = v.x, rb[2 * i + 1] = v.y;
      rs[i] = s;
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads, r = ch / (kBK / 8), c8 = (ch % (kBK / 8)) * 8;
      *reinterpret_cast<uint4*>(s_a + r * kPitch + c8) =
          make_uint4(ra[4 * i], ra[4 * i + 1], ra[4 * i + 2], ra[4 * i + 3]);
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads, r = ch / (kBK / 16), k16 = (ch % (kBK / 16)) * 16;
      float f[16];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        // ExLlama word: nibble j holds k = 0 2 4 6 1 3 5 7 of these 8.
        const uint32_t wd = rb[2 * i + h];
        constexpr int kOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};
#pragma unroll
        for (int j = 0; j < 8; ++j)
          f[8 * h + kOrder[j]] = float(int((wd >> (4 * j)) & 15u) - 8) * rs[i];
      }
      uint4 lo = make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]),
                            pack_bf16x2(f[4], f[5]), pack_bf16x2(f[6], f[7]));
      uint4 hi = make_uint4(pack_bf16x2(f[8], f[9]), pack_bf16x2(f[10], f[11]),
                            pack_bf16x2(f[12], f[13]), pack_bf16x2(f[14], f[15]));
      uint16_t* dst = s_b + r * kPitch + k16;
      *reinterpret_cast<uint4*>(dst) = lo;
      *reinterpret_cast<uint4*>(dst + 8) = hi;
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
  for (int k0 = 0; k0 < K; k0 += kBK) {
    const bool more = k0 + kBK < K;
    if (more) load(k0 + kBK);
#pragma unroll
    for (int kk = 0; kk < kBK; kk += 16) {
      bf16x16 fb[FN];
#pragma unroll
      for (int j = 0; j < FN; ++j)
        fb[j] = load_frag_bf16(s_b + (wn * TN + j * 16) * kPitch + kk, kPitch, lane);
#pragma unroll
      for (int i = 0; i < FM; ++i) {
        const bf16x16 fa = load_frag_bf16(s_a + (wm * TM + i * 16) * kPitch + kk, kPitch, lane);
#pragma unroll
        for (int j = 0; j < FN; ++j) acc[i][j] = wmma_bf16(fa, fb[j], acc[i][j]);
      }
    }
    __syncthreads();
    if (more) store();
    __syncthreads();
  }

  // acc[i][j][e] = C[m wm*TM + i*16 + acc_row(e)][n wn*TN + j*16 + col]
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int m = m0 + wm * TM + i * 16 + acc_row(e, lane);
      if (m >= M) continue;
#pragma unroll
      for (int j = 0; j < FN; ++j) {
        const int n = n0 + wn * TN + j * 16 + acc_col(lane);
        if (n >= N) continue;
        float v = acc[i][j][e];
        if (bias) v += __uint_as_float(uint32_t(bias[n]) << 16);
        c[int64_t(m) * ldc + n] = static_cast<uint16_t>(pack_bf16x2(v, 0.f) & 0xffffu);
      }
    }
}

// Whole-matrix decode to bf16 [N, K], the same (nibble - 8) * scale and
// round-to-nearest pack as the GEMM's staging (so the bits match its B tile):
// one thread per 16-element chunk (8 bytes, one group scale). For prefill M
// the plugin decodes once and runs hipBLASLt on the result.
__global__ __launch_bounds__(256) void w4a16_dequant_kernel(
    uint16_t* __restrict__ out, const uint8_t* __restrict__ w,
    const uint16_t* __restrict__ scales, int N, int K, int group, int64_t ldw,
    int64_t lds_n) {
  const int64_t ch = int64_t(blockIdx.x) * 256 + threadIdx.x;
  const int per_row = K / 16;
  if (ch >= int64_t(N) * per_row) return;
  const int r = int(ch / per_row), k16 = int(ch % per_row) * 16;
  const uint2 v = *reinterpret_cast<const uint2*>(w + int64_t(r) * ldw + k16 / 2);
  const float sc = __uint_as_float(uint32_t(scales[int64_t(r) * lds_n + k16 / group]) << 16);
  float f[16];
  constexpr int kOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t wd = h ? v.y : v.x;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      f[8 * h + kOrder[j]] = float(int((wd >> (4 * j)) & 15u) - 8) * sc;
  }
  uint4* dst = reinterpret_cast<uint4*>(out + int64_t(r) * K + k16);
  dst[0] = make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]),
                      pack_bf16x2(f[4], f[5]), pack_bf16x2(f[6], f[7]));
  dst[1] = make_uint4(pack_bf16x2(f[8], f[9]), pack_bf16x2(f[10], f[11]),
                      pack_bf16x2(f[12], f[13]), pack_bf16x2(f[14], f[15]));
}

}  // namespace

// x: [M, K] bf16 (contiguous rows); w: int8 [N, K/2] view of vLLM's skinny
// int4 layout (rows may be padded); scales: bf16 [N, K / group] (any row
// stride); bias: bf16 [N] or none. Returns [M, N] bf16.
torch::Tensor w4a16_wmma_gemm(const torch::Tensor& x, const torch::Tensor& w,
                              const torch::Tensor& scales,
                              const std::optional<torch::Tensor>& bias,
                              int64_t group_size) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 &&
                  x.stride(0) % 8 == 0,
              "x must be bf16 [M, K] with 16-byte aligned rows");
  TORCH_CHECK(w.scalar_type() == at::kChar && w.dim() == 2 && w.stride(1) == 1 &&
                  w.stride(0) % 8 == 0,
              "w must be int8 [N, K/2] with 8-byte aligned rows");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(w.size(1) * 2 == k && k % kBK == 0, "K mismatch or K % 64");
  TORCH_CHECK(group_size % 16 == 0 && k % group_size == 0, "group_size % 16 and K % group");
  TORCH_CHECK(scales.scalar_type() == at::kBFloat16 && scales.dim() == 2 &&
                  scales.size(0) == n && scales.size(1) == k / group_size &&
                  scales.stride(1) == 1,
              "scales must be bf16 [N, K / group]");
  if (bias) TORCH_CHECK(bias->scalar_type() == at::kBFloat16 && bias->numel() == n, "bias");
  auto c = torch::empty({m, n}, x.options());
  if (m == 0) return c;
  auto stream = at::hip::getCurrentHIPStream();
#define STRIX_W4(BM, BN, WM, WN)                                             \
  w4a16_gemm_kernel<BM, BN, WM, WN>                                          \
      <<<dim3((n + BN - 1) / BN, (m + BM - 1) / BM), WM * WN * kWave, 0, stream>>>( \
          reinterpret_cast<uint16_t*>(c.data_ptr()),                         \
          reinterpret_cast<const uint16_t*>(x.data_ptr()),                   \
          reinterpret_cast<const uint8_t*>(w.data_ptr()),                    \
          reinterpret_cast<const uint16_t*>(scales.data_ptr()),              \
          bias ? reinterpret_cast<const uint16_t*>(bias->data_ptr()) : nullptr, \
          m, n, k, group_size, x.stride(0), w.stride(0), scales.stride(0),   \
          c.stride(0))
  if (m <= 64) STRIX_W4(64, 128, 2, 4);
  else STRIX_W4(128, 128, 4, 2);
#undef STRIX_W4
  return c;
}

// w: int8 [N, K/2] ExLlama-shuffled int4 (uint4b8); scales: bf16 [N, K /
// group]. Returns the bf16 [N, K] matrix w4a16_wmma_gemm computes against.
torch::Tensor w4a16_dequant(const torch::Tensor& w, const torch::Tensor& scales,
                            int64_t group_size) {
  TORCH_CHECK(w.scalar_type() == at::kChar && w.dim() == 2 && w.stride(1) == 1 &&
                  w.stride(0) % 8 == 0,
              "w must be int8 [N, K/2] with 8-byte aligned rows");
  const int64_t n = w.size(0), k = w.size(1) * 2;
  TORCH_CHECK(k % 16 == 0 && group_size % 16 == 0 && k % group_size == 0,
              "K % 16, group_size % 16 and K % group");
  TORCH_CHECK(scales.scalar_type() == at::kBFloat16 && scales.dim() == 2 &&
                  scales.size(0) == n && scales.size(1) == k / group_size &&
                  scales.stride(1) == 1,
              "scales must be bf16 [N, K / group]");
  auto out = torch::empty({n, k}, w.options().dtype(at::kBFloat16));
  const int64_t chunks = n * (k / 16);
  if (chunks == 0) return out;
  w4a16_dequant_kernel<<<dim3(unsigned((chunks + 255) / 256)), 256, 0,
                         at::hip::getCurrentHIPStream()>>>(
      reinterpret_cast<uint16_t*>(out.data_ptr()),
      reinterpret_cast<const uint8_t*>(w.data_ptr()),
      reinterpret_cast<const uint16_t*>(scales.data_ptr()), int(n), int(k),
      int(group_size), w.stride(0), scales.stride(0));
  return out;
}

}  // namespace strix
