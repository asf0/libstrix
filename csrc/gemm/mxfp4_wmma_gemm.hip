// SPDX-License-Identifier: MIT
// MXFP4 W4A16 GEMM: C[M, N] = A[M, K] (bf16) x W[N, K]^T, W OCP MXFP4
// (e2m1 nibbles, one e8m0 byte per 32 elements) as Quark checkpoints store it:
// weight u8 [N, K/2], weight_scale u8 [N, K/32].
//
// Clone of w4a16_wmma_gemm.hip: the weights are decoded to bf16 as they are
// staged into LDS and the main loop is a plain bf16 WMMA GEMM, so this is
// W4A16 - no activation QDQ (vLLM's emulation kernel quantizes activations to
// fp4 first, this path is strictly more accurate) and no per-forward
// dequantized weight copy. That copy is the whole point of the kernel: 19 GB
// of fp4 resident against 59 GiB for the emulation's dequant-at-load mode:
// 127.5 ms per decode step against 271, on gemma-4-31B-it-MXFP4 at 512 context.
//
// Decode per byte: the low nibble is the even k, the high nibble the odd one
// (verified against quark.torch.kernel.mx.dq_mxfp4, see tests/test_mxfp4.py).
// LUT = {0, 0.5, 1, 1.5, 2, 3, 4, 6} = 0.5 * 2^(n >> 1) * (1 + 0.5 * (n & 1))
// for n >= 2 and 0.5 for n == 1, so the bf16 pattern is 0x3F00 for n == 1 and
// 0x3F00 + n * 0x40 above that: one add per value, no table, no fp multiply.
// The e8m0 scale 2^(s - 127) is folded in by adding (s - 127) << 7 to that
// pattern instead of multiplying: the product is exact in bf16 anyway (the
// value has at most one mantissa bit), so this saves the convert and the
// round-to-nearest of pack_bf16x2 on every weight element. s == 0 is the
// zero block and masks the whole value.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kBlock = 32;  // elements per e8m0 scale byte
constexpr int kPad = 8;     // LDS row pitch BK + kPad
// K granularity the op requires: the decode tiles stage 128 K per step, and
// every Quark MXFP4 shape (2688, 5376, 8192, 16384, 21504) is a multiple of it.
constexpr int kBK = 128;

// One FP4 e2m1 nibble -> bf16 bits, with the block exponent folded in. `base`
// is the bf16 pattern of 0.5 * 2^(s - 127) and `keep` is 0 for a zero block.
// The LUT is {0, 0.5, 1, 1.5, 2, 3, 4, 6}: t = 1 is base itself, t >= 2 is
// base + t * 0x40 (0x3F00 = 0.5, 0x3F80 = 1, 0x4000 = 2, 0x40C0 = 6), so the
// decode is one add per value with no table and no fp multiply - the e8m0
// scale is folded into the exponent bits, which is exact because an e2m1 value
// has at most one mantissa bit and 2^(s - 127) is a power of two.
__device__ __forceinline__ uint32_t mxfp4_nibble(uint32_t v, uint32_t base,
                                                 uint32_t keep) {
  const uint32_t t = v & 7u;
  const uint32_t r = base + ((t >= 2u) ? (t << 6) : 0u);
  return t ? (((r | ((v & 8u) << 12)) & keep)) : 0u;
}

// Two nibbles of one byte -> one bf16 pair (k even in the low half).
__device__ __forceinline__ uint32_t mxfp4_pair(uint32_t byte, uint32_t base,
                                               uint32_t keep) {
  return mxfp4_nibble(byte & 15u, base, keep) |
         (mxfp4_nibble(byte >> 4, base, keep) << 16);
}

// BK is the K staged per main-loop step. 64 is two MX blocks per row; 128 puts
// twice the weight bytes in flight per thread, which is what decode wants: one
// 16-byte load per thread per step ran the 21504x5376 stream at 139 GB/s, two
// at 155, and BK 256 halves occupancy (LDS) instead of helping.
template <int BM, int BN, int WAVES_M, int WAVES_N, int BK, bool STAGGER>
__global__ __launch_bounds__(WAVES_M* WAVES_N* kWave) void mxfp4_gemm_kernel(
    uint16_t* __restrict__ c, const uint16_t* __restrict__ a,
    const uint8_t* __restrict__ w, const uint8_t* __restrict__ scales,
    const uint16_t* __restrict__ bias, int M, int N, int K, int64_t lda,
    int64_t ldw, int64_t lds_n, int64_t ldb, int64_t ldc) {
  constexpr int kPitch = BK + kPad;
  constexpr int kThreads = WAVES_M * WAVES_N * kWave;
  constexpr int TM = BM / WAVES_M, TN = BN / WAVES_N;
  constexpr int FM = TM / 16, FN = TN / 16;
  constexpr int kChunksA = BM * BK / 8;       // 16-byte bf16 chunks of A
  constexpr int kChunksB = BN * BK / kBlock;  // 16-byte FP4 chunks (32 k)
  constexpr int kLoadsA = kChunksA / kThreads, kLoadsB = kChunksB / kThreads;
  static_assert(kChunksA % kThreads == 0 && kChunksB % kThreads == 0);

  __shared__ __attribute__((aligned(16))) uint16_t s_a[BM * kPitch];
  __shared__ __attribute__((aligned(16))) uint16_t s_b[BN * kPitch];

  // N tiles fastest: blocks running together share the A rows of one M tile.
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int wm = wave % WAVES_M, wn = wave / WAVES_M;

  uint4 ra[kLoadsA];
  uint4 rb[kLoadsB];
  int rs[kLoadsB];  // raw e8m0 bytes of the staged weight chunks

  auto load = [&](int k0) {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads, r = ch / (BK / 8), c8 = (ch % (BK / 8)) * 8;
      uint4 v = {};
      if (m0 + r < M) v = *reinterpret_cast<const uint4*>(a + int64_t(m0 + r) * lda + k0 + c8);
      ra[i] = v;
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads, r = ch / (BK / kBlock);
      const int k32 = (ch % (BK / kBlock)) * kBlock;
      uint4 v = {};
      int s = 0;
      if (n0 + r < N) {
        v = *reinterpret_cast<const uint4*>(
            w + int64_t(n0 + r) * ldw + (k0 + k32) / 2);
        s = scales[int64_t(n0 + r) * lds_n + (k0 + k32) / kBlock];
      }
      rb[i] = v;
      rs[i] = s;
    }
  };
  auto store = [&]() {
#pragma unroll
    for (int i = 0; i < kLoadsA; ++i) {
      const int ch = tid + i * kThreads, r = ch / (BK / 8), c8 = (ch % (BK / 8)) * 8;
      *reinterpret_cast<uint4*>(s_a + r * kPitch + c8) = ra[i];
    }
#pragma unroll
    for (int i = 0; i < kLoadsB; ++i) {
      const int ch = tid + i * kThreads, r = ch / (BK / kBlock);
      const int k32 = (ch % (BK / kBlock)) * kBlock;
      const uint32_t keep = rs[i] ? 0xffffu : 0u;
      // e8m0 -> bf16 exponent: 0.5 * 2^(s - 127) is 0x3F00 shifted by s - 127.
      // Signed so that s < 127 subtracts; the uint32 add wraps correctly.
      const uint32_t base = 0x3F00u + uint32_t((rs[i] - 127) * 128);
      const uint32_t word[4] = {rb[i].x, rb[i].y, rb[i].z, rb[i].w};
      uint32_t o[16];
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        o[4 * h] = mxfp4_pair(word[h] & 0xffu, base, keep);
        o[4 * h + 1] = mxfp4_pair((word[h] >> 8) & 0xffu, base, keep);
        o[4 * h + 2] = mxfp4_pair((word[h] >> 16) & 0xffu, base, keep);
        o[4 * h + 3] = mxfp4_pair(word[h] >> 24, base, keep);
      }
      // 32 values = 4 uint4 stores; the row pitch keeps them 16-byte aligned.
      uint32_t* dst = reinterpret_cast<uint32_t*>(s_b + r * kPitch + k32);
      *reinterpret_cast<uint4*>(dst) = make_uint4(o[0], o[1], o[2], o[3]);
      *reinterpret_cast<uint4*>(dst + 4) = make_uint4(o[4], o[5], o[6], o[7]);
      *reinterpret_cast<uint4*>(dst + 8) = make_uint4(o[8], o[9], o[10], o[11]);
      *reinterpret_cast<uint4*>(dst + 12) =
          make_uint4(o[12], o[13], o[14], o[15]);
    }
  };

  f32x8 acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) acc[i][j] = f32x8{};

  // STAGGER starts every CTA at a different K chunk and wraps around: the sum
  // over K does not care about the order, and rows ldw bytes apart that is a
  // power of two (K = 8192 -> 4096 B, K = 16384 -> 8192 B) otherwise camp on
  // the same DRAM channel for the whole loop - measured 21 GB/s at x1 against
  // 138 GB/s for the non-aliasing K = 5376 (bench/kernels/mxfp4_gemm.py).
  const int nch = K / BK;
  const int stagger = STAGGER ? (blockIdx.x * 7 + blockIdx.y * 3) % nch : 0;
  int k0 = stagger * BK;
  int knext = k0 + BK < K ? k0 + BK : 0;
  load(k0);
  store();
  __syncthreads();
  for (int done = 0; done < nch; ++done) {
    const bool more = done + 1 < nch;
    if (more) load(knext);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
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
    k0 = knext;
    knext = k0 + BK < K ? k0 + BK : 0;
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
        if (bias)
          v += __uint_as_float(uint32_t(bias[int64_t(m) * ldb + n]) << 16);
        c[int64_t(m) * ldc + n] = static_cast<uint16_t>(pack_bf16x2(v, 0.f) & 0xffffu);
      }
    }
}

// Whole-matrix decode to bf16 [N, K]: one thread per 16-byte chunk (32
// values, one e8m0 scale), the same exact nibble decode as the GEMM staging.
// For large M the plugin decodes each weight once and hands it to hipBLASLt
// (~40 TF) instead of re-decoding every weight tile once per 128-row M block.
__global__ __launch_bounds__(256) void mxfp4_dequant_kernel(
    uint16_t* __restrict__ out, const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ scales, int N, int K, int64_t ldw, int64_t lds_n) {
  const int64_t ch = int64_t(blockIdx.x) * 256 + threadIdx.x;
  const int per_row = K / kBlock;
  if (ch >= int64_t(N) * per_row) return;
  const int r = int(ch / per_row), b = int(ch % per_row);
  const uint4 v = *reinterpret_cast<const uint4*>(w + int64_t(r) * ldw + int64_t(b) * 16);
  const int sc = scales[int64_t(r) * lds_n + b];
  const uint32_t keep = sc ? 0xffffu : 0u;
  const uint32_t base = 0x3F00u + uint32_t((sc - 127) * 128);
  const uint32_t word[4] = {v.x, v.y, v.z, v.w};
  uint32_t o[16];
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    o[4 * h] = mxfp4_pair(word[h] & 0xffu, base, keep);
    o[4 * h + 1] = mxfp4_pair((word[h] >> 8) & 0xffu, base, keep);
    o[4 * h + 2] = mxfp4_pair((word[h] >> 16) & 0xffu, base, keep);
    o[4 * h + 3] = mxfp4_pair(word[h] >> 24, base, keep);
  }
  uint4* dst = reinterpret_cast<uint4*>(out + int64_t(r) * K + int64_t(b) * kBlock);
  dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
  dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
  dst[2] = make_uint4(o[8], o[9], o[10], o[11]);
  dst[3] = make_uint4(o[12], o[13], o[14], o[15]);
}

}  // namespace

// Launch one tile config; the host entry point picks it from M (tile-by-batch
// rule) unless the bench entry point pins one.
template <int BM, int BN, int WAVES_M, int WAVES_N, int BK, bool STAGGER>
static void mxfp4_launch(const at::Tensor& x, const at::Tensor& w,
                         const at::Tensor& scales,
                         const std::optional<at::Tensor>& bias,
                         at::Tensor& c, int64_t ldb) {
  const int m = (int)x.size(0), n = (int)w.size(0), k = (int)x.size(1);
  mxfp4_gemm_kernel<BM, BN, WAVES_M, WAVES_N, BK, STAGGER>
      <<<dim3((unsigned)((n + BN - 1) / BN), (unsigned)((m + BM - 1) / BM)),
         WAVES_M * WAVES_N * kWave, 0, at::hip::getCurrentHIPStream()>>>(
          reinterpret_cast<uint16_t*>(c.data_ptr()),
          reinterpret_cast<const uint16_t*>(x.data_ptr()),
          reinterpret_cast<const uint8_t*>(w.data_ptr()),
          reinterpret_cast<const uint8_t*>(scales.data_ptr()),
          bias ? reinterpret_cast<const uint16_t*>(bias->data_ptr()) : nullptr,
          m, n, k, x.stride(0), w.stride(0), scales.stride(0), ldb, c.stride(0));
}

// Bias row stride: 0 for a [N] bias, the row stride for [M, N].
static int64_t mxfp4_ldb(const std::optional<at::Tensor>& bias, int64_t m,
                         int64_t n) {
  if (!bias) return 0;
  TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be bf16");
  if (bias->dim() == 1) {
    TORCH_CHECK(bias->numel() == n && bias->stride(0) == 1,
                "bias must be [N=", n, "] contiguous");
    return 0;
  }
  TORCH_CHECK(bias->dim() == 2 && bias->size(0) == m && bias->size(1) == n &&
                  bias->stride(1) == 1,
              "bias must be bf16 [N] or [M, N]");
  return bias->stride(0);
}

// x: [M, K] bf16 (contiguous rows); w: uint8 [N, K/2] packed e2m1 nibbles
// (low nibble = even k, rows may be padded); scales: uint8 [N, K/32] e8m0.
// bias: bf16 [N] or [M, N] or none. Returns [M, N] bf16.
static torch::Tensor mxfp4_gemm_impl(const torch::Tensor& w,
                                    const torch::Tensor& x,
                                    const torch::Tensor& scales,
                                    const std::optional<torch::Tensor>& bias,
                                    int64_t cfg) {
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1 &&
                  x.stride(0) % 8 == 0,
              "x must be bf16 [M, K] with 16-byte aligned rows");
  TORCH_CHECK(w.scalar_type() == at::kByte && w.dim() == 2 && w.stride(1) == 1 &&
                  w.stride(0) % 16 == 0,
              "w must be uint8 [N, K/2] with 16-byte aligned rows");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  TORCH_CHECK(w.size(1) * 2 == k, "K mismatch: w is [N, ", w.size(1),
              "] bytes, x is [M, ", k, "]");
  TORCH_CHECK(k % kBK == 0, "K must be a multiple of ", kBK, " (got ", k, ")");
  TORCH_CHECK(scales.scalar_type() == at::kByte && scales.dim() == 2 &&
                  scales.size(0) == n && scales.size(1) == k / kBlock &&
                  scales.stride(1) == 1,
              "scale must be uint8 [N, K/32] = [", n, ", ", k / kBlock, "]");
  const int64_t ldb = mxfp4_ldb(bias, m, n);
  auto c = torch::empty({m, n}, x.options());
  if (m == 0 || n == 0) return c;

#define STRIX_MX(BM, BN, WM, WN) \
  mxfp4_launch<BM, BN, WM, WN, 64, true>(x, w, scales, bias, c, ldb)
#define STRIX_MXK(BM, BN, WM, WN, BK) \
  mxfp4_launch<BM, BN, WM, WN, BK, true>(x, w, scales, bias, c, ldb)
#define STRIX_MXKS(BM, BN, WM, WN, BK, ST) \
  mxfp4_launch<BM, BN, WM, WN, BK, ST>(x, w, scales, bias, c, ldb)
  switch (cfg) {
    case 0: STRIX_MX(16, 128, 1, 4); return c;   // decode
    case 1: STRIX_MX(16, 64, 1, 4); return c;    // decode, more CTAs
    case 2: STRIX_MX(32, 128, 2, 4); return c;   // batch 17..32
    case 3: STRIX_MX(64, 128, 2, 4); return c;   // batch 33..64
    case 4: STRIX_MX(128, 128, 4, 2); return c;  // prefill
    case 5: STRIX_MX(128, 256, 4, 4); return c;  // prefill, wider
    case 6: STRIX_MX(64, 256, 2, 8); return c;   // batch 33..64, wider
    case 7: STRIX_MX(128, 128, 2, 4); return c;  // prefill, taller wave
    case 8: STRIX_MX(16, 256, 1, 4); return c;   // decode, wider
    case 9: STRIX_MX(32, 64, 1, 4); return c;    // decode, BM 32 BN 64
    case 10: STRIX_MX(16, 64, 1, 2); return c;   // decode, 2 waves 2 chunks
    case 11: STRIX_MX(16, 128, 1, 2); return c;  // decode, 2 waves x 4 chunks
    case 12: STRIX_MX(64, 64, 1, 4); return c;   // decode, BM 64 BN 64
    case 13: STRIX_MX(16, 32, 1, 2); return c;   // decode, BM 16 BN 32
    case 14: STRIX_MXK(16, 32, 1, 2, 128); return c;   // decode, BK 128
    case 15: STRIX_MXK(16, 64, 1, 4, 128); return c;   // decode, BK 128
    case 16: STRIX_MXK(16, 64, 1, 2, 128); return c;   // decode, BK 128, 2 waves
    case 17: STRIX_MXK(32, 64, 1, 4, 128); return c;   // decode, BM 32, BK 128
    case 18: STRIX_MXK(16, 128, 1, 4, 128); return c;  // decode, BK 128 wide
    case 19: STRIX_MXK(64, 64, 1, 4, 128); return c;   // BM 64, BK 128
    case 20: STRIX_MXK(16, 32, 1, 2, 256); return c;   // decode, BK 256
    case 21: STRIX_MXK(16, 16, 1, 1, 128); return c;   // decode, 32 threads
    case 22: STRIX_MXK(16, 32, 1, 1, 128); return c;   // decode, 1 wave
    case 23: STRIX_MXK(16, 64, 1, 4, 256); return c;   // decode, BK 256 wide
    case 24: STRIX_MXK(16, 32, 1, 2, 512); return c;   // decode, BK 512
    case 25: STRIX_MXKS(16, 32, 1, 2, 128, false); return c;  // no K stagger
    case 26: STRIX_MXKS(128, 128, 2, 4, 64, false); return c;  // prefill, no stagger
    case 27: STRIX_MXKS(32, 64, 1, 4, 128, false); return c;   // M 32, no stagger
    case 28: STRIX_MXKS(64, 64, 1, 4, 128, false); return c;   // M 64, no stagger
    default: break;
  }
  // Tile by batch, as in w8a8_wmma_gemm: the smallest tile that still covers M
  // in ONE m-block, because a second m-block re-reads the whole weight matrix.
  // Here the weight is read in place at 4 bits, so at decode M the tile is pure
  // occupancy and in-flight bytes: 16 rows x 32 cols staging 128 K per step
  // runs the model's 240 linears in 116 ms against 194 ms for the 64-K steps
  // and 4-wave tiles of the w4a16 skeleton (bench/kernels/mxfp4_gemm.py sweeps
  // the 32/64-row and prefill tiles too). Past M 64 the bf16 WMMA math is the
  // cost instead: 128 x 128 with a 2 x 4 wave split runs 28 TFLOPS at M 512,
  // where hipBLASLt bf16 reaches 43.
  if (m <= 16) STRIX_MXK(16, 32, 1, 2, 128);
  else if (m <= 32) STRIX_MX(32, 128, 2, 4);
  else if (m <= 64) STRIX_MX(64, 128, 2, 4);
  else STRIX_MX(128, 128, 2, 4);
#undef STRIX_MX
  return c;
}

torch::Tensor mxfp4_wmma_gemm(const torch::Tensor& w, const torch::Tensor& x,
                              const torch::Tensor& scales,
                              const std::optional<torch::Tensor>& bias) {
  return mxfp4_gemm_impl(w, x, scales, bias, -1);
}

// Bench-only: the same GEMM on one of the fixed tile configs in
// bench/kernels/mxfp4_gemm.py instead of the tile-by-batch choice.
torch::Tensor mxfp4_wmma_gemm_cfg(const torch::Tensor& w, const torch::Tensor& x,
                                  const torch::Tensor& scales,
                                  const std::optional<torch::Tensor>& bias,
                                  int64_t cfg) {
  return mxfp4_gemm_impl(w, x, scales, bias, cfg);
}

// w: uint8 [N, K/2] packed e2m1 nibbles (rows may be padded); scales: uint8
// [N, K/32] e8m0. Returns the bf16 [N, K] matrix the GEMM computes against.
torch::Tensor mxfp4_dequant(const torch::Tensor& w, const torch::Tensor& scales) {
  TORCH_CHECK(w.scalar_type() == at::kByte && w.dim() == 2 && w.stride(1) == 1 &&
                  w.stride(0) % 16 == 0,
              "w must be uint8 [N, K/2] with 16-byte aligned rows");
  const int64_t n = w.size(0), k = w.size(1) * 2;
  TORCH_CHECK(k % kBlock == 0, "K must be a multiple of ", kBlock);
  TORCH_CHECK(scales.scalar_type() == at::kByte && scales.dim() == 2 &&
                  scales.size(0) == n && scales.size(1) == k / kBlock &&
                  scales.stride(1) == 1,
              "scale must be uint8 [N, K/32]");
  auto out = torch::empty({n, k}, w.options().dtype(at::kBFloat16));
  const int64_t chunks = n * (k / kBlock);
  if (chunks == 0) return out;
  mxfp4_dequant_kernel<<<dim3(unsigned((chunks + 255) / 256)), 256, 0,
                         at::hip::getCurrentHIPStream()>>>(
      reinterpret_cast<uint16_t*>(out.data_ptr()), w.data_ptr<uint8_t>(),
      scales.data_ptr<uint8_t>(), int(n), int(k), w.stride(0), scales.stride(0));
  return out;
}

}  // namespace strix
