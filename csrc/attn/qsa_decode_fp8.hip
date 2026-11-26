// SPDX-License-Identifier: MIT
// Qwen3.8-Flash-Next QSA sparse attention at decode / verify batch sizes on an
// fp8 (e4m3, unit scale) cache: each query row attends over its own selected
// tokens (up to ~2051), GQA 12, head_dim 256, two KV heads.
//
// The WMMA form of qsa_rows_bf16.hip (a row's 12 query heads as the WMMA
// columns over 16 of its tokens a step), reading the fp8 cache directly, with
// each row's selection split over blocks so that a few rows still fill the
// GPU; qsa_decode_merge_kernel combines the partial softmaxes. 2.3-4.8x
// vLLM's Triton kernel on its fp8 cache at 1-16 rows (5 rows, one request
// verifying 4 drafts: 75 against 180 us per layer).
//
// RDNA 3.5 has no fp8 convert: e4m3 is widened with integer ops (normals by
// moving the bits and adding the exponent bias, subnormals through the
// ubyte -> float convert). A vector-unit version (lanes over head_dim, the 12
// heads' scores reduced across the wave per token) was 1.3-3x Triton.
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include <algorithm>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kQD = 256;       // head_dim
constexpr int kQG = 12;        // query heads per KV head
constexpr int kQPerLane = kQD / kWave;  // 8 head_dim values per lane

// Four e4m3 bytes of `w` -> four floats. FNUZ: the gfx942 flavor (bias 8).
template <bool FNUZ>
__device__ __forceinline__ void e4m3x4_to_f32(uint32_t w, float* f) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t b = (w >> (8 * i)) & 0xffu;
    const uint32_t s = (b & 0x80u) << 24;
    const uint32_t em = b & 0x7fu;
    // Normal: exponent e at bits 3-6, mantissa at 0-2 -> fp32 bits 23-26 /
    // 20-22, plus the bias difference (127 - 7, or 127 - 8 for fnuz).
    const float normal = __uint_as_float(s | ((em << 20) + ((FNUZ ? 119u : 120u) << 23)));
    // Subnormal (e == 0): m * 2^-9 (2^-10 fnuz).
    float sub = static_cast<float>(em) * (FNUZ ? 0.0009765625f : 0.001953125f);
    sub = s ? -sub : sub;
    f[i] = em >= 8 ? normal : sub;
  }
}

// Four e4m3 bytes of `w` -> four bf16 (exact: e4m3 has 3 mantissa bits).
template <bool FNUZ>
__device__ __forceinline__ void e4m3x4_to_bf16(uint32_t w, short* o) {
  float f[4];
  e4m3x4_to_f32<FNUZ>(w, f);
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = static_cast<short>(__float_as_uint(f[i]) >> 16);
}

// The WMMA form (qsa_rows_bf16.hip's per-row kernel) on the fp8 cache, the
// selection split over blocks: block (split, row), one wave pair per KV head,
// each row's 12 query heads as the WMMA columns over 16 of its tokens a step.
// K fragments widen 16 fp8 bytes per lane in registers; V widens while it is
// staged as V^T in LDS. Each block writes unnormalized partials (in the log2
// domain of the scores) for qsa_decode_merge_kernel.
template <bool FNUZ>
__global__ __launch_bounds__(2 * 2 * kWave) __attribute__((amdgpu_waves_per_eu(4)))
void qsa_decode_wmma_kernel(
    float* __restrict__ ws_o, float* __restrict__ ws_ml, const uint16_t* __restrict__ q,
    const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const int32_t* __restrict__ idx, int64_t ld_idx, int width,
    const int32_t* __restrict__ bt, int64_t ld_bt, const int32_t* __restrict__ tok2req,
    int block_size, int64_t sq_t, int64_t sq_h, int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h, int splits, float scale_log2) {
  constexpr int D = kQD, G = kQG, BN = 16, kHalf = 128, W = 2, kPV = 16;
  __shared__ __attribute__((aligned(16))) uint16_t s_vt[2][D * kPV];
  __shared__ _Float16 s_x[2 * W][kWave][8];

  const int split = blockIdx.x, row = blockIdx.y;
  const int tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
  const int pair = wave / W, half = wave % W, lo = lane % 16, hi = lane / 16;
  const int kvh = pair;  // both KV heads of the row in one block
  const int ptid = tid % (W * kWave);
  const int req = tok2req ? tok2req[row] : 0;
  const int32_t* brow = bt + int64_t(req) * ld_bt;
  const int32_t* irow = idx + int64_t(row) * ld_idx;
  const int h = kvh * G + (lo < G ? lo : 0);

  int n_valid = 0;
  for (int base = ((width - 1) / kWave) * kWave; base >= 0; base -= kWave) {
    const int k = base + lane;
    const uint64_t m = __ballot(k < width && irow[k] >= 0);
    if (m) {
      n_valid = base + 64 - __builtin_clzll(m);
      break;
    }
  }
  // This block's range of the selection, whole 16-key steps.
  const int per = ((n_valid + splits - 1) / splits + BN - 1) / BN * BN;
  const int k_lo = min(n_valid, split * per), k_hi = min(n_valid, k_lo + per);
  const int n_steps = (k_hi - k_lo + BN - 1) / BN;

  bf16x16 qf[kHalf / 16];
  {
    const uint16_t* qrow = q + int64_t(row) * sq_t + h * sq_h + half * kHalf;
#pragma unroll
    for (int kk = 0; kk < kHalf / 16; ++kk) qf[kk] = load_frag_bf16(qrow + kk * 16, 0, 0);
  }
  f32x8 o_acc[kHalf / 16];
#pragma unroll
  for (int i = 0; i < kHalf / 16; ++i) o_acc[i] = f32x8{};
  float m_run = -INFINITY, l_run = 0.f;
  const uint32_t p_sel = hi ? 0x03020706u : 0x07060302u;

  // V: 16 keys x 256 d = 256 16-byte fp8 chunks over the pair's 64 threads,
  // a thread's 4 chunks all of one key.
  constexpr int kLoads = BN * D / 16 / (W * kWave);
  uint4 rv[kLoads];
  auto token = [&](int j) { return k_lo + j < k_hi ? irow[k_lo + j] : -1; };
  auto page_of = [&](int t) { return t >= 0 ? brow[t / block_size] : 0; };
  const int vkey = ptid % BN;
  auto fetch_v = [&](int t, int pg) {
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      const int ch = ptid + i * W * kWave;
      const int vd = (ch / BN) * 16;
      uint4 x = {};
      if (t >= 0)
        x = *reinterpret_cast<const uint4*>(vc + int64_t(pg) * sv_b +
                                            int64_t(t % block_size) * sv_s + kvh * sv_h + vd);
      rv[i] = x;
    }
  };
  int vt = token(vkey), vpg = page_of(vt);
  if (n_steps > 0) fetch_v(vt, vpg);
  vt = token(BN + vkey);
  vpg = page_of(vt);
  bf16x16 kf[kHalf / 16];
  auto load_k = [&](int t, int pg) {
    const uint8_t* kp = t >= 0 ? kc + int64_t(pg) * sk_b + int64_t(t % block_size) * sk_s +
                                     kvh * sk_h + half * kHalf
                               : nullptr;
#pragma unroll
    for (int kk = 0; kk < kHalf / 16; ++kk) {
      if (kp) {
        const uint4 raw = *reinterpret_cast<const uint4*>(kp + kk * 16);
        short tmp[16];
        e4m3x4_to_bf16<FNUZ>(raw.x, tmp);
        e4m3x4_to_bf16<FNUZ>(raw.y, tmp + 4);
        e4m3x4_to_bf16<FNUZ>(raw.z, tmp + 8);
        e4m3x4_to_bf16<FNUZ>(raw.w, tmp + 12);
#pragma unroll
        for (int i = 0; i < 16; ++i) kf[kk][i] = tmp[i];
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) kf[kk][i] = 0;
      }
    }
  };
  int kt = token(lo), kpg = page_of(kt);
  for (int step = 0; step < n_steps; ++step) {
    const int k0 = step * BN;
    load_k(kt, kpg);
    barrier();
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
      const int ch = ptid + i * W * kWave;
      const int vd = (ch / BN) * 16;
      const uint32_t w[4] = {rv[i].x, rv[i].y, rv[i].z, rv[i].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        short b4[4];
        e4m3x4_to_bf16<FNUZ>(w[e], b4);
#pragma unroll
        for (int x = 0; x < 4; ++x)
          s_vt[pair][(vd + 4 * e + x) * kPV + vkey] = static_cast<uint16_t>(b4[x]);
      }
    }
    kt = token(k0 + BN + lo);
    kpg = page_of(kt);
    if (step + 1 < n_steps) {
      fetch_v(vt, vpg);
      vt = token(k0 + 2 * BN + vkey);
      vpg = page_of(vt);
    }

    f32x8 sc = f32x8{};
#pragma unroll
    for (int kk = 0; kk < kHalf / 16; ++kk) sc = wmma_bf16(kf[kk], qf[kk], sc);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      sc[e] *= scale_log2;
      s_x[wave][lane][e] = static_cast<_Float16>(sc[e]);
    }
    barrier();

    float m_loc = -INFINITY;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float v = sc[e];
#pragma unroll
      for (int o = 1; o < W; ++o)
        v += static_cast<float>(s_x[pair * W + (half + o) % W][lane][e]);
      const bool seen = token(k0 + 2 * e + hi) >= 0;
      sc[e] = seen ? v : -INFINITY;
      m_loc = fmaxf(m_loc, sc[e]);
    }
    m_loc = fmaxf(m_loc, xor_shuffle_f32(m_loc, 16));
    const float m_new = fmaxf(m_run, m_loc);
    const float m_ref = m_new == -INFINITY ? 0.f : m_new;
    const float alpha = __builtin_amdgcn_exp2f(m_run - m_ref);
    if (__any(alpha != 1.f)) {
#pragma unroll
      for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
        for (int e = 0; e < 8; ++e) o_acc[dt][e] *= alpha;
    }
    float l_loc = 0.f;
    bf16x16 pf;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float p = __builtin_amdgcn_exp2f(sc[e] - m_ref);
      l_loc += p;
      const uint32_t mine = round_bf16_hi(p);
      const uint32_t other = __float_as_uint(xor_shuffle_f32(__uint_as_float(mine), 16));
      const uint32_t pk = __builtin_amdgcn_perm(other, mine, p_sel);
      pf[2 * e] = static_cast<short>(pk & 0xffffu);
      pf[2 * e + 1] = static_cast<short>(pk >> 16);
    }
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt) {
      const bf16x16 vf =
          load_frag_bf16(&s_vt[pair][(half * kHalf + dt * 16) * kPV], kPV, lane);
      o_acc[dt] = wmma_bf16(vf, pf, o_acc[dt]);
    }
    l_loc += xor_shuffle_f32(l_loc, 16);
    l_run = l_run * alpha + l_loc;
    m_run = m_new;
  }

  if (lo < G) {
    const int64_t slot = (int64_t(row) * (2 * G) + h) * splits + split;
    float* op = ws_o + slot * D + half * kHalf;
#pragma unroll
    for (int dt = 0; dt < kHalf / 16; ++dt)
#pragma unroll
      for (int e = 0; e < 8; ++e) op[dt * 16 + 2 * e + hi] = o_acc[dt][e];
    if (half == 0 && hi == 0) {
      ws_ml[slot * 2] = m_run;
      ws_ml[slot * 2 + 1] = l_run;
    }
  }
}

// out[row, head, :] = sum_p exp2(m_p - M) o_p / sum_p exp2(m_p - M) l_p.
__global__ __launch_bounds__(kQD) void qsa_decode_merge_kernel(
    uint16_t* __restrict__ out, const float* __restrict__ ws_o,
    const float* __restrict__ ws_ml, int parts, int64_t so_t, int64_t so_h, int heads) {
  const int row = blockIdx.x, head = blockIdx.y, d = threadIdx.x;
  const int64_t slot0 = (int64_t(row) * heads + head) * parts;
  float m = -INFINITY;
  for (int p = 0; p < parts; ++p) m = fmaxf(m, ws_ml[(slot0 + p) * 2]);
  float l = 0.f, acc = 0.f;
  if (m != -INFINITY) {
    for (int p = 0; p < parts; ++p) {
      const float w = __builtin_amdgcn_exp2f(ws_ml[(slot0 + p) * 2] - m);
      l += w * ws_ml[(slot0 + p) * 2 + 1];
      acc += w * ws_o[(slot0 + p) * kQD + d];
    }
  }
  const float v = l > 0.f ? acc / l : 0.f;
  out[int64_t(row) * so_t + int64_t(head) * so_h + d] =
      static_cast<uint16_t>(pack_bf16x2(v, 0.f) & 0xffffu);
}

}  // namespace

// out, q: [rows, 24, 256] bf16 (Hq = 12 * Hkv, Hkv = 2); key_cache /
// value_cache: fp8 e4m3 (unit scale) [num_blocks, block_size, 2, 256] views;
// indices: [rows, width] int32 request-relative positions, -1 padded;
// block_table: [reqs, max_blocks] int32; token_to_req: [rows] int32 or None.
void qsa_decode_fp8(torch::Tensor& out, const torch::Tensor& q,
                    const torch::Tensor& key_cache, const torch::Tensor& value_cache,
                    const torch::Tensor& indices, const torch::Tensor& block_table,
                    const std::optional<torch::Tensor>& token_to_req) {
  const auto kt = key_cache.scalar_type();
  const bool fnuz = kt == at::kFloat8_e4m3fnuz;
  TORCH_CHECK(kt == at::kFloat8_e4m3fn || fnuz, "qsa_decode_fp8: e4m3 caches");
  TORCH_CHECK(value_cache.scalar_type() == kt, "K and V caches of one dtype");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
              "bf16 q and out");
  TORCH_CHECK(indices.scalar_type() == at::kInt && block_table.scalar_type() == at::kInt,
              "int32 indices and block table");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kQD && q.stride(2) == 1 && out.stride(2) == 1 &&
                  q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0,
              "q / out [rows, Hq, 256], contiguous head_dim, 16-byte aligned heads");
  TORCH_CHECK(key_cache.dim() == 4 && key_cache.size(3) == kQD &&
                  key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
                  key_cache.stride(2) % 16 == 0 && value_cache.stride(2) % 16 == 0 &&
                  key_cache.stride(1) % 16 == 0 && value_cache.stride(1) % 16 == 0 &&
                  key_cache.stride(0) % 16 == 0 && value_cache.stride(0) % 16 == 0,
              "caches [blocks, block_size, Hkv, 256], 16-byte aligned heads");
  const int64_t rows = q.size(0), hkv = key_cache.size(2);
  TORCH_CHECK(hkv == 2 && q.size(1) == kQG * hkv, "GQA 12 over two KV heads");
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == rows && indices.stride(1) == 1,
              "indices [rows, width]");
  TORCH_CHECK(block_table.dim() == 2 && block_table.stride(1) == 1, "block table 2-D");
  const int32_t* t2r = nullptr;
  if (token_to_req.has_value()) {
    TORCH_CHECK(token_to_req->scalar_type() == at::kInt && token_to_req->dim() == 1 &&
                    token_to_req->size(0) >= rows && token_to_req->stride(0) == 1,
                "token_to_req [rows] int32");
    t2r = token_to_req->data_ptr<int32_t>();
  } else {
    TORCH_CHECK(block_table.size(0) == 1, "several requests need token_to_req");
  }
  if (rows == 0) return;
  // Blocks of both KV heads; about six per compute unit, at least one 16-token
  // step each.
  const int width = int(indices.size(1));
  const int splits_w = int(std::max<int64_t>(
      1, std::min<int64_t>({64, 240 / rows, (width + 31) / 32})));
  auto opts_w = q.options().dtype(at::kFloat);
  auto wo = at::empty({rows * kQG * hkv * splits_w * kQD}, opts_w);
  auto wml = at::empty({rows * kQG * hkv * splits_w * 2}, opts_w);
  auto st = at::hip::getCurrentHIPStream();
  const float sl2 = 1.4426950408889634f / 16.f;
  auto kw = fnuz ? qsa_decode_wmma_kernel<true> : qsa_decode_wmma_kernel<false>;
  kw<<<dim3(splits_w, rows), 2 * 2 * kWave, 0, st>>>(
      wo.data_ptr<float>(), wml.data_ptr<float>(),
      reinterpret_cast<const uint16_t*>(q.data_ptr()),
      reinterpret_cast<const uint8_t*>(key_cache.data_ptr()),
      reinterpret_cast<const uint8_t*>(value_cache.data_ptr()),
      indices.data_ptr<int32_t>(), indices.stride(0), width,
      block_table.data_ptr<int32_t>(), block_table.stride(0), t2r, int(key_cache.size(1)),
      q.stride(0), q.stride(1), key_cache.stride(0), key_cache.stride(1),
      key_cache.stride(2), value_cache.stride(0), value_cache.stride(1),
      value_cache.stride(2), splits_w, sl2);
  qsa_decode_merge_kernel<<<dim3(rows, kQG * hkv), kQD, 0, st>>>(
      reinterpret_cast<uint16_t*>(out.data_ptr()), wo.data_ptr<float>(),
      wml.data_ptr<float>(), splits_w, out.stride(0), out.stride(1), int(kQG * hkv));
}

}  // namespace strix
