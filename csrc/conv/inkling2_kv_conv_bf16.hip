// SPDX-License-Identifier: MIT
// Inkling 2 attention-prologue K / V short convolutions (W = 4
// taps, static or dynamic) over the port's paged conv-state cache ([blocks,
// H, N, Dpack], one slot per absolute position): K is then L2-clipped and V
// RMS-normalized per head and both go to the paged KV cache (bf16 or int8);
// each token's input goes to its conv-cache slot.
//
// One wave owns a KV head (K in lanes 0-15, V in 16-31, 8 channels a lane)
// for a run of kBT tokens and loads the run's input rows (and the three
// before it) once; taps before the request's first row in this forward come
// from the cache (chunked prefill), taps before position 0 are zero. The
// math follows the port's Triton kernel (vllm/models/inkling2/amd/ops/
// sconv.py) term by term; only the order of the per-head sum of squares
// differs. 5k-token prefill on gfx1151: 1.15-1.19 -> 0.78-0.85 ms a layer.
// (A hidden-stream variant only matched the Triton short conv: both are
// bound by the scattered conv-cache row writes.)
#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "../strix_common.h"

namespace strix {
namespace {

constexpr int kW = 4;     // taps
constexpr int kBT = 8;    // tokens per wave

struct ConvMeta {
  const int64_t* slot;    // [T] conv-cache slot (< 0: padding, no taps)
  const int64_t* pos;     // [T] absolute position
  const int32_t* seq;     // [T] request
  const int32_t* qstart;  // [T] first row of the token's request
  const int32_t* bt;      // [reqs, max_blocks] conv-cache blocks
  int64_t bt_stride;
  int max_blocks;
  int n_shift;            // log2 of the conv-cache block size
  uint16_t* cache;
  int64_t c_blk, c_n;
};

constexpr int kNV = 8;  // channels per lane (16-byte accesses)

__device__ __forceinline__ void bf16x8_to_f32(uint4 v, float f[kNV]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) f[2 * i] = bf16_lo(w[i]), f[2 * i + 1] = bf16_hi(w[i]);
}

__device__ __forceinline__ uint4 f32_to_bf16x8(const float f[kNV]) {
  return make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]), pack_bf16x2(f[4], f[5]),
                    pack_bf16x2(f[6], f[7]));
}

__device__ __forceinline__ void load_taps(const float* w, float wt[kNV][kW]) {
#pragma unroll
  for (int c = 0; c < kNV; ++c) {
    const float4 q = *reinterpret_cast<const float4*>(w + c * kW);
    wt[c][0] = q.x, wt[c][1] = q.y, wt[c][2] = q.z, wt[c][3] = q.w;
  }
}

// Tap of the token at position pos from input row `row` (= t - lag): the
// run's window copy `winv` if the row is this forward's, else the cache slot
// of position pos - lag (chunked prefill), else zero.
__device__ __forceinline__ uint4 tap(const ConvMeta& m, int row, int qs, int64_t src, bool valid,
                                     int req, int ccol, const uint4& winv) {
  if (!valid || src < 0) return make_uint4(0, 0, 0, 0);
  if (row >= qs) return winv;
  const int64_t lblk = min(src >> m.n_shift, int64_t(m.max_blocks - 1));
  const int64_t blk = m.bt[int64_t(req) * m.bt_stride + lblk];
  return *reinterpret_cast<const uint4*>(m.cache + blk * m.c_blk +
                                         (src & ((1 << m.n_shift) - 1)) * m.c_n + ccol);
}

// A run's loads, all issued before any store (the cache and output pointers
// may alias the inputs as far as the compiler knows, which serialized a
// per-token loop): win[j] is input row t0 - (kW - 1) + j (zero before row 0;
// rows past T repeat the last one).
struct RunMeta {
  int64_t slot[kBT], pos[kBT];
  int req[kBT], qs[kBT];
};

__device__ __forceinline__ void load_run(const ConvMeta& m, const uint16_t* x, int64_t sx,
                                         int xcol, int t0, int T, uint4 win[kW - 1 + kBT],
                                         RunMeta& r) {
#pragma unroll
  for (int j = 0; j < kW - 1 + kBT; ++j) {
    const int row = min(t0 - (kW - 1) + j, T - 1);
    win[j] = row >= 0 ? *reinterpret_cast<const uint4*>(x + int64_t(row) * sx + xcol)
                      : make_uint4(0, 0, 0, 0);
  }
#pragma unroll
  for (int i = 0; i < kBT; ++i) {
    const int t = min(t0 + i, T - 1);
    r.slot[i] = m.slot[t], r.pos[i] = m.pos[t], r.req[i] = m.seq[t], r.qs[i] = m.qstart[t];
  }
}

// The kW taps of run token i, newest first (xs[lag][channel]).
__device__ __forceinline__ void run_taps(const ConvMeta& m, const RunMeta& r, int i, int t0,
                                         int ccol, const uint4 win[kW - 1 + kBT],
                                         float xs[kW][kNV]) {
  const int t = t0 + i;
  const bool valid = r.slot[i] >= 0;
#pragma unroll
  for (int lag = 0; lag < kW; ++lag)
    bf16x8_to_f32(tap(m, t - lag, r.qs[i], r.pos[i] - lag, valid, r.req[i], ccol,
                      win[i + kW - 1 - lag]),
                  xs[lag]);
}

// K / V convs of the prologue: wave = KV head (lanes 0-15 K, 16-31 V),
// block = 4 waves. Static taps w[c][k] (oldest first, bf16-rounded); dynamic
// taps (w[c][lag] + dyn[t][head][lag], newest first). Q8: the KV cache is
// vLLM's int8_per_token_head layout - each (head, slot) content half is the
// 128 int8 values plus one fp32 scale at the end (amax / 127, floored at
// 1e-6, round half away from zero, clamped to [-128, 127], as vLLM's
// per-token-head quant kernel).
template <bool DYN, bool Q8>
__global__ __launch_bounds__(128) void kv_conv_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ v, int64_t sk, int64_t sv,
    const float* __restrict__ wk, const float* __restrict__ wv,
    const uint16_t* __restrict__ dk, const uint16_t* __restrict__ dv, int64_t sdk,
    int64_t sdv, ConvMeta m, int64_t c_h, uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache, const int64_t* __restrict__ kv_slot, int page_shift,
    int64_t kv_blk, int64_t kv_n, int64_t kv_h, int T, int heads, float scale,
    float clip, float clip_eps, float norm_eps) {
  constexpr int D = 128;
  const int lane = threadIdx.x % kWave;
  const int head = blockIdx.x * (blockDim.x / kWave) + threadIdx.x / kWave;
  if (head >= heads) return;
  const bool is_v = lane >= 16;
  const int d = (lane % 16) * kNV;
  const uint16_t* x = is_v ? v : k;
  const int64_t sx = is_v ? sv : sk;
  const uint16_t* dyn = is_v ? dv : dk;
  const int64_t sdyn = is_v ? sdv : sdk;
  const int xcol = head * D + d;
  const int ccol = int(head * c_h) + (is_v ? D : 0) + d;
  uint16_t* kvc = is_v ? value_cache : key_cache;
  float wt[kNV][kW];  // [channel][tap]
  load_taps((is_v ? wv : wk) + int64_t(xcol) * kW, wt);
  const int t0 = blockIdx.y * kBT;
  uint4 win[kW - 1 + kBT];
  RunMeta r;
  load_run(m, x, sx, xcol, t0, T, win, r);
  int64_t ks[kBT];
  uint2 dd[kBT];
#pragma unroll
  for (int i = 0; i < kBT; ++i) {
    const int t = min(t0 + i, T - 1);
    ks[i] = kv_slot[t];
    if constexpr (DYN) dd[i] = *reinterpret_cast<const uint2*>(dyn + int64_t(t) * sdyn + head * kW);
  }
#pragma unroll
  for (int i = 0; i < kBT; ++i) {
    if (t0 + i >= T) break;
    float xs[kW][kNV];
    run_taps(m, r, i, t0, ccol, win, xs);
    const uint4 cur = win[i + kW - 1];
    float cf[kNV];
    bf16x8_to_f32(cur, cf);
    float dl[kW];
    if constexpr (DYN)
      dl[0] = bf16_lo(dd[i].x), dl[1] = bf16_hi(dd[i].x), dl[2] = bf16_lo(dd[i].y),
      dl[3] = bf16_hi(dd[i].y);
    float y[kNV], ss = 0.f;
#pragma unroll
    for (int c = 0; c < kNV; ++c) {
      float acc;
      if constexpr (DYN) {
        acc = xs[0][c] * (wt[c][0] + dl[0]);
#pragma unroll
        for (int lag = 1; lag < kW; ++lag) acc = fmaf(xs[lag][c], wt[c][lag] + dl[lag], acc);
      } else {
        acc = 0.f;
#pragma unroll
        for (int kk = 0; kk < kW; ++kk) acc = fmaf(xs[kW - 1 - kk][c], wt[c][kk], acc);
      }
      // Rounded to bf16 like the input dtype.
      y[c] = bf16_hi(round_bf16_hi((acc + cf[c]) * scale) & 0xffff0000u);
      ss = fmaf(y[c], y[c], ss);
    }
#pragma unroll
    for (int s = 8; s >= 1; s >>= 1) ss += xor_shuffle_f32(ss, s);  // per half
    float f;
    if (is_v) {
      f = rsqrtf(__fdiv_rn(ss, float(D)) + norm_eps);
    } else {
      const float norm = ss > 0.f ? ss * rsqrtf(ss) : 0.f;
      f = fminf(__fdiv_rn(clip, norm + clip_eps), 1.f);
    }
#pragma unroll
    for (int c = 0; c < kNV; ++c) y[c] *= f;
    const int64_t slot = r.slot[i];
    if (slot >= 0)
      *reinterpret_cast<uint4*>(m.cache + (slot >> m.n_shift) * m.c_blk +
                                (slot & ((1 << m.n_shift) - 1)) * m.c_n + ccol) = cur;
    if (ks[i] >= 0) {
      const int64_t dst = (ks[i] >> page_shift) * kv_blk +
                          (ks[i] & ((1 << page_shift) - 1)) * kv_n + head * kv_h;
      if constexpr (Q8) {
        // The 16 lanes of this half-wave hold the head's 128 values: reduce
        // the amax like the sum of squares above.
        float am = 0.f;
#pragma unroll
        for (int c = 0; c < kNV; ++c) am = fmaxf(am, fabsf(y[c]));
#pragma unroll
        for (int s = 8; s >= 1; s >>= 1) am = fmaxf(am, xor_shuffle_f32(am, s));
        const float qscale = fmaxf(am / 127.f, 1e-6f);
        const float inv = 1.f / qscale;
        // The V rows are only 4-byte aligned (132 bytes after K's): two words.
        uint32_t qw[2] = {0, 0};
#pragma unroll
        for (int c = 0; c < kNV; ++c) {
          float z = y[c] * inv;
          z = z >= 0.f ? floorf(z + 0.5f) : ceilf(z - 0.5f);
          const int qi = static_cast<int>(fmaxf(-128.f, fminf(127.f, z)));
          qw[c >> 2] |= static_cast<uint32_t>(static_cast<uint8_t>(qi)) << (8 * (c & 3));
        }
        int8_t* kvc8 = reinterpret_cast<int8_t*>(kvc);
        *reinterpret_cast<uint32_t*>(kvc8 + dst + d) = qw[0];
        *reinterpret_cast<uint32_t*>(kvc8 + dst + d + 4) = qw[1];
        if (d == 0) *reinterpret_cast<float*>(kvc8 + dst + 128) = qscale;
      } else {
        *reinterpret_cast<uint4*>(kvc + dst + d) = f32_to_bf16x8(y);
      }
    }
  }
}

ConvMeta make_meta(const torch::Tensor& cache, const torch::Tensor& positions,
                   const torch::Tensor& block_table, const torch::Tensor& seq_idx,
                   const torch::Tensor& slot_mapping, const torch::Tensor& query_start) {
  TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && cache.dim() == 4 && cache.stride(3) == 1 &&
                  cache.stride(0) % 8 == 0 && cache.stride(1) % 8 == 0 &&
                  cache.stride(2) % 8 == 0,
              "conv cache: bf16 [blocks, H, N, D] with 16-byte aligned rows");
  TORCH_CHECK(positions.scalar_type() == at::kLong && slot_mapping.scalar_type() == at::kLong,
              "positions / slot_mapping: int64");
  TORCH_CHECK(seq_idx.scalar_type() == at::kInt && query_start.scalar_type() == at::kInt &&
                  block_table.scalar_type() == at::kInt && block_table.stride(1) == 1,
              "seq_idx / query_start / block_table: int32");
  ConvMeta m;
  m.slot = slot_mapping.data_ptr<int64_t>();
  m.pos = positions.data_ptr<int64_t>();
  m.seq = seq_idx.data_ptr<int32_t>();
  m.qstart = query_start.data_ptr<int32_t>();
  m.bt = block_table.data_ptr<int32_t>();
  m.bt_stride = block_table.stride(0);
  m.max_blocks = int(block_table.size(1));
  const int64_t n = cache.size(2);
  TORCH_CHECK(n > 0 && (n & (n - 1)) == 0, "conv-cache block size: a power of two");
  m.n_shift = __builtin_ctzll(n);
  m.cache = reinterpret_cast<uint16_t*>(cache.data_ptr());
  m.c_blk = cache.stride(0);
  m.c_n = cache.stride(2);
  return m;
}

}  // namespace

// k, v: bf16 [T, H * 128] (row-strided); wk, wv: fp32 [H * 128, 4] taps (static:
// oldest first, bf16-rounded; dynamic: the static part, newest first) with
// k_dyn / v_dyn bf16 [T, H * 4] per-token deltas (row-strided) when dynamic.
// key_cache / value_cache: bf16 or int8 [blocks, page, H, 128] views with
// equal strides; after each int8 row the kernel writes its fp32 scale.
void inkling2_kv_conv(const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& wk,
                      const torch::Tensor& wv, const std::optional<torch::Tensor>& k_dyn,
                      const std::optional<torch::Tensor>& v_dyn, const torch::Tensor& cache,
                      const torch::Tensor& key_cache, const torch::Tensor& value_cache,
                      const torch::Tensor& kv_slot_mapping, const torch::Tensor& positions,
                      const torch::Tensor& seq_idx, const torch::Tensor& slot_mapping,
                      const torch::Tensor& block_table, const torch::Tensor& query_start,
                      double scale, double clip_threshold, double clip_eps, double norm_eps) {
  const int64_t T = kv_slot_mapping.size(0);
  const int heads = int(key_cache.size(2));
  const bool q8 = key_cache.scalar_type() != at::kBFloat16;
  const auto cache_dtype = q8 ? at::kChar : at::kBFloat16;
  TORCH_CHECK(key_cache.size(3) == 128 && key_cache.strides() == value_cache.strides() &&
                  key_cache.stride(3) == 1 && key_cache.scalar_type() == cache_dtype &&
                  value_cache.scalar_type() == cache_dtype,
              "KV caches: bf16 or int8 [blocks, page, H, 128] with equal strides");
  if (q8)
    TORCH_CHECK(key_cache.stride(2) % 4 == 0, "int8 cache: 4-byte aligned head rows");
  for (const auto* t : {&k, &v})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->stride(1) == 1 && t->size(0) >= T &&
                    t->size(1) == heads * 128 && t->stride(0) % 8 == 0,
                "k / v: bf16 [T, H * 128]");
  for (const auto* t : {&wk, &wv})
    TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous() &&
                    t->size(0) == heads * 128 && t->size(1) == kW,
                "wk / wv: contiguous fp32 [H * 128, 4]");
  const bool dyn = k_dyn.has_value();
  TORCH_CHECK(dyn == v_dyn.has_value(), "k_dyn and v_dyn together");
  if (dyn)
    for (const auto* t : {&*k_dyn, &*v_dyn})
      TORCH_CHECK(t->scalar_type() == at::kBFloat16 && t->stride(1) == 1 &&
                      t->size(1) == heads * kW && t->stride(0) % 4 == 0,
                  "k_dyn / v_dyn: bf16 [T, H * 4]");
  TORCH_CHECK(kv_slot_mapping.scalar_type() == at::kLong, "kv_slot_mapping: int64");
  const int64_t page = key_cache.size(1);
  TORCH_CHECK(page > 0 && (page & (page - 1)) == 0, "KV page: a power of two");
  for (const torch::Tensor* t : {&k, &v, &cache})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "16-byte aligned data");
  // Int8 stores are 4-byte words (the V rows start 132 bytes after K's).
  for (const torch::Tensor* t : {&key_cache, &value_cache})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % (q8 ? 4 : 16) == 0,
                "aligned cache data");
  if (dyn)
    for (const torch::Tensor* t : {&*k_dyn, &*v_dyn})
      TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 8 == 0, "8-byte aligned deltas");
  if (T == 0) return;
  ConvMeta m = make_meta(cache, positions, block_table, seq_idx, slot_mapping, query_start);
  auto stream = at::hip::getCurrentHIPStream();
  // Head groups fastest: blocks in flight cover every head, whose cache rows
  // are 8 KB apart; token runs fastest put them all on one set of DRAM
  // channels (cache blocks are 64 KB apart) and the stores ran at ~1/4 speed.
  const dim3 grid((heads + 3) / 4, (T + kBT - 1) / kBT);
  auto u16 = [](const torch::Tensor& t) { return reinterpret_cast<const uint16_t*>(t.data_ptr()); };
  const uint16_t* dkp = dyn ? u16(*k_dyn) : nullptr;
  const uint16_t* dvp = dyn ? u16(*v_dyn) : nullptr;
  const int64_t sdk = dyn ? k_dyn->stride(0) : 0, sdv = dyn ? v_dyn->stride(0) : 0;
  auto kernel = q8 ? (dyn ? kv_conv_kernel<true, true> : kv_conv_kernel<false, true>)
                   : (dyn ? kv_conv_kernel<true, false> : kv_conv_kernel<false, false>);
  hipLaunchKernelGGL(kernel, grid, dim3(128), 0, stream, u16(k), u16(v), k.stride(0),
                     v.stride(0), wk.data_ptr<float>(), wv.data_ptr<float>(), dkp, dvp, sdk,
                     sdv, m, cache.stride(1),
                     reinterpret_cast<uint16_t*>(key_cache.data_ptr()),
                     reinterpret_cast<uint16_t*>(value_cache.data_ptr()),
                     kv_slot_mapping.data_ptr<int64_t>(), int(__builtin_ctzll(key_cache.size(1))),
                     key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), int(T),
                     heads, float(scale), float(clip_threshold), float(clip_eps),
                     float(norm_eps));
}

}  // namespace strix
