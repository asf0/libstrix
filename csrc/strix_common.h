// SPDX-License-Identifier: MIT
// Shared gfx1151 (RDNA3.5, wave32) device helpers.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

namespace strix {

constexpr int kWave = 32;

// Control words, from the RDNA3.5 ISA encodings.
// DPP quad_perm: lane i of each quad reads lane sel[i] of the same quad.
constexpr int dpp_quad_perm(int s0, int s1, int s2, int s3) {
  return s0 | (s1 << 2) | (s2 << 4) | (s3 << 6);
}
// ds_swizzle bitmask mode (offset bit 15 clear): within each group of 32
// lanes, lane i reads lane ((i & and_mask) | or_mask) ^ xor_mask.
constexpr int swizzle_bitmask(int and_mask, int or_mask, int xor_mask) {
  return and_mask | (or_mask << 5) | (xor_mask << 10);
}
// v_permlanex16 lane selects: lane i reads lane sel_i of the other 16-lane
// half; packed 4 bits per lane, lanes 0-7 in the low word.
constexpr unsigned permlane_identity_lo = 0x76543210u;
constexpr unsigned permlane_identity_hi = 0xfedcba98u;

// x from lane (lane ^ stride) of the wave, as __shfl_xor(x, stride, 32), but
// without ds_bpermute's LDS round trip for the power-of-two strides: strides
// 1 and 2 stay inside a quad (DPP), 4 and 8 inside 32 lanes (swizzle), and
// 16 swaps the two halves (permlanex16 with identity selects).
__device__ __forceinline__ int xor_shuffle_b32(int x, int stride) {
  constexpr int kAllRows = 0xf, kAllBanks = 0xf, kAllLanes = 0x1f;
  if (stride == 1)
    return __builtin_amdgcn_mov_dpp(x, dpp_quad_perm(1, 0, 3, 2), kAllRows,
                                    kAllBanks, false);
  if (stride == 2)
    return __builtin_amdgcn_mov_dpp(x, dpp_quad_perm(2, 3, 0, 1), kAllRows,
                                    kAllBanks, false);
  if (stride == 4)
    return __builtin_amdgcn_ds_swizzle(x, swizzle_bitmask(kAllLanes, 0, 4));
  if (stride == 8)
    return __builtin_amdgcn_ds_swizzle(x, swizzle_bitmask(kAllLanes, 0, 8));
  if (stride == 16)
    return __builtin_amdgcn_permlanex16(x, x, permlane_identity_lo,
                                        permlane_identity_hi, false, false);
  return __shfl_xor(x, stride, kWave);
}

__device__ __forceinline__ float xor_shuffle_f32(float x, int stride) {
  return __int_as_float(xor_shuffle_b32(__float_as_int(x), stride));
}

// Sum over aligned groups of `Width` lanes (a power of two <= 32).
template <int Width = kWave>
__device__ __forceinline__ float group_sum(float x) {
#pragma unroll
  for (int s = Width / 2; s >= 1; s >>= 1) x += xor_shuffle_f32(x, s);
  return x;
}

// Max over aligned groups of `Width` lanes (a power of two <= 32).
template <int Width = kWave>
__device__ __forceinline__ float group_max(float x) {
#pragma unroll
  for (int s = Width / 2; s >= 1; s >>= 1) x = fmaxf(x, xor_shuffle_f32(x, s));
  return x;
}

// Workgroup barrier for LDS hand-offs. In WGP mode a workgroup's waves span
// both CUs, whose LDS queues are not ordered, and the compiler may drop the
// s_waitcnt it places before __syncthreads()'s s_barrier
// (llvm/llvm-project#220357), so a read after the barrier can overtake
// another wave's LDS store. The wait is therefore written out explicitly,
// between release/acquire fences that keep the compiler from moving LDS
// accesses across it.
// gfx11 s_waitcnt simm16: expcnt [2:0], lgkmcnt [9:4], vmcnt [15:10]; a field
// at its maximum does not wait on that counter.
constexpr int gfx11_waitcnt(int vmcnt, int expcnt, int lgkmcnt) {
  return expcnt | (lgkmcnt << 4) | (vmcnt << 10);
}

__device__ __forceinline__ void barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_waitcnt(gfx11_waitcnt(0x3f, 0x7, 0));  // lgkmcnt(0) only
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ float bf16_lo(uint32_t p) {
  return __uint_as_float(p << 16);
}
__device__ __forceinline__ float bf16_hi(uint32_t p) {
  return __uint_as_float(p & 0xffff0000u);
}

// Round-to-nearest-even fp32 -> bf16, left in the high half of the word.
__device__ __forceinline__ uint32_t round_bf16_hi(float x) {
  const uint32_t a = __float_as_uint(x);
  return a + 0x7fffu + ((a >> 16) & 1u);
}

// Round-to-nearest-even fp32 -> bf16 pair; gfx1151 has no packed bf16 convert.
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  uint32_t a = __float_as_uint(lo), b = __float_as_uint(hi);
  a += 0x7fffu + ((a >> 16) & 1u);
  b += 0x7fffu + ((b >> 16) & 1u);
  return __builtin_amdgcn_perm(b, a, 0x07060302u);
}

// Recurrent state (GDN): fp32 or bf16, four values at a time. vLLM's
// --mamba-ssm-cache-dtype picks which, and the state is every GDN layer's
// dominant traffic at decode, so both dtypes have to be native here.
template <typename StateT>
__device__ __forceinline__ float4 load_state4(const StateT* s);
template <>
__device__ __forceinline__ float4 load_state4<float>(const float* s) {
  return *reinterpret_cast<const float4*>(s);
}
template <>
__device__ __forceinline__ float4 load_state4<uint16_t>(const uint16_t* s) {
  const uint2 w = *reinterpret_cast<const uint2*>(s);
  return make_float4(bf16_lo(w.x), bf16_hi(w.x), bf16_lo(w.y), bf16_hi(w.y));
}
template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* s, float4 v);
template <>
__device__ __forceinline__ void store_state4<float>(float* s, float4 v) {
  *reinterpret_cast<float4*>(s) = v;
}
template <>
__device__ __forceinline__ void store_state4<uint16_t>(uint16_t* s, float4 v) {
  *reinterpret_cast<uint2*>(s) =
      make_uint2(pack_bf16x2(v.x, v.y), pack_bf16x2(v.z, v.w));
}

// 8 signed int8 (two packed words) -> 8 bf16, without the scale: |v| <= 127 is
// exact in bf16, so a caller that has one scale for a whole key (int8 KV
// per-token-head) can fold it into the score and the P fragment instead of
// multiplying every element here.
__device__ __forceinline__ void q8x8_to_bf16(uint32_t a, uint32_t b,
                                             uint32_t (&o)[4]) {
  // Bias the bytes to unsigned (one xor per word) so each converts with a
  // single v_cvt_f32_ubyteN; the results are small integers, exact in bf16,
  // so the pair takes the high halves directly (no rounding). ~2.5 VALU ops
  // per byte instead of ~4.5 - the widening was the int8 prefill's largest
  // VALU cost.
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t w = (i ? b : a) ^ 0x80808080u;
    const float f0 = static_cast<float>(w & 0xffu) - 128.f;
    const float f1 = static_cast<float>((w >> 8) & 0xffu) - 128.f;
    const float f2 = static_cast<float>((w >> 16) & 0xffu) - 128.f;
    const float f3 = static_cast<float>(w >> 24) - 128.f;
    o[2 * i] = __builtin_amdgcn_perm(__float_as_uint(f1), __float_as_uint(f0), 0x07060302u);
    o[2 * i + 1] =
        __builtin_amdgcn_perm(__float_as_uint(f3), __float_as_uint(f2), 0x07060302u);
  }
}

// 8 signed int8 (two packed words) -> 8 fp16, without the scale. fp16 has
// the exponent of 1024 + u for every u in 0..255, so the biased byte u = x+128
// drops straight into a mantissa: one v_perm builds two 0x64uu halves (=1024+u)
// and one packed subtract of 1152 leaves x, exact. ~1 VALU op per byte where
// the bf16 route needs a convert per byte.
__device__ __forceinline__ void q8x8_to_f16(uint32_t a, uint32_t b, uint32_t (&o)[4]) {
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  const h2 bias = {static_cast<_Float16>(1152.f), static_cast<_Float16>(1152.f)};
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t w = (i ? b : a) ^ 0x80808080u;
    const uint32_t p0 = __builtin_amdgcn_perm(0x64646464u, w, 0x04010400u);
    const uint32_t p1 = __builtin_amdgcn_perm(0x64646464u, w, 0x04030402u);
    o[2 * i] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, p0) - bias);
    o[2 * i + 1] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, p1) - bias);
  }
}

// fp32 -> fp16 bits in the low half of the word (round to nearest even).
__device__ __forceinline__ uint32_t f16_bits(float x) {
  return __builtin_bit_cast(uint16_t, static_cast<_Float16>(x));
}

}  // namespace strix

namespace strix {

// gfx11 wave32 WMMA, 16x16x16, D = A * B^T + C with both operands K-major:
//  - A fragment: lane l holds A[l % 16][0..15] (16 bf16); lanes 16-31 repeat
//    lanes 0-15.
//  - B fragment: lane l holds B[l % 16][0..15], same replication.
//  - C/D fragment: element e of lane l is D[2 * e + l / 16][l % 16].
typedef short bf16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ f32x8 wmma_bf16(bf16x16 a, bf16x16 b, f32x8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// The same WMMA on fp16 operands; the fragments keep the 16-bit `short`
// containers (and load_frag_bf16), only the instruction reads them as fp16.
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
__device__ __forceinline__ f32x8 wmma_f16(bf16x16 a, bf16x16 b, f32x8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(f16x16, a),
                                                    __builtin_bit_cast(f16x16, b), c);
}

// Loads row (lane % 16) of a K-major 16x16 tile starting at `base` with row
// pitch `ld` elements; 32-byte aligned rows give two 16-byte loads.
__device__ __forceinline__ bf16x16 load_frag_bf16(const uint16_t* base,
                                                  int64_t ld, int lane) {
  const uint4* p = reinterpret_cast<const uint4*>(base + (lane % 16) * ld);
  const uint4 lo = p[0], hi = p[1];
  bf16x16 f;
  const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    f[2 * i] = static_cast<short>(w[i] & 0xffffu);
    f[2 * i + 1] = static_cast<short>(w[i] >> 16);
  }
  return f;
}

__device__ __forceinline__ int acc_row(int e, int lane) {
  return 2 * e + lane / 16;
}
__device__ __forceinline__ int acc_col(int lane) { return lane % 16; }

}  // namespace strix
