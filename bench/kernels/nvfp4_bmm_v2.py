"""NVFP4 expert GEMM, decode-friendly variant (candidate for
vllm/models/inkling2/amd/ops/triton_bmm.py).

E2M1 codes are turned into bf16 bit patterns with integer ops, the two nibbles
of a byte multiply the even and odd activation columns in separate dots (no
register interleave), and the MXFP8 activations are upcast to bf16 for a plain
`tl.dot`.
"""

import torch
import triton
import triton.language as tl

from vllm.models.inkling2.amd.ops.mxfp8 import _scale_offsets
from vllm.models.inkling2.amd.ops.triton_bmm import ExpertPlan


@triton.jit
def _e2m1_bf16(code):
    """bf16 value of each 4-bit e2m1 `code` (as int32), via its bit pattern."""
    exp = (code >> 1) & 3
    man = code & 1
    normal = ((exp + 126) << 7) | (man << 6)
    sub = tl.where(man == 1, 126 << 7, 0)
    bits = tl.where(exp > 0, normal, sub) | ((code & 8) << 12)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _nvfp4_bmm_v2_kernel(
    A,
    AScale,
    W,
    WScale,
    WAmax,
    WDenominator,
    Out,
    Tiles,
    max_ctas,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    SUB: tl.constexpr = TILE // BM if TILE >= BM else 1
    pid_m = tl.program_id(1)
    tile = pid_m // SUB
    if tile >= tl.load(Tiles + 2 * max_ctas):
        return
    row_end = tl.load(Tiles + max_ctas + tile)
    row0 = tile * TILE + (pid_m % SUB) * BM
    if row0 >= row_end:
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    outer = tl.load(WAmax + expert) / tl.load(WDenominator)
    rows = row0 + tl.arange(0, BM)
    row_mask = (rows < row_end)[:, None]
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    half = tl.arange(0, BK // 2)
    weight_rows = (expert * N + cols[:, None]).to(tl.int64)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + 2 * half[None, :]
    w_ptrs = W + weight_rows * (K // 2) + half[None, :]
    ws_ptrs = WScale + weight_rows * (K // 16) + tl.arange(0, BK // 16)[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        # Activations: e4m3 with an e8m0 scale per 32 columns, as bf16 halves.
        a_even = tl.load(a_ptrs + k, row_mask, other=0.0).to(tl.float32)
        a_odd = tl.load(a_ptrs + k + 1, row_mask, other=0.0).to(tl.float32)
        scale_cols = (k + 2 * half[None, :]) // 32
        a_scale = tl.load(
            AScale + _scale_offsets(rows[:, None], scale_cols, K // 32, TILE),
            row_mask,
            other=127,
        )
        a_mult = tl.exp2(a_scale.to(tl.float32) - 127.0)
        a_even = (a_even * a_mult).to(tl.bfloat16)
        a_odd = (a_odd * a_mult).to(tl.bfloat16)
        # Weights: two e2m1 per byte (even column low), e4m3 scale per 16.
        packed = tl.load(w_ptrs + k // 2).to(tl.int32)
        block_scale = (tl.load(ws_ptrs + k // 16).to(tl.float32) * outer).to(
            tl.bfloat16
        )
        scale = tl.broadcast_to(
            block_scale[:, :, None], (BN, BK // 16, 8)
        ).reshape(BN, BK // 2)
        w_low = _e2m1_bf16(packed & 15) * scale
        w_high = _e2m1_bf16(packed >> 4) * scale
        acc = tl.dot(a_even, tl.trans(w_low), acc)
        acc = tl.dot(a_odd, tl.trans(w_high), acc)
    out_ptrs = Out + rows[:, None].to(tl.int64) * N + cols[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), row_mask)


def bmm_nvfp4_v2(
    act, act_scale, weight, weight_scale, weight_amax, weight_denominator,
    tiles, plan: ExpertPlan, bn: int = 64, bk: int = 256, num_warps: int = 4,
    num_stages: int = 2,
):
    num_experts, n, k_half = weight.shape
    k = 2 * k_half
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tile = plan.tile
    bm = min(max(tile, 16), 64)
    while k % bk:
        bk //= 2
    sub = tile // bm if tile >= bm else 1
    _nvfp4_bmm_v2_kernel[(n // bn, plan.max_ctas * sub)](
        act, act_scale, weight, weight_scale, weight_amax, weight_denominator,
        out, tiles, plan.max_ctas, N=n, K=k, TILE=tile, BM=bm, BN=bn, BK=bk,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def _e2m1_f32(code):
    """fp32 value of each 4-bit e2m1 `code` (as int32), via its bit pattern."""
    exp = (code >> 1) & 3
    man = code & 1
    normal = ((exp + 126) << 23) | (man << 22)
    sub = tl.where(man == 1, 126 << 23, 0)
    bits = tl.where(exp > 0, normal, sub) | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _nvfp4_gemv_kernel(
    A,
    AScale,
    W,
    WScale,
    WAmax,
    WDenominator,
    Out,
    Tiles,
    max_ctas,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # One routed row and BN output columns per program: streams the expert's
    # NVFP4 rows once and reduces over K without a matrix instruction.
    row = tl.program_id(1)
    tile = row // TILE
    if tile >= tl.load(Tiles + 2 * max_ctas):
        return
    if row >= tl.load(Tiles + max_ctas + tile):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    outer = tl.load(WAmax + expert) / tl.load(WDenominator)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    half = tl.arange(0, BK // 2)
    weight_rows = (expert * N + cols[:, None]).to(tl.int64)
    a_row = A + row.to(tl.int64) * K
    acc = tl.zeros((BN,), tl.float32)
    for k in range(0, K, BK):
        a_even = tl.load(a_row + k + 2 * half).to(tl.float32)
        a_odd = tl.load(a_row + k + 2 * half + 1).to(tl.float32)
        a_scale = tl.load(
            AScale + _scale_offsets(row, (k + 2 * half) // 32, K // 32, TILE)
        )
        a_mult = tl.exp2(a_scale.to(tl.float32) - 127.0)
        packed = tl.load(W + weight_rows * (K // 2) + k // 2 + half[None, :]).to(
            tl.int32
        )
        block_scale = tl.load(
            WScale + weight_rows * (K // 16) + k // 16 + half[None, :] // 8
        ).to(tl.float32)
        partial = _e2m1_f32(packed & 15) * (a_even * a_mult)[None, :]
        partial += _e2m1_f32(packed >> 4) * (a_odd * a_mult)[None, :]
        acc += tl.sum(partial * block_scale, axis=1)
    tl.store(Out + row.to(tl.int64) * N + cols, (acc * outer).to(tl.bfloat16))


def bmm_nvfp4_gemv(
    act, act_scale, weight, weight_scale, weight_amax, weight_denominator,
    tiles, plan: ExpertPlan, bn: int = 32, bk: int = 256, num_warps: int = 4,
):
    num_experts, n, k_half = weight.shape
    k = 2 * k_half
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    while k % bk:
        bk //= 2
    _nvfp4_gemv_kernel[(n // bn, plan.rows)](
        act, act_scale, weight, weight_scale, weight_amax, weight_denominator,
        out, tiles, plan.max_ctas, N=n, K=k, TILE=plan.tile, BN=bn, BK=bk,
        num_warps=num_warps,
    )
    return out
