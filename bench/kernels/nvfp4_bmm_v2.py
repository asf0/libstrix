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


@triton.jit
def _nvfp4_bmm_v4_kernel(
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
):
    # K steps of 32: one MXFP8 activation scale per row per step, two e4m3
    # weight scales per row. Even / odd activation columns meet the low /
    # high nibbles in two K=16 dots, so no register interleave is needed.
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
    j = tl.arange(0, 16)
    weight_rows = (expert * N + cols[:, None]).to(tl.int64)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + 2 * j[None, :]
    w_ptrs = W + weight_rows * (K // 2) + j[None, :]
    ws_ptrs = WScale + weight_rows * (K // 16) + (j[None, :] // 8)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, 32):
        a_scale = tl.load(
            AScale + _scale_offsets(rows, k // 32, K // 32, TILE),
            rows < row_end,
            other=127,
        )
        a_mult = tl.exp2(a_scale.to(tl.float32) - 127.0)[:, None]
        a_even = (tl.load(a_ptrs + k, row_mask, other=0.0).to(tl.float32) * a_mult)
        a_odd = (tl.load(a_ptrs + k + 1, row_mask, other=0.0).to(tl.float32) * a_mult)
        packed = tl.load(w_ptrs + k // 2).to(tl.int32)
        scale = (tl.load(ws_ptrs + k // 16).to(tl.float32) * outer).to(tl.bfloat16)
        w_low = _e2m1_bf16(packed & 15) * scale
        w_high = _e2m1_bf16(packed >> 4) * scale
        acc = tl.dot(a_even.to(tl.bfloat16), tl.trans(w_low), acc)
        acc = tl.dot(a_odd.to(tl.bfloat16), tl.trans(w_high), acc)
    out_ptrs = Out + rows[:, None].to(tl.int64) * N + cols[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), row_mask)


def bmm_nvfp4_v4(
    act, act_scale, weight, weight_scale, weight_amax, weight_denominator,
    tiles, plan: ExpertPlan, bm: int = 64, bn: int = 128, num_warps: int = 4,
    num_stages: int = 2,
):
    num_experts, n, k_half = weight.shape
    k = 2 * k_half
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tile = plan.tile
    bm = min(bm, max(tile, 16))
    sub = tile // bm if tile >= bm else 1
    _nvfp4_bmm_v4_kernel[(n // bn, plan.max_ctas * sub)](
        act, act_scale, weight, weight_scale, weight_amax, weight_denominator,
        out, tiles, plan.max_ctas, N=n, K=k, TILE=tile, BM=bm, BN=bn,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def _nvfp4_dequant_kernel(W, WScale, WAmax, WDenominator, Out, N: tl.constexpr,
                          K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """bf16 `[E, N, K]` copy of NVFP4 experts; program (row block, K block)."""
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    expert = (rows // N).to(tl.int64)
    half = tl.arange(0, BK // 2)
    k0 = tl.program_id(1) * BK
    rows64 = rows.to(tl.int64)[:, None]
    packed = tl.load(W + rows64 * (K // 2) + k0 // 2 + half[None, :]).to(tl.int32)
    outer = tl.load(WAmax + expert) / tl.load(WDenominator)
    scale = tl.load(WScale + rows64 * (K // 16) + k0 // 16 + half[None, :] // 8)
    scale = scale.to(tl.float32) * outer[:, None]
    low = (_e2m1_f32(packed & 15) * scale).to(tl.bfloat16)
    high = (_e2m1_f32(packed >> 4) * scale).to(tl.bfloat16)
    values = tl.join(low, high).reshape(BN, BK)
    tl.store(Out + rows64 * K + k0 + tl.arange(0, BK)[None, :], values)


@triton.jit
def _bf16_bmm_kernel(A, AScale, Wb, Out, Tiles, max_ctas, N: tl.constexpr,
                     K: tl.constexpr, TILE: tl.constexpr, BM: tl.constexpr,
                     BN: tl.constexpr, BK: tl.constexpr):
    """Grouped GEMM of MXFP8 activation rows (upcast per 32-column group)
    against bf16 expert weights `[E, N, K]`."""
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
    rows = row0 + tl.arange(0, BM)
    row_mask = (rows < row_end)[:, None]
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + ks[None, :]
    w_ptrs = Wb + (expert * N + cols[:, None]).to(tl.int64) * K + ks[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a_scale = tl.load(AScale + _scale_offsets(rows, k // 32, K // 32, TILE),
                          rows < row_end, other=127)
        a = tl.load(a_ptrs + k, row_mask, other=0.0).to(tl.float32)
        a = (a * tl.exp2(a_scale.to(tl.float32) - 127.0)[:, None]).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(tl.load(w_ptrs + k)), acc)
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], acc.to(tl.bfloat16),
             row_mask)


_SCRATCH: dict = {}


def bmm_nvfp4_dequant(
    act, act_scale, weight, weight_scale, weight_amax, weight_denominator,
    tiles, plan: ExpertPlan, bm: int = 64, bn: int = 128, num_warps: int = 4,
    timing: dict | None = None,
):
    num_experts, n, k_half = weight.shape
    k = 2 * k_half
    key = (num_experts, n, k, act.device)
    wb = _SCRATCH.get(key)
    if wb is None:
        wb = _SCRATCH[key] = torch.empty((num_experts, n, k), dtype=torch.bfloat16,
                                         device=act.device)
    _nvfp4_dequant_kernel[(num_experts * n // 32, k // 128)](
        weight, weight_scale, weight_amax, weight_denominator, wb, N=n, K=k,
        BN=32, BK=128, num_warps=4)
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tile = plan.tile
    bm = min(bm, max(tile, 16))
    sub = tile // bm if tile >= bm else 1
    _bf16_bmm_kernel[(n // bn, plan.max_ctas * sub)](
        act, act_scale, wb, out, tiles, plan.max_ctas, N=n, K=k, TILE=tile, BM=bm,
        BN=bn, BK=32, num_warps=num_warps, num_stages=2)
    return out


@triton.jit
def _bf16_bf16_bmm_kernel(A, Wb, Out, Tiles, max_ctas, N: tl.constexpr,
                          K: tl.constexpr, TILE: tl.constexpr, BM: tl.constexpr,
                          BN: tl.constexpr, BK: tl.constexpr):
    """`_bf16_bmm_kernel` with bf16 activation rows (no MXFP8 upcast)."""
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
    rows = row0 + tl.arange(0, BM)
    row_mask = (rows < row_end)[:, None]
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + ks[None, :]
    w_ptrs = Wb + (expert * N + cols[:, None]).to(tl.int64) * K + ks[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs + k, row_mask, other=0.0)
        acc = tl.dot(a, tl.trans(tl.load(w_ptrs + k)), acc)
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], acc.to(tl.bfloat16),
             row_mask)


def bmm_bf16_act_only(act_bf16, wb, tiles, plan: ExpertPlan, bm=64, bn=128, nw=4):
    num_experts, n, k = wb.shape
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act_bf16.device)
    tile = plan.tile
    bm = min(bm, max(tile, 16))
    sub = tile // bm if tile >= bm else 1
    _bf16_bf16_bmm_kernel[(n // bn, plan.max_ctas * sub)](
        act_bf16, wb, out, tiles, plan.max_ctas, N=n, K=k, TILE=tile, BM=bm, BN=bn,
        BK=32, num_warps=nw, num_stages=2)
    return out


@triton.jit
def _nvfp4_bmm_v5_kernel(A, W, WScale, WAmax, WDenominator, Out, Tiles, max_ctas,
                         N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr):
    """`_nvfp4_bmm_v4_kernel` with bf16 activation rows."""
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
    j = tl.arange(0, 16)
    weight_rows = (expert * N + cols[:, None]).to(tl.int64)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + 2 * j[None, :]
    w_ptrs = W + weight_rows * (K // 2) + j[None, :]
    ws_ptrs = WScale + weight_rows * (K // 16) + (j[None, :] // 8)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, 32):
        a_even = tl.load(a_ptrs + k, row_mask, other=0.0)
        a_odd = tl.load(a_ptrs + k + 1, row_mask, other=0.0)
        packed = tl.load(w_ptrs + k // 2).to(tl.int32)
        scale = (tl.load(ws_ptrs + k // 16).to(tl.float32) * outer).to(tl.bfloat16)
        acc = tl.dot(a_even, tl.trans(_e2m1_bf16(packed & 15) * scale), acc)
        acc = tl.dot(a_odd, tl.trans(_e2m1_bf16(packed >> 4) * scale), acc)
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], acc.to(tl.bfloat16),
             row_mask)


def bmm_nvfp4_v5(act_bf16, weight, weight_scale, weight_amax, weight_denominator,
                 tiles, plan: ExpertPlan, bm=64, bn=64, nw=4, ns=2):
    num_experts, n, k_half = weight.shape
    k = 2 * k_half
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act_bf16.device)
    tile = plan.tile
    bm = min(bm, max(tile, 16))
    sub = tile // bm if tile >= bm else 1
    _nvfp4_bmm_v5_kernel[(n // bn, plan.max_ctas * sub)](
        act_bf16, weight, weight_scale, weight_amax, weight_denominator, out, tiles,
        plan.max_ctas, N=n, K=k, TILE=tile, BM=bm, BN=bn, num_warps=nw, num_stages=ns)
    return out


@triton.jit
def _nvfp4_gemv_v6_kernel(A, W32, WScale, WAmax, WDenominator, Out, Tiles, max_ctas,
                          N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr,
                          BN: tl.constexpr, BK: tl.constexpr):
    """GEMV over bf16 activations; NVFP4 weights read as int32 words (8
    nibbles each, element order)."""
    row = tl.program_id(1)
    tile = row // TILE
    if tile >= tl.load(Tiles + 2 * max_ctas):
        return
    if row >= tl.load(Tiles + max_ctas + tile):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    outer = tl.load(WAmax + expert) / tl.load(WDenominator)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    words = tl.arange(0, BK // 8)
    shifts = (tl.arange(0, 8) * 4)[None, None, :]
    weight_rows = (expert * N + cols[:, None]).to(tl.int64)
    a_row = A + row.to(tl.int64) * K
    acc = tl.zeros((BN,), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_row + k + tl.arange(0, BK)).to(tl.float32)
        packed = tl.load(W32 + weight_rows * (K // 8) + k // 8 + words[None, :])
        codes = (packed[:, :, None] >> shifts) & 15
        values = _e2m1_f32(codes).reshape(BN, BK)
        scale = tl.load(WScale + weight_rows * (K // 16) + k // 16
                        + tl.arange(0, BK // 16)[None, :]).to(tl.float32)
        partial = (values * a[None, :]).reshape(BN, BK // 16, 16)
        acc += tl.sum(tl.sum(partial, axis=2) * scale, axis=1)
    tl.store(Out + row.to(tl.int64) * N + cols, (acc * outer).to(tl.bfloat16))


def bmm_nvfp4_gemv_v6(act_bf16, weight, weight_scale, weight_amax, weight_denominator,
                      tiles, plan: ExpertPlan, bn=32, bk=256, nw=4):
    num_experts, n, k_half = weight.shape
    k = 2 * k_half
    while k % bk:
        bk //= 2
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act_bf16.device)
    _nvfp4_gemv_v6_kernel[(n // bn, plan.rows)](
        act_bf16, weight.view(torch.int32), weight_scale, weight_amax,
        weight_denominator, out, tiles, plan.max_ctas, N=n, K=k, TILE=plan.tile,
        BN=bn, BK=bk, num_warps=nw)
    return out
