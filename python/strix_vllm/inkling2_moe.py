# SPDX-License-Identifier: MIT
"""Inkling2 (ROCm port) mixture of experts on gfx1151.

RDNA has no FP8 conversion instructions, so the port's MXFP8 activations cost
the expert GEMMs most of their speed (3-6 TFLOPS against 20-22 with bf16
rows). Here the router scatters and the SwiGLU writes bf16 rows, and the
experts run on:

- decode (a row or two per expert tile): GEMVs that stream the NVFP4 / MXFP8
  weights once and reduce on the vector units;
- larger batches: grouped GEMMs that decode the weights from their bits into
  bf16 WMMA tiles.

Installed over `vllm.models.inkling2.amd.moe.Inkling2MoE.forward`; the
checkpoint layout and the routing / combine kernels are the port's.
"""

from __future__ import annotations

import torch
from vllm import envs
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.inkling2.amd.ops.activation import _row_inputs, _situ
from vllm.models.inkling2.amd.ops.linear import k_group_partials
from vllm.models.inkling2.amd.ops.moe import (
    _pick,
    _place_rows,
    _route_tokens_kernel,
    combine,
)
from vllm.models.inkling2.amd.ops.mxfp8 import _scale_offsets
from vllm.triton_utils import tl, triton

# Batches up to this many tokens (8 experts each of 128: about a row per
# expert tile) take the GEMVs.
GEMV_MAX_CTAS = 64


@triton.jit
def _e2m1_f32_bits(code):
    """fp32 value of each 4-bit e2m1 `code` (int32), built from its bits."""
    exp = (code >> 1) & 3
    man = code & 1
    normal = ((exp + 126) << 23) | (man << 22)
    sub = tl.where(man == 1, 126 << 23, 0)
    bits = tl.where(exp > 0, normal, sub) | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _e2m1_bf16(code):
    """bf16 value of each 4-bit e2m1 `code` (int32), built from its bits."""
    exp = (code >> 1) & 3
    man = code & 1
    normal = ((exp + 126) << 7) | (man << 6)
    sub = tl.where(man == 1, 126 << 7, 0)
    bits = tl.where(exp > 0, normal, sub) | ((code & 8) << 12)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _e4m3_f32(code):
    """fp32 value of each e4m3fn byte `code` (int32) from its bits: RDNA has
    no FP8 conversion instructions and Triton's software one is slow."""
    exp = (code >> 3) & 15
    man = code & 7
    normal = (((exp + 120) << 23) | (man << 20)).to(tl.float32, bitcast=True)
    value = tl.where(exp > 0, normal, man.to(tl.float32) * 0.001953125)
    return tl.where((code & 128) != 0, -value, value)


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
    A_BF16: tl.constexpr,
):
    # One routed row and BN output columns per program: streams the expert's
    # NVFP4 rows once and reduces over K on the vector units. For decode,
    # where each expert tile holds a row or two, this runs at memory speed
    # where `tl.dot` on gfx1151 does not.
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
        if A_BF16:
            a_even = tl.load(a_row + k + 2 * half).to(tl.float32)
            a_odd = tl.load(a_row + k + 2 * half + 1).to(tl.float32)
        else:
            a_scale = tl.load(
                AScale + _scale_offsets(row, (k + 2 * half) // 32, K // 32, TILE)
            )
            a_mult = tl.exp2(a_scale.to(tl.float32) - 127.0)
            a_even = tl.load(a_row + k + 2 * half).to(tl.float32) * a_mult
            a_odd = tl.load(a_row + k + 2 * half + 1).to(tl.float32) * a_mult
        packed = tl.load(W + weight_rows * (K // 2) + k // 2 + half[None, :])
        packed = packed.to(tl.int32)
        block_scale = tl.load(
            WScale + weight_rows * (K // 16) + k // 16 + half[None, :] // 8
        ).to(tl.float32)
        partial = _e2m1_f32_bits(packed & 15) * a_even[None, :]
        partial += _e2m1_f32_bits(packed >> 4) * a_odd[None, :]
        acc += tl.sum(partial * block_scale, axis=1)
    tl.store(Out + row.to(tl.int64) * N + cols, (acc * outer).to(tl.bfloat16))


@triton.jit
def _nvfp4_bmm_bf16_kernel(
    A,
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
    # bf16 activation rows against NVFP4 weights decoded from their bits, K
    # in steps of 32: even / odd activation columns meet the low / high
    # nibbles in two K=16 dots (no register interleave of the decoded values).
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
    out_ptrs = Out + rows[:, None].to(tl.int64) * N + cols[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), row_mask)


@triton.jit
def _mxfp8_gemv_kernel(
    A,
    AScale,
    W,
    WScale,
    Out,
    Tiles,
    max_ctas,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    A_BF16: tl.constexpr,
):
    # `_nvfp4_gemv_kernel` for MXFP8 weights (e4m3, an e8m0 scale per 32).
    row = tl.program_id(1)
    tile = row // TILE
    if tile >= tl.load(Tiles + 2 * max_ctas):
        return
    if row >= tl.load(Tiles + max_ctas + tile):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    weight_rows = (expert * N + cols[:, None]).to(tl.int64)
    a_row = A + row.to(tl.int64) * K
    acc = tl.zeros((BN,), tl.float32)
    for k in range(0, K, BK):
        if A_BF16:
            a = tl.load(a_row + k + ks).to(tl.float32)
        else:
            a_scale = tl.load(
                AScale + _scale_offsets(row, (k + ks) // 32, K // 32, TILE)
            )
            a = tl.load(a_row + k + ks).to(tl.float32) * tl.exp2(
                a_scale.to(tl.float32) - 127.0
            )
        codes = tl.load(
            (W + weight_rows * K + k + ks[None, :]).to(tl.pointer_type(tl.uint8))
        )
        w = _e4m3_f32(codes.to(tl.int32))
        w_scale = tl.load(WScale + weight_rows * (K // 32) + (k + ks[None, :]) // 32)
        w = w * tl.exp2(w_scale.to(tl.float32) - 127.0)
        acc += tl.sum(w * a[None, :], axis=1)
    tl.store(Out + row.to(tl.int64) * N + cols, acc.to(tl.bfloat16))


@triton.jit
def _mxfp8_bmm_bf16_kernel(
    A,
    W,
    WScale,
    Out,
    Tiles,
    max_ctas,
    N: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    # bf16 activation rows against MXFP8 weights decoded from their bits, K
    # in steps of one 32-column scale block.
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
    ks = tl.arange(0, 32)
    weight_rows = (expert * N + cols).to(tl.int64)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + ks[None, :]
    w_ptrs = (W + weight_rows[:, None] * K + ks[None, :]).to(tl.pointer_type(tl.uint8))
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, 32):
        a = tl.load(a_ptrs + k, row_mask, other=0.0)
        scale = tl.exp2(
            tl.load(WScale + weight_rows * (K // 32) + k // 32).to(tl.float32) - 127.0
        )
        w = _e4m3_f32(tl.load(w_ptrs + k).to(tl.int32)) * scale[:, None]
        acc = tl.dot(a, tl.trans(w.to(tl.bfloat16)), acc)
    out_ptrs = Out + rows[:, None].to(tl.int64) * N + cols[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), row_mask)


@triton.jit
def _scatter_bf16_kernel(
    X,
    Starts,
    Ids,
    Ranks,
    Weights,
    Src2Dst,
    Gammas,
    Rows,
    Values,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    """`_scatter_kernel` that copies the bf16 row instead of quantizing it."""
    token = tl.program_id(0).to(tl.int64)
    head = tl.load(X + token * H + tl.arange(0, HEAD))
    if TAIL > 0:
        tail = tl.load(X + token * H + HEAD + tl.arange(0, TAIL))
    dst = _place_rows(
        Starts, Ids, Ranks, Weights, Src2Dst, Gammas, Rows, token, TOPK, LOCAL_POW2
    )
    for slot in range(dst.shape[0]):
        row = _pick(dst, slot).to(tl.int64)
        if row >= 0:
            tl.store(Values + row * H + tl.arange(0, HEAD), head)
            if TAIL > 0:
                tl.store(Values + row * H + HEAD + tl.arange(0, TAIL), tail)


@triton.jit
def _situ_bf16_kernel(
    X,
    Gammas,
    Values,
    Rows,
    NumRows,
    N: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    # `_situ_quantize_kernel` writing the bf16 values themselves.
    num_rows = tl.load(NumRows)
    r = tl.program_id(0)
    row, gamma = _row_inputs(Rows, Gammas, r, num_rows)
    start = tl.program_id(1) * (HEAD + TAIL)
    while r < num_rows:
        head = _situ(X, row, start, gamma, N, HEAD, GATE_CAP, UP_CAP)
        tl.store(Values + row * N + start + tl.arange(0, HEAD), head.to(tl.bfloat16))
        if TAIL > 0:
            tail = _situ(X, row, start + HEAD, gamma, N, TAIL, GATE_CAP, UP_CAP)
            tl.store(
                Values + row * N + start + HEAD + tl.arange(0, TAIL),
                tail.to(tl.bfloat16),
            )
        r += tl.num_programs(0)
        row, gamma = _row_inputs(Rows, Gammas, r, num_rows)


def route_bf16(
    x,
    gate_weight,
    bias,
    counters,
    counts,
    topk,
    route_scale,
    expert_bounds,
    plan,
    is_padding=None,
):
    """The port's unfused `route`, scattering bf16 rows instead of MXFP8."""
    tokens, hidden = x.shape
    experts = gate_weight.shape[0]
    num_local = counts.numel()
    n = tokens * topk
    device = x.device
    rows = plan.rows
    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    gammas = torch.empty(rows, dtype=torch.float32, device=device)
    routed = torch.empty(n, dtype=torch.int32, device=device)
    tiles = torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device)
    ids = torch.empty(n, dtype=torch.int32, device=device)
    ranks = torch.empty(n, dtype=torch.int32, device=device)
    weights = torch.empty(n, dtype=torch.float32, device=device)
    values = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
    head = 1 << (hidden.bit_length() - 1)
    tail = hidden - head
    partials = k_group_partials(x, gate_weight)
    local_pow2 = triton.next_power_of_2(num_local)
    starts = torch.empty(3 * local_pow2, dtype=torch.int32, device=device)
    _route_tokens_kernel[(tokens,)](
        partials,
        bias,
        expert_bounds,
        is_padding,
        counts,
        counters,
        ids,
        ranks,
        weights,
        starts,
        tiles,
        tokens,
        route_scale,
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=partials.shape[0],
        TOPK=topk,
        SORT=tokens <= 128,
        NUM_LOCAL=num_local,
        LOCAL_POW2=local_pow2,
        TILE=plan.tile,
        MAX_CTAS=plan.max_ctas,
        num_warps=4 if tokens <= 128 else 1,
    )
    _scatter_bf16_kernel[(tokens,)](
        x,
        starts,
        ids,
        ranks,
        weights,
        src2dst,
        gammas,
        routed,
        values,
        H=hidden,
        TOPK=topk,
        LOCAL_POW2=local_pow2,
        HEAD=head,
        TAIL=tail,
        num_warps=2,
    )
    return (src2dst, gammas, routed, tiles), values


def swiglu_bf16(x, gammas, gate_cap, up_cap, rows, tile):
    """The port's softcapped SwiGLU, writing bf16 rows."""
    num_rows, n = x.shape[0], x.shape[1] // 2
    routed, num_routed = rows
    max_rows = routed.numel()
    if max_rows <= 512:
        head, tail = 256 if n % 256 == 0 else 128, 0
    else:
        head = 1 << (n.bit_length() - 1)
        tail = n - head
    values = torch.empty((num_rows, n), dtype=torch.bfloat16, device=x.device)
    _situ_bf16_kernel[(min(max_rows, 16384), n // (head + tail))](
        x,
        gammas,
        values,
        routed,
        num_routed,
        N=n,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        HEAD=head,
        TAIL=tail,
        num_warps=2,
        enable_fp_fusion=False,
    )
    return values


def _gemv_launch(plan, k):
    return plan.tile == 16 and plan.max_ctas <= GEMV_MAX_CTAS and k % 128 == 0


def gemm_nvfp4(act, weight, weight_scale, weight_amax, weight_denominator, tiles, plan):
    _, n, k_half = weight.shape
    k = 2 * k_half
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tile = plan.tile
    if _gemv_launch(plan, k):
        bn, warps = (64, 4) if k >= 2048 else (16, 1)
        _nvfp4_gemv_kernel[(n // bn, plan.rows)](
            act,
            act,
            weight,
            weight_scale,
            weight_amax,
            weight_denominator,
            out,
            tiles,
            plan.max_ctas,
            N=n,
            K=k,
            TILE=tile,
            BN=bn,
            BK=256 if k % 256 == 0 else 128,
            A_BF16=True,
            num_warps=warps,
        )
        return out
    bm = min(max(tile, 16), 128)
    bn = 128 if n % 128 == 0 else 64
    sub = tile // bm if tile >= bm else 1
    _nvfp4_bmm_bf16_kernel[(n // bn, plan.max_ctas * sub)](
        act,
        weight,
        weight_scale,
        weight_amax,
        weight_denominator,
        out,
        tiles,
        plan.max_ctas,
        N=n,
        K=k,
        TILE=tile,
        BM=bm,
        BN=bn,
        num_warps=8 if bm == 128 else 4,
        num_stages=2,
    )
    return out


def gemm_mxfp8(act, weight, weight_scale, tiles, plan):
    _, n, k = weight.shape
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tile = plan.tile
    if _gemv_launch(plan, k):
        bn = 32 if n % 32 == 0 else 16
        _mxfp8_gemv_kernel[(n // bn, plan.rows)](
            act,
            act,
            weight,
            weight_scale,
            out,
            tiles,
            plan.max_ctas,
            N=n,
            K=k,
            TILE=tile,
            BN=bn,
            BK=128,
            A_BF16=True,
            num_warps=2,
        )
        return out
    bm = min(max(tile, 16), 64)
    sub = tile // bm if tile >= bm else 1
    _mxfp8_bmm_bf16_kernel[(n // 64, plan.max_ctas * sub)](
        act,
        weight,
        weight_scale,
        out,
        tiles,
        plan.max_ctas,
        N=n,
        K=k,
        TILE=tile,
        BM=bm,
        BN=64,
        num_warps=4,
        num_stages=2,
    )
    return out


def _expert_gemm(experts, act, name, tiles, plan):
    if experts.nvfp4:
        return gemm_nvfp4(
            act,
            getattr(experts, f"{name}_weight"),
            getattr(experts, f"{name}_weight_scale"),
            getattr(experts, f"{name}_weight_amax"),
            getattr(experts, f"{name}_weight_outer_scale_denominator"),
            tiles,
            plan,
        )
    return gemm_mxfp8(
        act,
        getattr(experts, f"{name}_weight"),
        getattr(experts, f"{name}_weight_scale"),
        tiles,
        plan,
    )


def moe_forward(self, x: torch.Tensor, pass_idx: int = 0) -> torch.Tensor:
    """`Inkling2MoE.forward` on bf16 rows (see the module docstring)."""
    x = x.contiguous()
    gate, experts = self.gate, self.experts
    plan = experts.plan(x.shape[0])
    is_padding = None
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
    routing, act = route_bf16(
        x,
        gate.weight,
        gate.bias,
        gate.counters,
        experts.counts,
        gate.topk,
        gate.route_scale,
        experts.expert_bounds,
        plan,
        is_padding,
    )
    src2dst, gammas, rows, tiles = routing
    gate_up = _expert_gemm(experts, act, "w13", tiles, plan)
    activated = swiglu_bf16(
        gate_up,
        gammas,
        experts.gate_cap,
        experts.up_cap,
        (rows, tiles[2 * plan.max_ctas + 2 :]),
        plan.tile,
    )
    del gate_up
    down = _expert_gemm(experts, activated, "w2", tiles, plan)
    out = combine(down, src2dst, experts.topk, x.dtype)
    return out if experts.ep_size == 1 else tensor_model_parallel_all_reduce(out)
