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

import os

import strix  # noqa: F401  (registers torch.ops.strix)
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
FUSED_SWIGLU = os.environ.get("STRIX_INKLING2_FUSED_SWIGLU", "1") == "1"
HIP_GEMV = os.environ.get("STRIX_INKLING2_HIP_GEMV", "1") == "1"
# NVFP4 tile GEMV up to 152 expert tiles (64 tokens). Zipf-routed w13 at
# 1 / 8 / 25 tokens: 0.14 / 0.64 / 1.32 ms against 0.17 / 0.80 / 1.39 for a
# row per program; the grouped dot kernel takes over from about 96 tokens.
NVFP4_TILE_GEMV_MAX_CTAS = 152


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
def _pt_bytes(expert, wrows, byte, N: tl.constexpr, K: tl.constexpr):
    """Offsets of NVFP4 bytes `byte` (index in a K / 2 row) of weight rows
    `wrows` in the pre-tiled layout (`pretile_nvfp4`)."""
    tile = (expert * (N // 128) + wrows // 128) * (K // 64) + byte // 32
    return tile * 4096 + (wrows % 128) * 32 + byte % 32


@triton.jit
def _pt_scales(expert, wrows, group, N: tl.constexpr, K: tl.constexpr):
    """Offsets of e4m3 block scales `group` (16 columns each) of weight rows
    `wrows` in the pre-tiled layout."""
    tile = (expert * (N // 128) + wrows // 128) * (K // 64) + group // 4
    return tile * 512 + (wrows % 128) * 4 + group % 4


@triton.jit
def _nvfp4_gemv_tile_kernel(
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
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # NVFP4 GEMV over all routed rows of an expert tile (decode, verify and
    # small batches): each K step of the expert's weight rows is loaded and
    # decoded once and reduced against every routed row on the vector units,
    # at memory speed where `tl.dot` on gfx1151 is not (a sequence's tokens
    # often share experts, so tiles hold several rows).
    tile = tl.program_id(1)
    if tile >= tl.load(Tiles + 2 * max_ctas):
        return
    row0 = tile * TILE
    num_rows = tl.load(Tiles + max_ctas + tile) - row0
    expert = tl.load(Tiles + tile).to(tl.int64)
    outer = tl.load(WAmax + expert) / tl.load(WDenominator)
    cols = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    half = tl.arange(0, BK // 2)
    tile_rows = tl.arange(0, TILE)
    acc = tl.zeros((TILE, BN), tl.float32)
    for k in range(0, K, BK):
        byte = k // 2 + half[None, :]
        packed = tl.load(W + _pt_bytes(expert, cols[:, None], byte, N, K)).to(tl.int32)
        scale = _e4m3_f32(
            tl.load(WScale + _pt_scales(expert, cols[:, None], byte // 8, N, K)).to(
                tl.int32
            )
        )
        w_even = _e2m1_f32_bits(packed & 15) * scale
        w_odd = _e2m1_f32_bits(packed >> 4) * scale
        for r in range(num_rows):
            a_row = A + (row0 + r).to(tl.int64) * K + k
            a_even = tl.load(a_row + 2 * half).to(tl.float32)
            a_odd = tl.load(a_row + 2 * half + 1).to(tl.float32)
            part = tl.sum(w_even * a_even[None, :] + w_odd * a_odd[None, :], axis=1)
            acc += tl.where(tile_rows[:, None] == r, part[None, :], 0.0)
    out = Out + (row0 + tile_rows[:, None]).to(tl.int64) * N + cols[None, :]
    tl.store(out, (acc * outer).to(tl.bfloat16), tile_rows[:, None] < num_rows)


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
    # nibbles in two K=16 dots. For tiles too small for `nvfp4_moe_wmma`.
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
    cols = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    j = tl.arange(0, 16)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + 2 * j[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, 32):
        a_even = tl.load(a_ptrs + k, row_mask, other=0.0)
        a_odd = tl.load(a_ptrs + k + 1, row_mask, other=0.0)
        byte = k // 2 + j[None, :]
        packed = tl.load(W + _pt_bytes(expert, cols[:, None], byte, N, K)).to(tl.int32)
        scale = _e4m3_f32(
            tl.load(WScale + _pt_scales(expert, cols[:, None], byte // 8, N, K)).to(
                tl.int32
            )
        )
        scale = (scale * outer).to(tl.bfloat16)
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
    # A routed row and BN output columns per program, reduced over K on the
    # vector units, for MXFP8 weights (e4m3, an e8m0 scale per 32).
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
    RowToken,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    LOCAL_POW2: tl.constexpr,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
    COPY: tl.constexpr,
):
    """`_scatter_kernel` that copies the bf16 row instead of quantizing it
    (COPY), or records each layout row's token for kernels that gather."""
    token = tl.program_id(0).to(tl.int64)
    dst = _place_rows(
        Starts, Ids, Ranks, Weights, Src2Dst, Gammas, Rows, token, TOPK, LOCAL_POW2
    )
    if COPY:
        head = tl.load(X + token * H + tl.arange(0, HEAD))
        if TAIL > 0:
            tail = tl.load(X + token * H + HEAD + tl.arange(0, TAIL))
        for slot in range(dst.shape[0]):
            row = _pick(dst, slot).to(tl.int64)
            if row >= 0:
                tl.store(Values + row * H + tl.arange(0, HEAD), head)
                if TAIL > 0:
                    tl.store(Values + row * H + HEAD + tl.arange(0, TAIL), tail)
    else:
        tl.store(RowToken + dst, token.to(tl.int32), dst >= 0)


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
    gather=False,
):
    """The port's unfused `route`, scattering bf16 rows instead of MXFP8.
    With `gather`, returns each layout row's token (int32 `[rows]`) in place
    of the scattered rows, for kernels that read `x` directly."""
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
    if gather:
        values = torch.empty(rows, dtype=torch.int32, device=device)
    else:
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
        values,
        H=hidden,
        TOPK=topk,
        LOCAL_POW2=local_pow2,
        HEAD=head,
        TAIL=tail,
        COPY=not gather,
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


def pretile_nvfp4(weight: torch.Tensor, scale: torch.Tensor):
    """NVFP4 bytes `[..., N, K / 2]` and e4m3 scales `[..., N, K / 16]` in the
    pre-tiled layout the kernels read: per 128 weight rows and 64 columns, the
    rows' 32 code bytes (byte j: column 2j low, 2j + 1 high), and their 4
    scale bytes, contiguous (`[N / 128, K / 64, 128, 32 | 4]`)."""
    *lead, n, k_half = weight.shape
    tiled = weight.reshape(*lead, n // 128, 128, k_half // 32, 32).transpose(-3, -2)
    s8 = scale.view(torch.uint8)
    tiled_s = s8.reshape(*lead, n // 128, 128, k_half // 32, 4).transpose(-3, -2)
    return (
        tiled.reshape(weight.shape),
        tiled_s.reshape(scale.shape).view(scale.dtype),
    )


def _pretile_experts(experts) -> None:
    """Rewrite the experts' NVFP4 weights and block scales in the pre-tiled
    layout, in place, one expert at a time (first forward: the profile run)."""
    if getattr(experts, "_strix_pretiled_nvfp4", False):
        return
    for name in ("w13", "w2"):
        weight = getattr(experts, f"{name}_weight")
        scale = getattr(experts, f"{name}_weight_scale")
        assert weight.shape[-2] % 128 == 0 and weight.shape[-1] % 32 == 0
        for e in range(weight.shape[0]):
            w, sc = pretile_nvfp4(weight[e], scale[e])
            weight[e].copy_(w)
            scale[e].view(torch.uint8).copy_(sc.view(torch.uint8))
    experts._strix_pretiled_nvfp4 = True


def gemm_nvfp4(act, weight, weight_scale, weight_amax, weight_denominator, tiles, plan):
    """Grouped NVFP4 GEMM; `weight` / `weight_scale` pre-tiled
    (`pretile_nvfp4`)."""
    _, n, k_half = weight.shape
    k = 2 * k_half
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tile = plan.tile
    if _wmma_ok(weight, plan):
        # fp16 WMMA (CIRU-derived): 31 / 29 TFLOPS (w13 / w2) at a 5k-token
        # prefill, against 24 for the Triton kernel below.
        torch.ops.strix.nvfp4_moe_wmma(
            out,
            act,
            weight,
            weight_scale,
            weight_amax,
            weight_denominator,
            tiles,
            plan.max_ctas,
            tile,
        )
        return out
    if HIP_GEMV and plan.tile == 16 and plan.max_ctas <= NVFP4_TILE_GEMV_MAX_CTAS:
        torch.ops.strix.nvfp4_moe_gemv(
            out,
            act,
            weight,
            weight_scale,
            weight_amax,
            weight_denominator,
            tiles,
            plan.max_ctas,
            tile,
        )
        return out
    if plan.tile == 16 and plan.max_ctas <= NVFP4_TILE_GEMV_MAX_CTAS and k % 128 == 0:
        # 16 columns, 1 warp, K step 128: w13 0.58 / 1.23 ms at 8 / 25
        # Zipf-routed tokens (64 columns, 4 warps: 0.69 / 1.49).
        _nvfp4_gemv_tile_kernel[(n // 16, plan.max_ctas)](
            act,
            weight,
            weight_scale.view(torch.uint8),
            weight_amax,
            weight_denominator,
            out,
            tiles,
            plan.max_ctas,
            N=n,
            K=k,
            TILE=tile,
            BN=16,
            BK=128,
            num_warps=1,
        )
        return out
    bm = min(max(tile, 16), 64)
    bn = 64
    sub = tile // bm if tile >= bm else 1
    _nvfp4_bmm_bf16_kernel[(n // bn, plan.max_ctas * sub)](
        act,
        weight,
        weight_scale.view(torch.uint8),
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
        num_warps=4,
        num_stages=2,
    )
    return out


def _mxfp8_scales_ok(weight_scale) -> bool:
    """Whether every e8m0 scale is in [105, 134], where `mxfp8_moe_wmma`
    applies it to fp16 weights exactly (checked once, kept on the tensor)."""
    ok = getattr(weight_scale, "_strix_wmma_scales_ok", None)
    if ok is None:
        s = weight_scale.view(torch.uint8)
        ok = bool((s.min() >= 105).item() and (s.max() <= 134).item())
        weight_scale._strix_wmma_scales_ok = ok
    return ok


def gemm_mxfp8(act, weight, weight_scale, tiles, plan):
    _, n, k = weight.shape
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    tile = plan.tile
    if tile >= 64 and n % 128 == 0 and k % 64 == 0 and _mxfp8_scales_ok(weight_scale):
        # fp16 WMMA, the NVFP4 kernel's MXFP8 variant.
        torch.ops.strix.mxfp8_moe_wmma(
            out, act, weight, weight_scale, tiles, plan.max_ctas, tile
        )
        return out
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


def _wmma_ok(weight, plan) -> bool:
    """Whether `gemm_nvfp4` runs `strix::nvfp4_moe_wmma` for this batch."""
    return plan.tile >= 64 and weight.shape[1] % 128 == 0 and weight.shape[2] % 32 == 0


def _swiglu_wmma_ok(experts, plan) -> bool:
    """Whether the w13 GEMM of this batch runs on a WMMA kernel."""
    weight = experts.w13_weight
    if experts.nvfp4:
        return _wmma_ok(weight, plan)
    n, k = weight.shape[1], weight.shape[2]
    return (
        plan.tile >= 64
        and n % 128 == 0
        and k % 64 == 0
        and _mxfp8_scales_ok(experts.w13_weight_scale)
    )


def gemm_nvfp4_swiglu(act, experts, tiles, plan, gammas, row_token=None):
    """w13 GEMM with the softcapped SwiGLU in its epilogue (`[rows, N / 2]`
    bf16, as `swiglu_bf16` of the GEMM), for batches on a WMMA kernel. With
    `row_token`, `act` is the token rows and layout row r reads
    `act[row_token[r]]`."""
    weight = experts.w13_weight
    out = torch.empty(
        (plan.rows, weight.shape[1] // 2), dtype=torch.bfloat16, device=act.device
    )
    caps = (float(experts.gate_cap or 0.0), float(experts.up_cap or 0.0))
    if not experts.nvfp4:
        torch.ops.strix.mxfp8_moe_wmma(
            out,
            act,
            weight,
            experts.w13_weight_scale,
            tiles,
            plan.max_ctas,
            plan.tile,
            gammas,
            *caps,
            row_token,
        )
        return out
    torch.ops.strix.nvfp4_moe_wmma(
        out,
        act,
        weight,
        experts.w13_weight_scale,
        experts.w13_weight_amax,
        experts.w13_weight_outer_scale_denominator,
        tiles,
        plan.max_ctas,
        plan.tile,
        gammas,
        *caps,
        row_token,
    )
    return out


def _expert_gemm(experts, act, name, tiles, plan):
    if experts.nvfp4:
        _pretile_experts(experts)
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
    fused = FUSED_SWIGLU and _swiglu_wmma_ok(experts, plan)
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
        gather=fused,
    )
    src2dst, gammas, rows, tiles = routing
    if fused:
        if experts.nvfp4:
            _pretile_experts(experts)
        # `act` holds each layout row's token: w13 reads the rows of x.
        activated = gemm_nvfp4_swiglu(x, experts, tiles, plan, gammas, act)
    else:
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
