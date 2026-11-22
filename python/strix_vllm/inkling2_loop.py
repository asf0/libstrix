# SPDX-License-Identifier: MIT
"""Inkling2 (ROCm port) loop message controllers for decode batches.

Each loop pass after the first blends a message into four streams through
rank-64 controllers: `hidden = pair @ down.T` (bf16) and
`gain_forget = hidden @ up.T` (fp32). For a handful of tokens these are
hipBLAS batched GEMMs of a few rows (about 17 us each, 96 per Inkling 2 Mini
token). Here: a split-K kernel for `down` and one for `up` that sums the
split partials, rounds `hidden` to bf16 as the GEMM does, and multiplies.
"""

from __future__ import annotations

import torch
from vllm.models.inkling2.amd.ops.loop import blend_rms_norm, pair_rms_norm
from vllm.triton_utils import tl, triton

# Batches up to this many tokens take the kernels (one 16-row dot tile).
MAX_TOKENS = 16
_K_SPLIT = 18


@triton.jit
def _down_kernel(
    Pair,
    Down,
    Partials,
    T,
    K: tl.constexpr,
    R: tl.constexpr,
    CHUNK: tl.constexpr,
    BK: tl.constexpr,
):
    """Partial `pair[s, :T, chunk] @ down[s, :, chunk].T` (fp32 `[16, R]`)."""
    s = tl.program_id(0)
    split = tl.program_id(1)
    rows = tl.arange(0, 16)
    ks = tl.arange(0, BK)
    rs = tl.arange(0, R)
    acc = tl.zeros((16, R), tl.float32)
    for k in range(split * CHUNK, (split + 1) * CHUNK, BK):
        p = tl.load(
            Pair + (s * T + rows[:, None]) * K + k + ks[None, :],
            rows[:, None] < T,
            other=0.0,
        )
        w = tl.load(Down + (s * R + rs[:, None]) * K + k + ks[None, :])
        acc = tl.dot(p, tl.trans(w), acc)
    tl.store(
        Partials
        + ((s * tl.num_programs(1) + split) * 16 + rows[:, None]) * R
        + rs[None, :],
        acc,
    )


@triton.jit
def _up_kernel(
    Partials,
    Up,
    Out,
    T,
    N: tl.constexpr,
    R: tl.constexpr,
    SPLITS: tl.constexpr,
    BN: tl.constexpr,
):
    """`round_bf16(sum of partials) @ up[s, cols].T` into fp32 `out[s, :T]`."""
    s = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    rows = tl.arange(0, 16)
    rs = tl.arange(0, R)
    hidden = tl.zeros((16, R), tl.float32)
    for split in range(SPLITS):
        hidden += tl.load(
            Partials + ((s * SPLITS + split) * 16 + rows[:, None]) * R + rs[None, :]
        )
    w = tl.load(Up + (s * N + cols[:, None]) * R + rs[None, :])
    out = tl.dot(hidden.to(tl.bfloat16), tl.trans(w))
    tl.store(Out + (s * T + rows[:, None]) * N + cols[None, :], out, rows[:, None] < T)


def controllers(pair: torch.Tensor, down: torch.Tensor, up: torch.Tensor):
    """fp32 `(S, T, 2D)` `[gain, forget]` of bf16 `(S, T, 2D)` `pair` through
    `down` `(S, R, 2D)` and `up` `(S, 2D, R)`."""
    num_streams, num_tokens, k = pair.shape
    rank = down.shape[1]
    n = up.shape[1]
    chunk = k // _K_SPLIT
    partials = pair.new_empty((num_streams, _K_SPLIT, 16, rank), dtype=torch.float32)
    _down_kernel[(num_streams, _K_SPLIT)](
        pair,
        down,
        partials,
        num_tokens,
        K=k,
        R=rank,
        CHUNK=chunk,
        BK=64,
        num_warps=4,
    )
    out = pair.new_empty((num_streams, num_tokens, n), dtype=torch.float32)
    _up_kernel[(num_streams, n // 64)](
        partials,
        up,
        out,
        num_tokens,
        N=n,
        R=rank,
        SPLITS=_K_SPLIT,
        BN=64,
        num_warps=4,
    )
    return out


def usable(pair: torch.Tensor, down: torch.Tensor, up: torch.Tensor) -> bool:
    k, rank, n = pair.shape[2], down.shape[1], up.shape[1]
    return (
        pair.shape[1] <= MAX_TOKENS
        and k % (_K_SPLIT * 64) == 0
        and n % 64 == 0
        and rank in (16, 32, 64, 128)
        and pair.is_contiguous()
        and down.is_contiguous()
        and up.is_contiguous()
    )


def blend(self, streams, dest, message, norm_weight, norm_eps):
    """`Inkling2LoopDecoderLayer.blend` with the controller kernels for
    decode batches."""
    pair = pair_rms_norm(dest, message)
    down = self.controller_down[streams]
    up = self.controller_gain_forget[streams]
    if usable(pair, down, up):
        gain_forget = controllers(pair, down, up)
    else:
        hidden = torch.bmm(pair, down.transpose(1, 2))
        gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
    return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)
