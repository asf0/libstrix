# SPDX-License-Identifier: Apache-2.0
# Moved from the asf0/strix-vllm fork (dev, "Qwen4Exp on gfx1151: prefill 790
# -> ~1600 tok/s"): the gfx1151 kernels it added to
# vllm/models/qwen4_exp/amd/ops/hc.py, which is vLLM's file.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next prefill kernels in Triton, used by strix_vllm.qwen4_exp."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _hc_combine_norm_rows_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    # One program per row, streams in turn: the block output and shared norm
    # weight are read once per row instead of once per stream.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HC_DIM
    block = tl.load(block_ptr + row * stride_block + offs, mask, other=0.0)
    block = block.to(tl.float32)
    if W_SHARED:
        w_shared = tl.load(w_ptr + offs, mask, other=0.0).to(tl.float32)
    for stream in tl.static_range(HC):
        offs_s = stream * HC_DIM + offs
        res = tl.load(res_ptr + row * stride_res + offs_s, mask, other=0.0)
        inj = tl.load(inj_ptr + row * stride_inj + stream).to(tl.float32)
        inj = 2.0 * tl.sigmoid(inj / HC)
        # Same rounding point as vLLM's _hc_combine_norm_kernel.
        out = (res.to(tl.float32) + block * inj).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_out + offs_s, out, mask=mask)
        out = out.to(tl.float32)
        rrms = tl.rsqrt(tl.sum(out * out, axis=0) / HC_DIM + EPS)
        if W_SHARED:
            w = w_shared
        else:
            w = tl.load(w_ptr + offs_s, mask, other=0.0).to(tl.float32)
        y = out * rrms
        y += y * w
        tl.store(y_ptr + row * stride_y + offs_s, y, mask)


def hc_combine_norm_rows(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's hc_combine_norm with one program per row (gfx1151, 16k rows:
    9.2 -> 5.2 ms). Same layout requirements as vLLM's _hc_combine_norm."""
    n, dim = residual.shape
    hc_dim = dim // hc_count
    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    _hc_combine_norm_rows_kernel[(n,)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        out.stride(0),
        y.stride(0),
        hc_dim,
        hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        BLOCK_SIZE=triton.next_power_of_2(hc_dim),
        num_warps=16,
    )
    return out, y
