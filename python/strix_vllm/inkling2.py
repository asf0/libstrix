# SPDX-License-Identifier: MIT
"""Inkling2 (ROCm port) attention projections on dense_w8's int8 copies.

The ROCm Inkling2 attention multiplies by its `qkvr` weight, by row slices of
it (the query side, k and v of full layers and of loop passes after the
first) and by `wo_ud`, through `inkling2_linear`. dense_w8 gives each of these
layers an int8 copy at load (per-row groups, so a row slice of the copy is the
copy of the slice); decode-sized batches read it here, the rest keep bf16.
"""

from __future__ import annotations

import torch

from strix_vllm import dense_w8, disabled


def register() -> None:
    if disabled("dense_w8") or disabled("inkling2_linear"):
        return
    try:
        from vllm.models.inkling2.amd import attention
    except ImportError:
        return
    if not hasattr(attention, "inkling2_linear"):
        return
    bf16_linear = attention.inkling2_linear

    def inkling2_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base = weight if weight._base is None else weight._base
        entry = dense_w8.lookup(base)
        x_2d = x.reshape(-1, x.shape[-1])
        if entry is None or not dense_w8.skinny_ok(x_2d.size(0), entry[0]):
            return bf16_linear(x, weight)
        w_q, w_s, group, cu = entry
        rows = weight.size(0)
        first = (weight.data_ptr() - base.data_ptr()) // (
            base.stride(0) * base.element_size()
        )
        out = torch.ops.strix.w8a16_skinny_gemm(
            w_q[first : first + rows], x_2d, w_s[first : first + rows], None, cu, group
        )
        return out.reshape(*x.shape[:-1], rows)

    attention.inkling2_linear = inkling2_linear
