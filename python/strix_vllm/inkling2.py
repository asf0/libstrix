# SPDX-License-Identifier: MIT
"""Inkling2 (ROCm port) on gfx1151: the MoE on bf16 rows (`inkling2_moe`) and
the attention projections on dense_w8's int8 copies.

The ROCm Inkling2 attention multiplies by its `qkvr` weight, by row slices of
it (the query side, k and v of full layers and of loop passes after the
first) and by `wo_ud`, through `inkling2_linear`. dense_w8 gives each of these
layers an int8 copy at load (per-row groups, so a row slice of the copy is the
copy of the slice); decode-sized batches read it here, the rest keep bf16.
"""

from __future__ import annotations

import os

import torch

from strix_vllm import dense_w8, disabled

# int4 copies (RTN, group 32) of the attention weights for batches of <= 5
# rows, on vLLM's skinny int4 GEMV: half the int8 bytes. Inkling 2 Mini decode
# 35.4 -> 40.2 tok/s; GSM8K 211 / 213 of 250 int8 / int4, McNemar p = 0.77.
# STRIX_INKLING2_W4=0 keeps the int8 copies.
_W4 = os.environ.get("STRIX_INKLING2_W4", "1") == "1"
# Experimental (off): the same for the LM head (201k x 2304 on Mini).
_W4_HEAD = os.environ.get("STRIX_INKLING2_W4_HEAD", "0") == "1"
_W4_GROUP = 32
_W4_MAX_ROWS = 5


def register() -> None:
    _register_moe()
    _register_loop()
    _register_linear()
    _register_head()


def _register_loop() -> None:
    if disabled("inkling2_loop"):
        return
    try:
        from vllm.models.inkling2.amd import loop

        from strix_vllm import inkling2_loop
    except ImportError:
        return
    loop.Inkling2LoopDecoderLayer.blend = inkling2_loop.blend


def _register_moe() -> None:
    if disabled("inkling2_moe"):
        return
    try:
        from vllm.models.inkling2.amd import moe

        from strix_vllm import inkling2_moe
    except ImportError:
        return
    moe.Inkling2MoE.forward = inkling2_moe.moe_forward


def _register_linear() -> None:
    if disabled("dense_w8") or disabled("inkling2_linear"):
        return
    try:
        from vllm.models.inkling2.amd import attention
    except ImportError:
        return
    if not hasattr(attention, "inkling2_linear"):
        return
    bf16_linear = attention.inkling2_linear

    def w4_copy(base: torch.Tensor) -> tuple:
        return _w4_copy(base)

    def inkling2_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base = weight if weight._base is None else weight._base
        x_2d = x.reshape(-1, x.shape[-1])
        if _W4 and base.size(1) % _W4_GROUP == 0:
            w_q, w_s, cu = w4_copy(base)
            if x_2d.size(0) <= _W4_MAX_ROWS:
                import vllm._custom_ops as ops

                rows = weight.size(0)
                first = (weight.data_ptr() - base.data_ptr()) // (
                    base.stride(0) * base.element_size()
                )
                out = ops.wvSplitK_int4_g(
                    w_q[first : first + rows],
                    x_2d,
                    w_s[first : first + rows],
                    cu,
                    _W4_GROUP,
                    None,
                    None,
                )
                return out.reshape(*x.shape[:-1], rows)
        entry = dense_w8.lookup(base)
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


_W4_COPIES: dict[int, tuple] = {}


def _w4_copy(base: torch.Tensor) -> tuple:
    """Group-32 int4 copy of bf16 `base` `[N, K]` (packed for vLLM's skinny
    int4 GEMV), made at its first use (the profile run, so its memory is
    counted) and quantized in row chunks to bound the fp32 temporary."""
    entry = _W4_COPIES.get(base.data_ptr())
    if entry is not None:
        return entry
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        pack_skinny_int4,
    )
    from vllm.utils.platform_utils import num_compute_units

    n, k = base.shape
    q = torch.empty((n, k), dtype=torch.int32, device=base.device)
    scale = torch.empty((n, k // _W4_GROUP), dtype=base.dtype, device=base.device)
    for r0 in range(0, n, 16384):
        wf = base[r0 : r0 + 16384].float().view(-1, k // _W4_GROUP, _W4_GROUP)
        s = (wf.abs().amax(-1) / 7.0).clamp_min(1e-12)
        q[r0 : r0 + 16384] = (
            torch.round(wf / s[..., None]).clamp(-8, 7).view(-1, k) + 8
        ).to(torch.int32)
        scale[r0 : r0 + 16384] = s.to(base.dtype)
    entry = (pack_skinny_int4(q), scale, num_compute_units())
    del q
    _W4_COPIES[base.data_ptr()] = entry
    return entry


def _register_head() -> None:
    """With STRIX_INKLING2_W4_HEAD=1: decode logits of the Inkling2 fp32 head
    from an int4 copy (bf16 output, upcast)."""
    if not _W4_HEAD:
        return
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    apply_head = LogitsProcessor._apply_head

    def inkling2_apply_head(self, lm_head, hidden_states, embedding_bias):
        x_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        weight = getattr(lm_head, "weight", None)
        if (
            self.head_dtype != torch.float32
            or embedding_bias is not None
            or weight is None
            or weight.dtype != torch.bfloat16
            or weight.size(1) % _W4_GROUP
            or x_2d.size(0) > _W4_MAX_ROWS
        ):
            return apply_head(self, lm_head, hidden_states, embedding_bias)
        import vllm._custom_ops as ops

        w_q, w_s, cu = _w4_copy(weight)
        out = ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu, _W4_GROUP, None, None)
        return out.float().reshape(*hidden_states.shape[:-1], -1)

    LogitsProcessor._apply_head = inkling2_apply_head
