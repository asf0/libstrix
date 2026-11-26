# SPDX-License-Identifier: MIT
"""Inkling2 (ROCm port) on gfx1151: the MoE on bf16 rows (`inkling2_moe`),
the attention projections on dense_w8's int8 copies, and the attention
itself - prefill (`inkling_rel_attn_prefill`), decode/verify
(`inkling_rel_attn_decode`) and the K/V conv write (`inkling2_kv_conv`) -
on a bf16 or an int8 (`int8_per_token_head`) KV cache. The port's own kernels
write and read bf16 only: with an int8 cache every batch runs here.

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
    _register_attention()
    _register_decode()
    _register_kv_conv()
    _register_moe()
    _register_loop()
    _register_linear()
    _register_head()
    _register_draft_head()


def _kv_cache_ok(key_cache: torch.Tensor) -> bool:
    return key_cache.shape[3] == 128 and key_cache.dtype in (torch.bfloat16, torch.int8)


def _attention_ok(q, key_cache, gate, block_table) -> bool:
    """Whether the prefill and decode attention kernels take the batch."""
    return (
        _kv_cache_ok(key_cache)
        and q.dtype == torch.bfloat16
        and q.shape[1] == 3 * key_cache.shape[2]
        and gate.stride(0) % 8 == 0
        and gate.data_ptr() % 16 == 0
        and block_table.dtype == torch.int32
    )


def _check_fallback(key_cache: torch.Tensor, what: str) -> None:
    """Before a batch goes to the port's kernel."""
    if key_cache.dtype == torch.int8:
        raise RuntimeError(
            f"Inkling 2 int8 KV: no {what} kernel takes this batch (key cache "
            f"{tuple(key_cache.shape)}), and the port's kernels are bf16 only"
        )


# Query tiles of 16 per block (GQA 3, 32-key steps): full layers 4, window
# layers 2. Inkling 2 Mini, 8k-token chunk over 16k keys: full 88 -> 37-39 ms,
# window 10.0 -> 5.0 ms (attention + gate/sink, against the Triton kernel).
_REL_Q_TILES_FULL = 4
_REL_Q_TILES_WINDOW = 2
_REL_BLOCK_N = 32


_REL_PROJ_T: dict[int, torch.Tensor] = {}


def _rel_attention(
    q, key_cache, value_cache, rel, gate, sinks, result, block_table, cache_seqlens,
    cu_seqlens_q, num_actual_tokens, softmax_scale, window, rel_proj_t, k_scale,
    v_scale,
) -> None:
    """strix::inkling_rel_attn_prefill over the requests' tokens (later rows
    of `result` are zeroed); `rel` is the sheared logits, or with
    `rel_proj_t` the clipped r; `k_scale` / `v_scale` for an int8 cache."""
    nt = num_actual_tokens
    q_tiles = _REL_Q_TILES_WINDOW if window >= 0 else _REL_Q_TILES_FULL
    tile = 16 * q_tiles
    cu = cu_seqlens_q.to(torch.int32)
    max_tiles = (nt + tile - 1) // tile + cu.numel() - 1
    torch.ops.strix.inkling_rel_attn_prefill(
        result, q, key_cache, value_cache, k_scale, v_scale, rel, gate, sinks, cu,
        cache_seqlens.to(torch.int32), block_table, cu, max_tiles,  # tile_offsets: unread
        softmax_scale, window, q_tiles, _REL_BLOCK_N, rel_proj_t,
    )
    result[nt:].zero_()


def _register_attention() -> None:
    """Prefill (and mixed-batch) relative attention on the HIP kernel, with
    the output gate and sink in its epilogue. Window layers compute their
    relative logits in the kernel (the port's prologue then writes only the
    clipped r, not the 315 MB sheared table of a 5k-token chunk); full layers
    read the table (their logits only cover the last 1024 of many keys, and
    the in-kernel version's registers and LDS slowed every step)."""
    if disabled("inkling2_attention"):
        return
    try:
        from vllm.models.inkling2.amd import attention
    except ImportError:
        return
    if not hasattr(attention, "gated_rel_attention"):
        return
    triton_attention = attention.gated_rel_attention

    def gated_rel_attention(
        q, key_cache, value_cache, sheared_bias, gate, sinks, result, *,
        block_table, cache_seqlens, cu_seqlens_q, max_seqlen_q,
        num_actual_tokens, softmax_scale, window, k_scale=None, v_scale=None,
    ) -> None:
        if not _attention_ok(q, key_cache, gate, block_table) or (
            key_cache.shape[1] % _REL_BLOCK_N
        ):
            _check_fallback(key_cache, "prefill")
            return triton_attention(
                q, key_cache, value_cache, sheared_bias, gate, sinks, result,
                block_table=block_table, cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
                num_actual_tokens=num_actual_tokens, softmax_scale=softmax_scale,
                window=window,
            )
        _rel_attention(
            q, key_cache, value_cache, sheared_bias, gate, sinks, result, block_table,
            cache_seqlens, cu_seqlens_q, num_actual_tokens, softmax_scale, window,
            None, k_scale, v_scale,
        )

    attention.gated_rel_attention = gated_rel_attention
    if not hasattr(attention, "rel_attention_prefill"):
        return

    def rel_attention_prefill(
        q, key_cache, value_cache, r, rel_proj, gate, sinks, result, *,
        block_table, cache_seqlens, cu_seqlens_q, max_seqlen_q,
        num_actual_tokens, softmax_scale, window, k_scale=None, v_scale=None,
    ) -> None:
        proj_t = _REL_PROJ_T.get(rel_proj.data_ptr())
        if proj_t is None:
            proj_t = rel_proj.t().contiguous()
            _REL_PROJ_T[rel_proj.data_ptr()] = proj_t
        _rel_attention(
            q, key_cache, value_cache, r, gate, sinks, result, block_table,
            cache_seqlens, cu_seqlens_q, num_actual_tokens, softmax_scale, window,
            proj_t, k_scale, v_scale,
        )

    def supports(num_heads, num_kv_heads, head_dim, block_size, extent, window) -> bool:
        return (
            window >= 0
            and head_dim == 128
            and num_heads == 3 * num_kv_heads
            and block_size % _REL_BLOCK_N == 0
        )

    rel_attention_prefill.supports = supports
    attention.rel_attention_prefill = rel_attention_prefill


def _verify_plan(rows, reqs, key_cache, is_local) -> tuple[int, int]:
    """Query tokens per block of strix::inkling_rel_attn_decode and key
    segments per block of several (the port's split count is fitted to its own
    program). The op takes any batch, so a CUDA graph may replay both."""
    if rows <= 1:
        return 1, 1
    q_rows = 3 if rows <= 3 else 5
    # One wave per segment: enough of them to stream at DRAM speed (int8 keys
    # come twice as fast), and few enough for all to be resident at once.
    waves = 192 if key_cache.dtype == torch.int8 else 96 if is_local else 160
    blocks = key_cache.shape[2] * reqs * -(-rows // q_rows)
    return q_rows, min(64, max(1, waves // blocks))


def _register_decode() -> None:
    """Decode and verify relative attention on the HIP kernels: the relative
    logits from the sheared table, the output gate and the per-token sink."""
    if disabled("inkling2_rel_decode"):
        return
    try:
        from vllm.models.inkling2.amd import attention
    except ImportError:
        return
    if not hasattr(attention, "rel_decode_attention"):
        return
    triton_decode = attention.rel_decode_attention

    def rel_decode_attention(
        q, key_cache, value_cache, rel_logits, gate, sinks, out, *,
        block_table, cache_seqlens, cu_seqlens_q, max_query_len,
        num_actual_tokens, softmax_scale, is_local, num_splits,
        k_scale=None, v_scale=None,
    ) -> None:
        if not _attention_ok(q, key_cache, gate, block_table) or (
            # More than one 5-token tile re-reads a deep bf16 KV (8 rows,
            # 131k keys: 2.5 ms against the port's 2.3).
            max_query_len > 5 and not is_local and key_cache.dtype == torch.bfloat16
        ):
            _check_fallback(key_cache, "decode")
            return triton_decode(
                q, key_cache, value_cache, rel_logits, gate, sinks, out,
                block_table=block_table, cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q, max_query_len=max_query_len,
                num_actual_tokens=num_actual_tokens, softmax_scale=softmax_scale,
                is_local=is_local, num_splits=num_splits,
            )
        # The sheared table is 256 wider than the window of a local layer.
        window = rel_logits.shape[2] - 257 if is_local else -1
        q_rows, splits = _verify_plan(
            max_query_len, cache_seqlens.shape[0], key_cache, is_local
        )
        torch.ops.strix.inkling_rel_attn_decode(
            out, q, key_cache, value_cache, k_scale, v_scale, rel_logits, gate,
            sinks, cu_seqlens_q.to(torch.int32), cache_seqlens.to(torch.int32),
            block_table, softmax_scale, splits, window, q_rows,
        )

    attention.rel_decode_attention = rel_decode_attention


_KV_CONV_TAPS: dict[tuple, torch.Tensor] = {}


def _kv_conv_taps(weight: torch.Tensor, dynamic: bool) -> torch.Tensor:
    """fp32 `[C, 4]` taps for strix::inkling2_kv_conv, made once per weight:
    static `[C, W]` rounded to bf16 as the port's kernel reads them, dynamic
    `[W, C]` (the static part) transposed."""
    key = (weight.data_ptr(), dynamic)
    taps = _KV_CONV_TAPS.get(key)
    if taps is None:
        taps = weight.t() if dynamic else weight.to(torch.bfloat16)
        taps = taps.float().contiguous()
        _KV_CONV_TAPS[key] = taps
    return taps


def _register_kv_conv() -> None:
    """The attention prologue's K/V convs on the HIP kernel: prefill batches
    (5k tokens: 1.15-1.19 -> 0.78-0.85 ms a layer), every batch of an int8
    cache (the kernel writes the scales inline)."""
    if disabled("inkling2_kv_conv"):
        return
    try:
        from vllm.models.inkling2.amd.ops import sconv
    except ImportError:
        return
    if not hasattr(sconv, "kv_conv_prefill"):
        return

    def aligned(*tensors) -> bool:
        return all(t.data_ptr() % 16 == 0 for t in tensors)

    def kv_conv_prefill(
        k, v, k_weight, v_weight, k_dynamic, v_dynamic, cache, key_cache,
        value_cache, kv_slot_mapping, positions, seq_idx, slot_mapping,
        block_table, query_start, *, scale, clip_threshold, clip_eps, norm_eps,
    ) -> bool:
        dynamic = k_dynamic is not None
        int8 = key_cache.dtype == torch.int8
        page, n = key_cache.shape[1], cache.shape[2]
        if (
            not _kv_cache_ok(key_cache)
            or k.dtype != torch.bfloat16
            or cache.dtype != torch.bfloat16
            or (k_weight.shape[0] if dynamic else k_weight.shape[1]) != 4
            or page & (page - 1)
            or n & (n - 1)
            or k.stride(0) % 8
            or v.stride(0) % 8
            or any(st % 8 for st in cache.stride()[:3])
            or not aligned(k, v, cache)
            or (not int8 and not aligned(key_cache, value_cache))
            or (dynamic and (k_dynamic.dtype != torch.bfloat16
                             or k_dynamic.stride(0) % 4 or v_dynamic.stride(0) % 4
                             or k_dynamic.data_ptr() % 8 or v_dynamic.data_ptr() % 8))
        ):
            _check_fallback(key_cache, "K/V conv")
            return False
        torch.ops.strix.inkling2_kv_conv(
            k, v, _kv_conv_taps(k_weight, dynamic), _kv_conv_taps(v_weight, dynamic),
            k_dynamic, v_dynamic, cache, key_cache, value_cache, kv_slot_mapping,
            positions, seq_idx, slot_mapping, block_table, query_start, scale,
            clip_threshold, clip_eps, norm_eps,
        )
        return True

    sconv.kv_conv_prefill = kv_conv_prefill


def _register_draft_head() -> None:
    """DSpark draft base logits and draft layers from dense_w8's int8 copies
    for skinny batches (drafts are verified, so this only moves acceptance)."""
    if disabled("dense_w8") or disabled("inkling2_draft_head"):
        return
    try:
        from vllm.models.inkling2.amd import dspark
    except ImportError:
        return
    bf16_logits = dspark.Inkling2DSparkModel.compute_base_logits

    def compute_base_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w8 = getattr(self.lm_head, "_strix_w8", None)
        x_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        if w8 is None or not dense_w8.skinny_ok(x_2d.size(0), w8[0]):
            return bf16_logits(self, hidden_states)
        w_q, w_s, group, cu = w8
        return torch.ops.vllm.strix_dense_w8_apply(
            x_2d, self.lm_head.weight, w_q, w_s, None, cu, group
        )

    dspark.Inkling2DSparkModel.compute_base_logits = compute_base_logits
    if not hasattr(dspark, "draft_linear"):
        return
    bf16_linear = dspark.draft_linear

    def draft_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        entry = dense_w8.lookup(weight)
        x_2d = x.reshape(-1, x.shape[-1])
        if entry is None or not dense_w8.skinny_ok(x_2d.size(0), entry[0]):
            return bf16_linear(x, weight)
        w_q, w_s, group, cu = entry
        out = torch.ops.strix.w8a16_skinny_gemm(w_q, x_2d, w_s, None, cu, group)
        return out.reshape(*x.shape[:-1], weight.size(0))

    dspark.draft_linear = draft_linear


def _register_loop() -> None:
    """The loop controllers' blend. The strix-vllm fork's own blend raises
    (its Triton norms moved here), so with the kernels off the port's unfused
    blend is installed instead."""
    try:
        from vllm.models.inkling2.amd import loop

        from strix_vllm import inkling2_loop
    except ImportError:
        return
    loop.Inkling2LoopDecoderLayer.blend = (
        inkling2_loop.blend_unfused if disabled("inkling2_loop") else inkling2_loop.blend
    )


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
