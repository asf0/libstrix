# SPDX-License-Identifier: MIT
"""Qwen3.8-Flash-Next (qwen4_exp) prefill swaps.

qsa_union: QSA sparse attention for a single-request prefill on
strix::qsa_union_attn_bf16.

Groups of 16 consecutive query rows attend over the union of their selected
tokens (csrc/attn/qsa_union_bf16.hip): 1.51x the RDNA-tuned Triton per-row
kernel on a 26k-token prompt's real selections, more as the context grows.
Everything else (decode, several requests, other shapes, positions past the
128k union bitmap) keeps vLLM's Triton kernel.

vLLM's AMD QSA module imports `qsa_sparse_paged_attention` from
`vllm.models.qwen4_exp.amd.ops.qsa` at call time, so swapping the module
attribute is enough.

qsa_indexer: the QSA indexer's paged scores for a single-request prefill on
strix::qsa_indexer_bf16 (1.7-1.9x vLLM's tl.dot kernel, 4k to 128k context);
vLLM's qsa_select_paged_tokens calls `qsa_mqa_paged` from its module globals.

hc_up_gate_mix: the hyper-connection up projection fused with the gated mix
(strix::hc_up_gate_mix_bf16) from 512 rows: 5.2 vs 6.7 ms at 16k rows and no
[M, 4H] gate in memory. vLLM's qwen4_exp_hc_up_gate_mix op calls
`hc_up_gate_mix_impl` of its ops module per call; smaller batches keep the
unfused path (vLLM's skinny GEMMs at decode).
"""

import math
import os

import torch

from strix_vllm import disabled

# STRIX_QSA_UNION_MIN_ROWS lowers the floor for accuracy runs on short prompts.
_MIN_ROWS = int(os.environ.get("STRIX_QSA_UNION_MIN_ROWS", "1024"))
_MAX_POSITIONS = 131072


def _union_shape_ok(q, k_cache, v_cache, logical_indices, block_table) -> bool:
    """A single-request prefill the union kernel takes, positions aside."""
    return (
        q.shape[0] >= _MIN_ROWS
        and block_table.dim() == 2
        and block_table.shape[0] == 1
        and q.dtype == torch.bfloat16
        and k_cache.dtype == v_cache.dtype
        and k_cache.dtype in (torch.bfloat16, torch.int8)
        and q.dim() == 3
        and q.shape[2] == 256
        and q.stride(2) == 1
        and k_cache.dim() == 4
        and k_cache.stride(3) == 1
        and v_cache.stride(3) == 1
        and q.shape[1] == 12 * k_cache.shape[2]
        and logical_indices.dtype == torch.int32
        and logical_indices.stride(1) == 1
    )


def _dequant_pages(cache, scale, block_table, tokens):
    """The request's first `tokens` positions of an int8_per_token_head cache
    as a contiguous bf16 cache [pages, block, Hkv, 256], scaled in fp32."""
    pages = block_table[0, : -(-tokens // cache.shape[1])].long()
    return (cache[pages].float() * scale[pages][..., None]).to(torch.bfloat16)


_HC_MIN_ROWS = 512
_F16 = os.environ.get("STRIX_QSA_UNION_F16", "0") == "1"
_UNION_ROWS = int(os.environ.get("STRIX_QSA_UNION_ROWS", "16"))


_IDX_MIN_ROWS = 128


def register() -> None:
    _register_qsa_union()
    _register_qsa_indexer()
    _register_hc_up_gate_mix()


def _register_qsa_indexer() -> None:
    if disabled("qsa_indexer"):
        return
    try:
        from vllm.models.qwen4_exp.amd.ops import qsa as ops
    except ImportError:
        return
    orig = ops.qsa_mqa_paged

    def qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns=None,
        score_scale=None,
    ):
        columns = (
            page_table.shape[1] * k_cache.shape[1] if num_columns is None else num_columns
        )
        if not (
            q.shape[0] >= _IDX_MIN_ROWS
            and columns > 0
            and page_table.dim() == 2
            and page_table.shape[0] == 1
            and q.dtype == k_cache.dtype == torch.bfloat16
            and q.dim() == 3
            and q.shape[1] == 4
            and q.shape[2] == 128
            and q.stride(2) == 1
            and q.stride(0) % 8 == 0
            and q.stride(1) % 8 == 0
            and k_cache.dim() == 4
            and k_cache.shape[2] == 1
            and k_cache.stride(3) == 1
            and query_positions.dtype == torch.int64
            and sequence_lengths.dtype == torch.int32
            and compress_ratio > 0
        ):
            return orig(
                q,
                k_cache,
                page_table,
                token_to_req,
                query_positions,
                sequence_lengths,
                compress_ratio,
                num_columns,
                score_scale,
            )
        divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
        logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
        visible = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
        torch.ops.strix.qsa_indexer_bf16(
            logits,
            visible,
            q,
            k_cache,
            page_table[0],
            query_positions.contiguous(),
            sequence_lengths,
            compress_ratio,
            float(divisor),
        )
        return logits, visible

    ops.qsa_mqa_paged = qsa_mqa_paged


def _register_hc_up_gate_mix() -> None:
    if disabled("hc_up_gate_mix"):
        return
    try:
        from vllm.models.qwen4_exp.amd.ops import hc
    except ImportError:
        return
    unfused = getattr(hc, "hc_up_gate_mix_impl", None)
    if unfused is None:
        return

    from strix_vllm import dense_w8

    def hc_up_gate_mix(x, lowrank, weight, hc_count):
        if x.shape[0] <= dense_w8.MAX_SKINNY_BATCH_SIZE:
            w8 = dense_w8.lookup(weight)
            if (
                w8 is not None
                and lowrank.is_contiguous()
                and dense_w8.skinny_ok(x.shape[0], w8[0])
            ):
                # Decode: the int8 copy dense_w8 made of input_mix_weight_up.
                w_q, w_s, group, cu = w8
                gate = torch.ops.strix.w8a16_skinny_gemm(w_q, lowrank, w_s, None, cu, group)
                return hc._hc_gate_mix(x, gate, hc_count)
        if not (
            hc_count == 4
            and x.shape[0] >= _HC_MIN_ROWS
            and x.dtype == lowrank.dtype == weight.dtype == torch.bfloat16
            and lowrank.shape[1] % 64 == 0
            and x.stride(1) == lowrank.stride(1) == weight.stride(1) == 1
            and lowrank.stride(0) % 8 == 0
            and weight.stride(0) % 8 == 0
        ):
            return unfused(x, lowrank, weight, hc_count)
        y = x.new_empty((x.shape[0], x.shape[1] // hc_count))
        torch.ops.strix.hc_up_gate_mix_bf16(y, x, lowrank, weight)
        return y

    hc.hc_up_gate_mix_impl = hc_up_gate_mix


def _register_qsa_union() -> None:
    if disabled("qsa_union"):
        return
    try:
        from vllm.models.qwen4_exp.amd.ops import qsa as ops
    except ImportError:
        return
    orig = ops.qsa_sparse_paged_attention

    def qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, out=None,
        k_scale=None, v_scale=None,
    ):
        int8 = k_cache.dtype == torch.int8
        scales = {} if k_scale is None else {"k_scale": k_scale, "v_scale": v_scale}
        if (int8 and k_scale is None) or not _union_shape_ok(
            q, k_cache, v_cache, logical_indices, block_table
        ):
            return orig(
                q, k_cache, v_cache, logical_indices, block_table, token_to_req, out,
                **scales,
            )
        # The union bitmap covers positions < 131072. A cache that cannot hold
        # more needs no check; otherwise the request's own length decides
        # (the largest selected position, one host sync), so a server with a
        # longer --max-model-len still takes the union kernel up to 128k.
        # An int8 cache and STRIX_QSA_UNION_F16 need that length anyway: the
        # kernel gathers from a compact fp16 copy of the visible K/V.
        capacity = block_table.shape[1] * k_cache.shape[1]
        tokens = 0
        if int8 or _F16 or capacity > _MAX_POSITIONS:
            tokens = int(logical_indices.max()) + 1
            if tokens <= 0:  # nothing selected: the Triton kernel writes zeros
                return orig(
                    q, k_cache, v_cache, logical_indices, block_table, token_to_req,
                    out, **scales,
                )
        if tokens > _MAX_POSITIONS or (int8 and _UNION_ROWS != 16):
            if not int8:
                return orig(
                    q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
                )
            # Past the union bitmap, int8: the Triton kernel reads bf16 K/V 2.3x
            # faster than int8 (16k rows, 2k selected, 32k context: 227 vs
            # 531 ms per layer), so dequantize the request's pages once.
            k16 = _dequant_pages(k_cache, k_scale, block_table, tokens)
            v16 = _dequant_pages(v_cache, v_scale, block_table, tokens)
            pages = torch.arange(k16.shape[0], device=q.device, dtype=torch.int32)
            return orig(q, k16, v16, logical_indices, pages[None], token_to_req, out)
        if out is None:
            out = torch.empty_like(q)
        torch.ops.strix.qsa_union_attn_bf16(
            out, q, k_cache, v_cache, logical_indices, block_table[0],
            tokens if (int8 or _F16) else 0, k_scale, v_scale,
        )
        return out

    ops.qsa_sparse_paged_attention = qsa_sparse_paged_attention
