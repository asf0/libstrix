# SPDX-License-Identifier: MIT
"""qsa_union: Qwen3.8-Flash-Next (qwen4_exp) QSA sparse attention for a
single-request prefill on strix::qsa_union_attn_bf16.

Groups of 16 consecutive query rows attend over the union of their selected
tokens (csrc/attn/qsa_union_bf16.hip): 1.39x the RDNA-tuned Triton per-row
kernel on a 26k-token prompt's real selections, more as the context grows.
Everything else (decode, several requests, other shapes, positions past the
128k union bitmap) keeps vLLM's Triton kernel.

vLLM's AMD QSA module imports `qsa_sparse_paged_attention` from
`vllm.models.qwen4_exp.amd.ops.qsa` at call time, so swapping the module
attribute is enough.
"""

import os

import torch

from strix_vllm import disabled

# STRIX_QSA_UNION_MIN_ROWS lowers the floor for accuracy runs on short prompts.
_MIN_ROWS = int(os.environ.get("STRIX_QSA_UNION_MIN_ROWS", "1024"))
_MAX_POSITIONS = 131072


def _union_ok(q, k_cache, v_cache, logical_indices, block_table) -> bool:
    return (
        q.shape[0] >= _MIN_ROWS
        and block_table.dim() == 2
        and block_table.shape[0] == 1
        and block_table.shape[1] * k_cache.shape[1] <= _MAX_POSITIONS
        and q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
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


def register() -> None:
    if disabled("qsa_union"):
        return
    try:
        from vllm.models.qwen4_exp.amd.ops import qsa as ops
    except ImportError:
        return
    orig = ops.qsa_sparse_paged_attention

    def qsa_sparse_paged_attention(
        q, k_cache, v_cache, logical_indices, block_table, token_to_req, out=None
    ):
        if not _union_ok(q, k_cache, v_cache, logical_indices, block_table):
            return orig(
                q, k_cache, v_cache, logical_indices, block_table, token_to_req, out
            )
        if out is None:
            out = torch.empty_like(q)
        torch.ops.strix.qsa_union_attn_bf16(
            out, q, k_cache, v_cache, logical_indices, block_table[0]
        )
        return out

    ops.qsa_sparse_paged_attention = qsa_sparse_paged_attention
