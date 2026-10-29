# SPDX-License-Identifier: MIT
"""Qwen3.8-Flash-Next (qwen4_exp) prefill swaps.

qsa_union: QSA sparse attention for a single-request prefill on
strix::qsa_union_attn_bf16.

Groups of 16 consecutive query rows attend over the union of their selected
tokens (csrc/attn/qsa_union_bf16.hip): 1.39x the RDNA-tuned Triton per-row
kernel on a 26k-token prompt's real selections, more as the context grows.
Everything else (decode, several requests, other shapes, positions past the
128k union bitmap) keeps vLLM's Triton kernel.

vLLM's AMD QSA module imports `qsa_sparse_paged_attention` from
`vllm.models.qwen4_exp.amd.ops.qsa` at call time, so swapping the module
attribute is enough.

hc_up_gate_mix: the hyper-connection up projection fused with the gated mix
(strix::hc_up_gate_mix_bf16) from 512 rows: 5.2 vs 6.7 ms at 16k rows and no
[M, 4H] gate in memory. vLLM's qwen4_exp_hc_up_gate_mix op calls
`hc_up_gate_mix_impl` of its ops module per call; smaller batches keep the
unfused path (vLLM's skinny GEMMs at decode).
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


_HC_MIN_ROWS = 512


def register() -> None:
    _register_qsa_union()
    _register_hc_up_gate_mix()


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

    def hc_up_gate_mix(x, lowrank, weight, hc_count):
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
