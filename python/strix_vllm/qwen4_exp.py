# SPDX-License-Identifier: MIT
"""Qwen3.8-Flash-Next (qwen4_exp) prefill swaps.

qsa_union: QSA sparse attention for a single-request prefill on
strix::qsa_union_attn_bf16.

Groups of 16 consecutive query rows attend over the union of their selected
tokens (csrc/attn/qsa_union_bf16.hip): 1.51x the RDNA-tuned Triton per-row
kernel on a 26k-token prompt's real selections, more as the context grows.
An fp8 cache (--kv-cache-dtype=fp8) is cast to bf16 per request first:
vLLM's kernel on fp8 K/V is 4-7.6x slower. Everything else (decode, several
requests, other shapes, positions past the 128k union bitmap) keeps vLLM's
Triton kernel.

vLLM's AMD QSA module imports `qsa_sparse_paged_attention` from
`vllm.models.qwen4_exp.amd.ops.qsa` at call time, so swapping the module
attribute is enough.

qsa_indexer: the QSA indexer's paged scores for a single-request prefill on
strix::qsa_indexer_bf16 (1.7-1.9x vLLM's tl.dot kernel, 4k to 128k context);
vLLM's qsa_select_paged_tokens calls `qsa_mqa_paged` from its module globals.

hc_up_gate_mix: the hyper-connection up projection fused with the gated mix
(strix::hc_up_gate_mix_bf16) from 512 rows: 5.2 vs 6.7 ms at 16k rows and no
[M, 4H] gate in memory; decode batches use dense_w8's int8 copy of the up
weight, other batches the unfused path (vLLM's GEMM, then hc_gate_mix).
vLLM's GatedResidual computes `hc_gate_mix(xn, input_mix_weight_up(lowrank))`.
The plugin gives input_mix_weight_up a forward that returns `(lowrank,
weight)` instead of the product (the layer stays a ReplicatedLinear) and
rebinds the module's `hc_gate_mix`, which sends such a pair to one custom op,
vllm.strix_hc_up_gate_mix. A vLLM that still has the `hc_up_gate_mix_impl`
hook of the older strix-vllm fork gets the same body through that hook.

hc_combine_norm: the combine of a block's output into the residual streams
plus their RMSNorm, one program per row from 512 rows (Triton, moved from the
strix-vllm fork: 9.2 -> 5.2 ms at 16k rows).
"""

import math
import os

import torch

from strix_vllm import disabled

# STRIX_QSA_UNION_MIN_ROWS lowers the floor for accuracy runs on short prompts.
_MIN_ROWS = int(os.environ.get("STRIX_QSA_UNION_MIN_ROWS", "1024"))
_MAX_POSITIONS = 131072
# vLLM's fp8 QSA cache (--kv-cache-dtype=fp8), viewed in the platform's flavor.
_FP8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def _union_shape_ok(q, k_cache, v_cache, logical_indices, block_table) -> bool:
    """A single-request prefill the union kernel takes, positions aside."""
    return (
        q.shape[0] >= _MIN_ROWS
        and block_table.dim() == 2
        and block_table.shape[0] == 1
        and q.dtype == torch.bfloat16
        and k_cache.dtype == v_cache.dtype
        and k_cache.dtype in (torch.bfloat16, torch.int8, *_FP8)
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


def _fp8_pages(cache, block_table, tokens):
    """The request's first `tokens` positions of an fp8 cache as a contiguous
    bf16 cache [pages, block, Hkv, 256]. vLLM's fp8 QSA cache has unit scales
    (its layer refuses others), so e4m3 -> bf16 is an exact cast."""
    pages = block_table[0, : -(-tokens // cache.shape[1])].long()
    return cache[pages].to(torch.bfloat16)


_HC_MIN_ROWS = 512
_F16 = os.environ.get("STRIX_QSA_UNION_F16", "0") == "1"
_UNION_ROWS = int(os.environ.get("STRIX_QSA_UNION_ROWS", "16"))


_IDX_MIN_ROWS = 128


def register() -> None:
    _register_qsa_union()
    _register_qsa_indexer()
    _register_hc_up_gate_mix()
    _register_hc_combine_norm()


def _register_hc_combine_norm() -> None:
    """hc_combine_norm (residual + gated block output, then the per-stream
    RMSNorm) with one program per row from _HC_MIN_ROWS rows; decode keeps
    vLLM's per-stream grid. GatedResidual calls its module's `hc_combine_norm`."""
    if disabled("hc_combine_norm"):
        return
    try:
        from vllm.models.qwen4_exp.amd import hyperconnection as hcm
        from vllm.models.qwen4_exp.amd.ops import hc
    except ImportError:
        return
    orig = getattr(hcm, "hc_combine_norm", None)
    if orig is None or hasattr(torch.ops.vllm, "strix_hc_combine_norm"):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    from strix_vllm.qwen4_exp_triton import hc_combine_norm_rows

    def op(
        residual: torch.Tensor,
        block_output: torch.Tensor,
        injection_logits: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        hc_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, dim = residual.shape
        if not (
            n >= _HC_MIN_ROWS
            and dim % hc_count == 0
            and block_output.shape == (n, dim // hc_count)
            and injection_logits.shape == (n, hc_count)
            and residual.stride(1) == block_output.stride(1) == 1
            and injection_logits.stride(1) == 1
            and norm_weight.is_contiguous()
            and norm_weight.numel() in (dim // hc_count, dim)
        ):
            return hc._hc_combine_norm(
                residual, block_output, injection_logits, norm_weight, eps, hc_count
            )
        return hc_combine_norm_rows(
            residual, block_output, injection_logits, norm_weight, eps, hc_count
        )

    def op_fake(
        residual: torch.Tensor,
        block_output: torch.Tensor,
        injection_logits: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        hc_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return residual.new_empty(residual.shape), residual.new_empty(residual.shape)

    direct_register_custom_op(
        op_name="strix_hc_combine_norm", op_func=op, fake_impl=op_fake
    )

    def hc_combine_norm(residual, block_output, injection_logits, norm_weight, eps, hc_count):
        return torch.ops.vllm.strix_hc_combine_norm(
            residual, block_output, injection_logits, norm_weight, eps, hc_count
        )

    hcm.hc_combine_norm = hc_combine_norm


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
        **kwargs,
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
                **kwargs,
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


def _hc_up_gate_mix(x, lowrank, weight, hc_count, unfused):
    """hc_gate_mix(x, lowrank @ weight.T, hc_count) on the fastest path for
    the batch; `unfused(x, lowrank, weight, hc_count)` otherwise."""
    from strix_vllm import dense_w8
    from vllm.models.qwen4_exp.amd.ops import hc

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


def _hc_unfused(x, lowrank, weight, hc_count):
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
    from vllm.models.qwen4_exp.amd.ops import hc

    gate = dispatch_unquantized_gemm()(None, lowrank, weight)
    return hc._hc_gate_mix(x, gate, hc_count)


def _register_hc_up_gate_mix() -> None:
    if disabled("hc_up_gate_mix"):
        return
    try:
        from vllm.models.qwen4_exp.amd import hyperconnection as hcm
        from vllm.models.qwen4_exp.amd.ops import hc
    except ImportError:
        return
    _install_hc_hook(hcm, hc)


def _install_hc_hook(hcm, hc) -> None:
    """Hook GatedResidual of the hyperconnection module `hcm` (ops module `hc`)."""
    unfused = getattr(hc, "hc_up_gate_mix_impl", None)
    if unfused is not None:
        # Older strix-vllm fork: its op calls this attribute per call.
        def impl(x, lowrank, weight, hc_count):
            return _hc_up_gate_mix(x, lowrank, weight, hc_count, unfused)

        hc.hc_up_gate_mix_impl = impl
        return

    orig_gate_mix = getattr(hcm, "hc_gate_mix", None)
    gated = getattr(hcm, "GatedResidual", None)
    if orig_gate_mix is None or gated is None:
        return

    from vllm.model_executor.layers.linear import (
        ReplicatedLinear,
        UnquantizedLinearMethod,
    )
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "strix_hc_up_gate_mix"):

        def op(
            x: torch.Tensor, lowrank: torch.Tensor, weight: torch.Tensor, hc_count: int
        ) -> torch.Tensor:
            return _hc_up_gate_mix(x, lowrank, weight, hc_count, _hc_unfused)

        def op_fake(
            x: torch.Tensor, lowrank: torch.Tensor, weight: torch.Tensor, hc_count: int
        ) -> torch.Tensor:
            return x.new_empty((x.shape[0], x.shape[1] // hc_count))

        direct_register_custom_op(
            op_name="strix_hc_up_gate_mix", op_func=op, fake_impl=op_fake
        )

    class _UpWeight(ReplicatedLinear):
        """input_mix_weight_up whose forward hands its input and weight to
        hc_gate_mix instead of multiplying them."""

        def forward(self, x):
            return x, self.weight

    orig_init = gated.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        up = getattr(self, "input_mix_weight_up", None)
        if (
            type(up) is ReplicatedLinear
            and isinstance(up.quant_method, UnquantizedLinearMethod)
            and up.bias is None
            and not up.return_bias
        ):
            up.__class__ = _UpWeight

    def hc_gate_mix(x, gate, hc_count):
        if isinstance(gate, tuple):
            lowrank, weight = gate
            return torch.ops.vllm.strix_hc_up_gate_mix(x, lowrank, weight, hc_count)
        return orig_gate_mix(x, gate, hc_count)

    gated.__init__ = __init__
    hcm.hc_gate_mix = hc_gate_mix


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
        fp8 = k_cache.dtype in _FP8
        tokens = 0
        if int8 or fp8 or _F16 or capacity > _MAX_POSITIONS:
            tokens = int(logical_indices.max()) + 1
            if tokens <= 0:  # nothing selected: the Triton kernel writes zeros
                return orig(
                    q, k_cache, v_cache, logical_indices, block_table, token_to_req,
                    out, **scales,
                )
        if fp8:
            # vLLM's Triton kernel on fp8 K/V is 4-7.6x slower than the union
            # kernel on bf16 (16k rows: 1891 vs 250 ms per layer at 16k context,
            # 2068 vs 524 ms at 128k), so cast the request's pages to bf16
            # once per layer and take the bf16 union path. Past the bitmap the
            # bf16 copy goes to vLLM's kernel, which reads bf16 faster than fp8
            # (261k-token prefill 1,267 vs 1,221 tok/s with the fp8 cache
            # passed straight through).
            k16 = _fp8_pages(k_cache, block_table, tokens)
            v16 = _fp8_pages(v_cache, block_table, tokens)
            pages = torch.arange(k16.shape[0], device=q.device, dtype=torch.int32)
            if tokens > _MAX_POSITIONS:
                return orig(q, k16, v16, logical_indices, pages[None], token_to_req, out)
            if out is None:
                out = torch.empty_like(q)
            torch.ops.strix.qsa_union_attn_bf16(
                out, q, k16, v16, logical_indices, pages, tokens if _F16 else 0,
                None, None,
            )
            return out
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
