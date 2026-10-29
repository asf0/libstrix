# SPDX-License-Identifier: MIT
"""dense_w8: int8 decode copies of unquantized bf16 linear weights.

Decode on a bf16 model streams every dense weight once per step: on
Qwen3.8-Flash-Next (int4 experts, bf16 attention / GDN / shared-expert / HC
projections) that is ~9 GB and 67 % of a decode step, at the DRAM ceiling.
This keeps the bf16 weight for prefill (hipBLASLt) and adds a symmetric int8
copy (scale per 128-element group, or per channel when K is not a multiple
of 128) for batches of <= 5 rows, run on strix::w8a16_skinny_gemm: half the
bytes per step for ~+50 % of the dense weight memory.

Covers vLLM's UnquantizedLinearMethod layers and the ParallelLMHead (1.27 GB
at Qwen3.8's 248k vocabulary, 5.2 -> ~2.6 ms per step); `lookup(weight)`
hands the copy to ops that take a bare weight (the qwen4_exp HC mix).

Every layer's copy is checked once at load against the bf16 product on a
random row; a layer the kernel cannot take, or that disagrees, keeps bf16
only. STRIX_DISABLE=dense_w8 turns it off.
"""

import torch

from strix_vllm import disabled
from strix_vllm.linear import MAX_SKINNY_BATCH_SIZE, _pack_w8_rows

_GROUP = 128
# weight.data_ptr() -> (w_q, w_s, group, cu_count) for every converted layer.
_BY_WEIGHT: dict[int, tuple] = {}


def lookup(weight: torch.Tensor) -> tuple | None:
    """The int8 decode copy of a converted layer's bf16 weight, if any."""
    return _BY_WEIGHT.get(weight.data_ptr())


def _quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """bf16 [N, K] -> (uint8b128 [N, K] padded rows, bf16 scales, group)."""
    n, k = w.shape
    group = _GROUP if k % _GROUP == 0 else 0
    wf = w.float()
    if group:
        amax = wf.view(n, k // group, group).abs().amax(-1)
        s = (amax / 127.0).clamp_min(1e-12)
        q = torch.round(wf.view(n, k // group, group) / s[..., None]).view(n, k)
    else:
        amax = wf.abs().amax(-1)
        s = (amax / 127.0).clamp_min(1e-12)
        q = torch.round(wf / s[:, None])
    q = (q.clamp(-127, 127) + 128).to(torch.int32)
    return _pack_w8_rows(q), s.to(w.dtype).contiguous(), group


def _impl(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    group: int,
) -> torch.Tensor:
    if x_2d.size(0) <= MAX_SKINNY_BATCH_SIZE:
        return torch.ops.strix.w8a16_skinny_gemm(w_q, x_2d, w_s, bias, cu_count, group)
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    return rocm_unquantized_gemm_impl(x_2d, weight, bias)


def _fake(x_2d, weight, w_q, w_s, bias, cu_count, group):
    return x_2d.new_empty((x_2d.size(0), weight.size(0)))


def register() -> None:
    if disabled("dense_w8"):
        return
    from vllm.logger import init_logger
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )
    from vllm.utils.platform_utils import num_compute_units
    from vllm.utils.torch_utils import direct_register_custom_op

    logger = init_logger("vllm.strix")
    direct_register_custom_op(
        op_name="strix_dense_w8_apply",
        op_func=_impl,
        mutates_args=[],
        fake_impl=_fake,
    )
    stats = {"layers": 0, "bytes": 0}
    logger.info_once("libstrix: dense_w8 int8 decode copies of bf16 linears + LM head")

    def convert(layer: torch.nn.Module) -> None:
        w = getattr(layer, "weight", None)
        if (
            w is None
            or w.dtype != torch.bfloat16
            or w.dim() != 2
            or not w.is_cuda
            or w.size(1) % 16
            or w.size(0) < 16
        ):
            return
        w_q, w_s, group = _quantize(w.data)
        cu = num_compute_units()
        x = torch.randn(1, w.size(1), device=w.device, dtype=w.dtype)
        try:
            got = torch.ops.strix.w8a16_skinny_gemm(w_q, x, w_s, None, cu, group).float()
        except RuntimeError:
            return
        want = (x.float() @ w.data.float().t())
        err = (got - want).norm() / want.norm().clamp_min(1e-6)
        if not torch.isfinite(err) or err > 0.05:
            logger.warning_once("dense_w8: a layer's int8 copy disagrees; kept bf16")
            return
        layer._strix_w8 = (w_q, w_s, group, cu)
        _BY_WEIGHT[w.data_ptr()] = layer._strix_w8
        stats["layers"] += 1
        stats["bytes"] += w_q.numel()

    def apply_w8(layer, x, bias, orig_apply, method):
        w8 = getattr(layer, "_strix_w8", None)
        if w8 is None:
            return orig_apply(method, layer, x, bias)
        w_q, w_s, group, cu = w8
        x_2d = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.strix_dense_w8_apply(
            x_2d, layer.weight, w_q, w_s, bias, cu, group
        )
        return out.reshape(*x.shape[:-1], layer.weight.size(0))

    lin_process = UnquantizedLinearMethod.process_weights_after_loading
    lin_apply = UnquantizedLinearMethod.apply

    def linear_process(self, layer: torch.nn.Module) -> None:
        lin_process(self, layer)
        convert(layer)

    def linear_apply(self, layer, x, bias=None):
        return apply_w8(layer, x, bias, lin_apply, self)

    UnquantizedLinearMethod.process_weights_after_loading = linear_process
    UnquantizedLinearMethod.apply = linear_apply

    # The LM head only: input embeddings share the method but are gathered
    # by row, never multiplied.
    emb_process = UnquantizedEmbeddingMethod.process_weights_after_loading
    emb_apply = UnquantizedEmbeddingMethod.apply

    def embedding_process(self, layer: torch.nn.Module) -> None:
        emb_process(self, layer)
        if isinstance(layer, ParallelLMHead):
            convert(layer)

    def embedding_apply(self, layer, x, bias=None):
        return apply_w8(layer, x, bias, emb_apply, self)

    UnquantizedEmbeddingMethod.process_weights_after_loading = embedding_process
    UnquantizedEmbeddingMethod.apply = embedding_apply
