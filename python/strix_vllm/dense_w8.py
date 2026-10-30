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

STRIX_DENSE_W4=1 (experimental, off): linears with K % 128 == 0 get a
symmetric int4 (RTN, group 128) copy instead, on vLLM's skinny int4 GEMV; the
LM head and other K stay int8. Plain RTN costs accuracy on Qwen3.8-Flash-Next:
GSM8K 100q 84 vs 92 (bf16) / 93 (int8 copies) - it would need calibrated
(AWQ/GPTQ) weights to be usable.

Measured on Qwen3.8-Flash-Next only, so it is on for that architecture alone
(strix_vllm.ARCH_GATED); STRIX_ENABLE=dense_w8 tries it on another model. The
copies stop at STRIX_DENSE_W8_MAX_GB (default 12, about 4.5 GB on QFN): on a
bf16 dense model they would otherwise add half the checkpoint (~27 GB on
Qwen3.8-27B), and the layers past the budget keep bf16 only.
"""

import os

import torch

from strix_vllm import disabled, model_gate
from strix_vllm.linear import MAX_SKINNY_BATCH_SIZE, _pack_w8_rows

_GROUP = 128
_MAX_BYTES = float(os.environ.get("STRIX_DENSE_W8_MAX_GB", "12")) * 2**30
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


def _quantize4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (vLLM skinny int4 uint4b8 pack, bf16 [N, K/128])."""
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        pack_skinny_int4,
    )

    n, k = w.shape
    wf = w.float().view(n, k // _GROUP, _GROUP)
    s = (wf.abs().amax(-1) / 7.0).clamp_min(1e-12)
    q = torch.round(wf / s[..., None]).clamp(-8, 7).view(n, k) + 8
    return pack_skinny_int4(q.to(torch.int32)), s.to(w.dtype).contiguous()


def _impl4(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
) -> torch.Tensor:
    if x_2d.size(0) <= MAX_SKINNY_BATCH_SIZE:
        import vllm._custom_ops as ops

        return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, _GROUP, None, bias)
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    return rocm_unquantized_gemm_impl(x_2d, weight, bias)


def _fake4(x_2d, weight, w_q, w_s, bias, cu_count):
    return x_2d.new_empty((x_2d.size(0), weight.size(0)))


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
    use_w4 = os.environ.get("STRIX_DENSE_W4", "0") == "1"
    if use_w4:
        direct_register_custom_op(
            op_name="strix_dense_w4_apply",
            op_func=_impl4,
            mutates_args=[],
            fake_impl=_fake4,
        )
    stats = {"layers": 0, "bytes": 0}
    # Decided at the first layer, while vLLM's config names the model.
    gate: dict[str, bool] = {}

    def convert(layer: torch.nn.Module, allow_w4: bool = True) -> None:
        if "on" not in gate:
            gate["on"] = model_gate("dense_w8")
        if not gate["on"]:
            return
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
        if stats["bytes"] + w.numel() > _MAX_BYTES:
            logger.warning_once(
                "libstrix: dense_w8 reached STRIX_DENSE_W8_MAX_GB=%.0f; "
                "further layers keep bf16 only",
                _MAX_BYTES / 2**30,
            )
            return
        cu = num_compute_units()
        x = torch.randn(1, w.size(1), device=w.device, dtype=w.dtype)
        want = x.float() @ w.data.float().t()
        if use_w4 and allow_w4 and w.size(1) % _GROUP == 0:
            import vllm._custom_ops as ops

            w4q, w4s = _quantize4(w.data)
            try:
                got = ops.wvSplitK_int4_g(w4q, x, w4s, cu, _GROUP, None, None).float()
                err = (got - want).norm() / want.norm().clamp_min(1e-6)
                if torch.isfinite(err) and err < 0.2:
                    layer._strix_w4 = (w4q, w4s, cu)
                    stats["layers"] += 1
                    stats["bytes"] += w4q.numel() * w4q.element_size()
                    return
            except RuntimeError:
                pass
        w_q, w_s, group = _quantize(w.data)
        try:
            got = torch.ops.strix.w8a16_skinny_gemm(w_q, x, w_s, None, cu, group).float()
        except RuntimeError:
            return
        err = (got - want).norm() / want.norm().clamp_min(1e-6)
        if not torch.isfinite(err) or err > 0.05:
            logger.warning_once("dense_w8: a layer's int8 copy disagrees; kept bf16")
            return
        layer._strix_w8 = (w_q, w_s, group, cu)
        _BY_WEIGHT[w.data_ptr()] = layer._strix_w8
        stats["layers"] += 1
        stats["bytes"] += w_q.numel()

    def apply_w8(layer, x, bias, orig_apply, method):
        w4 = getattr(layer, "_strix_w4", None)
        if w4 is not None:
            w_q, w_s, cu = w4
            x_2d = x.reshape(-1, x.shape[-1])
            out = torch.ops.vllm.strix_dense_w4_apply(x_2d, layer.weight, w_q, w_s, bias, cu)
            return out.reshape(*x.shape[:-1], layer.weight.size(0))
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
            convert(layer, allow_w4=False)

    def embedding_apply(self, layer, x, bias=None):
        return apply_w8(layer, x, bias, emb_apply, self)

    UnquantizedEmbeddingMethod.process_weights_after_loading = embedding_process
    UnquantizedEmbeddingMethod.apply = embedding_apply
