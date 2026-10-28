# SPDX-License-Identifier: MIT
"""Native MXFP4 (e2m1 + e8m0) W4A16 linears on gfx1151.

Quark MXFP4 checkpoints carry `weight` u8 [N, K/2] (two e2m1 values per byte)
and `weight_scale` u8 [N, K/32] (e8m0). vLLM has no native ROCm kernel for
them - AiterMxfp4LinearKernel needs supports_mx(), false on gfx1151 - so
EmulationMxfp4LinearKernel runs, and by default it dequantizes every weight
matrix to bf16 on every forward. This kernel keeps the packed bytes and the
e8m0 scales exactly as loaded and runs strix::mxfp4_wmma_gemm, which decodes
the nibbles into LDS as it stages them, into a plain bf16 WMMA main loop.

The activation QDQ of the emulation is deliberately skipped (W4A16): the
checkpoint's activation format is only dynamic fp4, so keeping bf16
activations is strictly more accurate, and there is no per-forward pass over x.
Measured on gemma-4-31B-it-MXFP4 at 512 context: 611 ms/step decode for the
default emulation, 271 ms for dequant-at-load and 127.5 ms for the WMMA tile,
with 18.1 GiB of weights instead of 59.2 and teacher-forced top-1 agreement of
1.0000 against the dequant-at-load text.

M <= 2 takes `mxfp4_gemv` instead: warp-per-row register decoding measured
1.4-1.5x the WMMA tile at M=1 (177-201 vs 125-135 GB/s on the 31B shapes) and
still ahead at M=2; the tile wins from M=4.

Prefill M (>= DEQUANT_MIN_M rows) decodes the weight once with
`mxfp4_dequant` and runs hipBLASLt on the bf16 matrix: the fused tile re-decodes
every weight tile once per 128-row M block and stays at ~30 TF, while decode +
hipBLASLt reaches 37-44 TF on the 31B's qkv / o / gate_up shapes at M 2048-8192
(1.16-1.40x). The 21504-K down projection wins too up to M 2048 but keeps the
fused tile from DEQUANT_LONG_K_MAX_M up: hipBLASLt itself drops to 26 TF there
at M 8192. linear.py's W4A16 path uses the same rule (`use_dequant`); its
crossover measured the same.
"""

import torch

from strix_vllm import disabled

DEQUANT_MIN_M = 1024
DEQUANT_MAX_K = 8192
DEQUANT_LONG_K_MAX_M = 4096


def use_dequant(m: int, k: int) -> bool:
    """Decode the 4-bit weight once and run hipBLASLt, or keep the fused tile."""
    return m >= DEQUANT_MIN_M and (k <= DEQUANT_MAX_K or m < DEQUANT_LONG_K_MAX_M)


def _apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # Decode (M <= 2) is a pure weight stream; the GEMV reads each byte once
    # with register decoding and measured 1.4-1.5x the WMMA tile at M=1
    # (177-201 vs 125-135 GB/s). Past M=2 the WMMA tile wins, so keep it.
    if (
        x_2d.size(0) <= 2
        and bias is None
        and x_2d.is_contiguous()
        and x_2d.stride(0) % 8 == 0
        and w_q.stride(0) % 16 == 0
    ):
        return torch.ops.strix.mxfp4_gemv(x_2d, w_q, w_s)
    if use_dequant(x_2d.size(0), x_2d.size(1)):
        out = torch.nn.functional.linear(x_2d, torch.ops.strix.mxfp4_dequant(w_q, w_s))
        return out if bias is None else out + bias
    return torch.ops.strix.mxfp4_wmma_gemm(w_q, x_2d.contiguous(), w_s, bias)


def _apply_fake(x_2d, w_q, w_s, bias):
    return x_2d.new_empty((x_2d.size(0), w_q.size(0)))


def _model_dtype() -> torch.dtype | None:
    """The model's activation dtype, or None when no config is active yet."""
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return None if config is None else config.model_config.dtype


def register() -> None:
    if disabled("mxfp4"):
        return
    import vllm.model_executor.kernels.linear as linear_kernels
    from torch.nn.parameter import Parameter
    from vllm.model_executor.kernels.linear.mxfp4.base import (
        MxFp4LinearKernel,
        MxFp4LinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp4Dynamic,
    )
    from vllm.platforms import PlatformEnum
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="strix_mxfp4_apply",
        op_func=_apply_impl,
        mutates_args=[],
        fake_impl=_apply_fake,
    )

    class StrixMxfp4LinearKernel(MxFp4LinearKernel):
        @classmethod
        def is_supported(
            cls, compute_capability: int | None = None
        ) -> tuple[bool, str | None]:
            # The kernel is bf16 wave32 WMMA tuned for gfx1151; the op itself
            # has no arch guard, so this is the only gate.
            from vllm.platforms.rocm import on_gfx1151

            if not on_gfx1151():
                return False, "strix MXFP4 W4A16 GEMM is built for gfx1151"
            return True, None

        @classmethod
        def can_implement(
            cls, config: MxFp4LinearLayerConfig
        ) -> tuple[bool, str | None]:
            # None is weight-only; kMxfp4Dynamic is what Quark's MXFP4
            # checkpoints declare, and this kernel simply does not quantize
            # the activations (more accurate, and no extra pass over x).
            if config.activation_quant_key not in (None, kMxfp4Dynamic):
                return False, (
                    "strix MXFP4 is weight-only (bf16 activations), got "
                    f"activation_quant_key={config.activation_quant_key}"
                )
            dtype = _model_dtype()
            if dtype is not None and dtype != torch.bfloat16:
                return False, f"needs bf16 activations, model dtype is {dtype}"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # The packed weights and e8m0 scales are used as they are: the op
            # decodes them in place. The checkpoint's own Parameters (with the
            # loaders vLLM may still consult) are kept; only a plain tensor or
            # a padded view needs wrapping, which shares storage, and the
            # .contiguous() then is a load-time copy, not a per-forward one.
            for name in ("weight", "weight_scale"):
                param = getattr(layer, name)
                if not param.is_contiguous():
                    param = param.contiguous()
                if not isinstance(param, Parameter):
                    setattr(layer, name, Parameter(param, requires_grad=False))
            k = layer.weight.size(1) * 2
            if k % 128:
                raise ValueError(
                    f"strix MXFP4: K={k} must be a multiple of 128 "
                    f"(N={layer.weight.size(0)})"
                )

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            w_q, w_s = layer.weight, layer.weight_scale
            x_2d = x.reshape(-1, x.shape[-1])
            out = torch.ops.vllm.strix_mxfp4_apply(x_2d, w_q, w_s, bias)
            return out.reshape(*x.shape[:-1], w_q.size(0))

    # Selection takes the first kernel that reports supported.
    linear_kernels._POSSIBLE_MXFP4_KERNELS[PlatformEnum.ROCM].insert(
        0, StrixMxfp4LinearKernel
    )
