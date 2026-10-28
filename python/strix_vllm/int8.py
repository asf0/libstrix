# SPDX-License-Identifier: MIT
"""W8A8 int8 linears (compressed-tensors int-quantized) on gfx1151.

vLLM's ROCm fallback is `TritonInt8ScaledMMLinearKernel`: per-channel int8
weights and dynamic per-token int8 activations. Checkpoints store the weights
as signed int8 [N, K]; `strix::w8a8_wmma_gemm` reads the same rows as
excess-128 bytes (it XORs 0x80 at the LDS commit), so the only repack is a
byte flip, done in place - no second copy of a 30+ GiB expert set.

The op quantizes activations to int8 per 128-element group inside the kernel,
which is finer than the reference's one scale per token. The weights and their
per-channel scales are bit-identical to the checkpoint's.
"""

import torch

from strix_vllm import disabled

_registered = False


def _apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    out = torch.ops.strix.w8a8_wmma_gemm(w_q, x_2d.contiguous(), w_s)
    return out if bias is None else out + bias


def _apply_fake(x_2d, w_q, w_s, bias):
    return x_2d.new_empty((x_2d.size(0), w_q.size(0)))


def _kernel_class():
    from vllm.model_executor.kernels.linear.scaled_mm.triton import (
        TritonInt8ScaledMMLinearKernel,
    )

    class StrixW8A8LinearKernel(TritonInt8ScaledMMLinearKernel):
        """vLLM's int8 scaled-MM contract, with the GEMM on iu8 WMMA."""

        @classmethod
        def can_implement(cls, c) -> tuple[bool, str | None]:
            if c.is_static_input_scheme:
                return False, "strix W8A8 takes dynamic per-token activations"
            if not c.is_channelwise:
                return False, "strix W8A8 takes per-channel weight scales"
            if not c.input_symmetric:
                return False, "strix W8A8 takes symmetric activations"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            w = layer.weight
            if w.dtype != torch.uint8:
                # Signed int8 -> excess-128 bytes, in place (shared storage).
                layer.weight = torch.nn.Parameter(
                    w.data.view(torch.uint8).bitwise_xor_(0x80), requires_grad=False
                )
                w = layer.weight
            s = layer.weight_scale
            if s.dim() > 1:
                s = s.reshape(-1)
            layer.weight_scale = torch.nn.Parameter(
                s.data.float().contiguous(), requires_grad=False
            )

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            w_q, w_s = layer.weight, layer.weight_scale
            x_2d = x.reshape(-1, x.shape[-1])
            out = torch.ops.vllm.strix_w8a8_apply(
                x_2d, w_q, w_s, bias if bias is None else bias.contiguous()
            )
            return out.reshape(*x.shape[:-1], w_q.size(0))

    StrixW8A8LinearKernel.__name__ = "StrixW8A8LinearKernel"
    return StrixW8A8LinearKernel


def register() -> None:
    """Put the kernel at the front of ROCm's int8 candidates (idempotent)."""
    global _registered
    if _registered or disabled("w8a8"):
        return
    _registered = True
    from vllm.model_executor.kernels.linear import _POSSIBLE_INT8_KERNELS
    from vllm.platforms import PlatformEnum
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="strix_w8a8_apply",
        op_func=_apply_impl,
        mutates_args=[],
        fake_impl=_apply_fake,
    )
    _POSSIBLE_INT8_KERNELS.setdefault(PlatformEnum.ROCM, []).insert(0, _kernel_class())
