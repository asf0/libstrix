# SPDX-License-Identifier: MIT
"""W8A16 (compressed-tensors uint8b128) and W4A16 (uint4b8) linears on gfx1151.

Decode (M <= 5) runs strix::w8a16_skinny_gemm on the int8 weights; larger M
runs strix::w8a8_wmma_gemm (activations quantized per 128-element group
inside the op, native iu8 WMMA). Weights are stored once as uint8 [N, K]
(raw byte - 128 is the signed value) with row strides moved off the
1 KiB-multiple cliff. Moved from the fork's RDNAHybridW8A16LinearKernel.
"""

import weakref

import torch

from strix_vllm import disabled

MAX_SKINNY_BATCH_SIZE = 5
W4A16_MIN_M = 128
_W8A8_GROUP_SIZE = 128
# Row strides that are a multiple of this alias in the vector caches on
# gfx11; such rows get one extra cache line.
_WEIGHT_CLIFF_BYTES = 1024
_STRIDE_PAD_BYTES = 128
# fp32 W8A8 scale copies keyed by weight identity; covers every quantized
# linear of a model (Qwen3.8-27B: 400) so none is rebuilt per forward.
_W8A8_PREP_MAX_ENTRIES = 1024
_W8A8_PREP: dict[int, tuple[weakref.ref, torch.Tensor]] = {}


def _pack_w8_rows(unpacked: torch.Tensor) -> torch.Tensor:
    """[N, K] int32 holding 0..255 -> uint8 [N, K], padded row stride."""
    n, k = unpacked.shape
    w_u8 = (unpacked & 0xFF).to(torch.uint8)
    if k % _WEIGHT_CLIFF_BYTES:
        return w_u8.contiguous()
    buf = torch.empty((n, k + _STRIDE_PAD_BYTES), dtype=torch.uint8, device=w_u8.device)
    out = buf[:, :k]
    out.copy_(w_u8)
    return out


def _w8a8_scales(
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    group_size: int,
    k: int | None = None,
    groups: tuple[int, ...] = (_W8A8_GROUP_SIZE,),
):
    """fp32 scales for the W8A8 op ([N] per channel, [G, N] grouped), cached.
    The W4A8 op also takes 32-element groups (groups=(128, 32))."""
    n = w_q.size(0)
    k = w_q.size(1) if k is None else k
    if group_size == 0:
        if w_s.numel() != n:
            return None
    elif group_size not in groups or w_s.shape != (n, k // group_size):
        return None
    if k % _W8A8_GROUP_SIZE:
        return None
    entry = _W8A8_PREP.get(id(w_q))
    if entry is not None and entry[0]() is w_q:
        return entry[1]
    s = (
        w_s.reshape(-1).float().contiguous()
        if group_size == 0
        else w_s.float().t().contiguous()
    )
    if len(_W8A8_PREP) >= _W8A8_PREP_MAX_ENTRIES:
        _W8A8_PREP.pop(next(iter(_W8A8_PREP)))
    _W8A8_PREP[id(w_q)] = (weakref.ref(w_q), s)
    return s


def _apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    if x_2d.size(0) <= MAX_SKINNY_BATCH_SIZE:
        return torch.ops.strix.w8a16_skinny_gemm(
            w_q, x_2d, w_s, bias, cu_count, group_size
        )
    s = _w8a8_scales(w_q, w_s, group_size)
    if s is None:
        raise RuntimeError(
            f"strix W8A16: W8A8 path cannot run N={w_q.size(0)} K={w_q.size(1)} "
            f"group_size={group_size}"
        )
    out = torch.ops.strix.w8a8_wmma_gemm(w_q, x_2d.contiguous(), s)
    return out if bias is None else out + bias


def _apply_fake(x_2d, w_q, w_s, bias, cu_count, group_size):
    return x_2d.new_empty((x_2d.size(0), w_q.size(0)))


def _w4_apply_impl(
    x_2d: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    group_size: int,
) -> torch.Tensor:
    """Decode stays on vLLM's skinny int4 kernel; larger M runs the W4A8 WMMA
    op on the same packed buffer."""
    if x_2d.size(0) <= MAX_SKINNY_BATCH_SIZE:
        import vllm._custom_ops as ops

        return ops.wvSplitK_int4_g(w_q, x_2d, w_s, cu_count, group_size, None, bias)
    # 32-element groups at prefill sizes: bf16-dequant WMMA (no per-group
    # folds, bf16 activations) is 1.16-1.24x the W4A8 op from M ~512; W4A8
    # stays ahead for small M and for 128-element groups.
    if group_size == 32 and x_2d.size(0) > W4A16_MIN_M and w_s.dtype == torch.bfloat16:
        # Prefill M: decode the weight once and run hipBLASLt (1.10-1.41x the
        # fused tile from M 1024 on the gemma-31B shapes; same rule as MXFP4).
        from strix_vllm.mxfp4 import use_dequant

        if use_dequant(x_2d.size(0), x_2d.size(1)):
            out = torch.nn.functional.linear(
                x_2d, torch.ops.strix.w4a16_dequant(w_q, w_s, 32)
            )
            return out if bias is None else out + bias
        return torch.ops.strix.w4a16_wmma_gemm(x_2d.contiguous(), w_q, w_s, bias, 32)
    s = _w8a8_scales(
        w_q, w_s, 0 if group_size == -1 else group_size, 2 * w_q.size(1), (128, 32)
    )
    if s is None:
        raise RuntimeError(
            f"strix W4A16: W4A8 path cannot run N={w_q.size(0)} "
            f"K={2 * w_q.size(1)} group_size={group_size}"
        )
    out = torch.ops.strix.w4a8_wmma_gemm(w_q, x_2d.contiguous(), s)
    return out if bias is None else out + bias


def register() -> None:
    use_w8, use_w4 = not disabled("w8a16"), not disabled("w4a16")
    if not (use_w8 or use_w4):
        return
    import vllm.model_executor.kernels.linear as linear_kernels
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearKernel,
        MPLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        unpack_quantized_values_into_int32,
    )
    from vllm.model_executor.parameter import permute_param_layout_
    from vllm.platforms import PlatformEnum
    from vllm.scalar_type import scalar_types
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="strix_w8a16_apply",
        op_func=_apply_impl,
        mutates_args=[],
        fake_impl=_apply_fake,
    )

    class StrixW8A16LinearKernel(MPLinearKernel):
        @classmethod
        def get_min_capability(cls) -> int:
            return 0

        @classmethod
        def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
            if c.weight_type != scalar_types.uint8b128:
                return False, "strix W8A16 takes symmetric uint8b128 weights"
            if c.zero_points:
                return False, "zero points not supported"
            if c.act_type not in (torch.float16, torch.bfloat16):
                return False, "needs fp16/bf16 activations"
            if c.group_size not in (-1, 32, 128):
                return False, f"group_size {c.group_size} not in (-1, 32, 128)"
            k = c.partition_weight_shape[0]
            if k % 128:
                return False, f"K={k} must be a multiple of 128"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            c = self.config
            w_q_raw = getattr(layer, self.w_q_name)
            w_s_raw = getattr(layer, self.w_s_name)
            unpacked = unpack_quantized_values_into_int32(
                w_q_raw.data, c.weight_type, packed_dim=w_q_raw.packed_dim
            )
            if getattr(w_q_raw, "output_dim", 0) != 0:
                unpacked = unpacked.t().contiguous()
            w_q = _pack_w8_rows(unpacked)
            if w_s_raw.dim() == 2:
                permute_param_layout_(w_s_raw, input_dim=1, output_dim=0)
            w_s = w_s_raw.data.contiguous()
            self._transform_param(layer, self.w_q_name, lambda _: w_q)
            self._transform_param(layer, self.w_s_name, lambda _: w_s)
            from vllm.utils.platform_utils import num_compute_units

            # Read once here: apply_weights is traced by torch.compile.
            self._cu_count = num_compute_units()

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            w_q, w_s, _ = self._get_weight_params(layer)
            x_2d = x.reshape(-1, x.shape[-1])
            group_size = 0 if self.config.group_size == -1 else self.config.group_size
            out = torch.ops.vllm.strix_w8a16_apply(
                x_2d, w_q, w_s, bias, self._cu_count, group_size
            )
            return out.reshape(*x.shape[:-1], w_q.size(0))

    direct_register_custom_op(
        op_name="strix_w4a16_apply",
        op_func=_w4_apply_impl,
        mutates_args=[],
        fake_impl=_apply_fake,
    )

    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        RDNAHybridW4A16LinearKernel,
    )

    class StrixW4A16LinearKernel(RDNAHybridW4A16LinearKernel):
        """vLLM's RDNA int4 kernel (same packed weights, same decode path)
        with prefill on the W4A8 WMMA op instead of Triton."""

        @classmethod
        def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
            ok, why = super().can_implement(c)
            if not ok:
                return ok, why
            if c.weight_type != scalar_types.uint4b8 or c.zero_points:
                return False, "strix W4A16 takes symmetric uint4b8 weights"
            if c.group_size not in (-1, 32, 128):
                return False, f"group_size {c.group_size} not in (-1, 32, 128)"
            if c.partition_weight_shape[0] % 128:
                return False, "K must be a multiple of 128"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            super().process_weights_after_loading(layer)
            from vllm.utils.platform_utils import num_compute_units

            # Read once here: apply_weights is traced by torch.compile.
            self._cu_count = num_compute_units()

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            w_q, w_s, _ = self._get_weight_params(layer)
            x_2d = x.reshape(-1, x.shape[-1])
            out = torch.ops.vllm.strix_w4a16_apply(
                x_2d, w_q, w_s, bias, self._cu_count, self.config.group_size
            )
            return out.reshape(*x.shape[:-1], w_q.size(0))

    ours = [StrixW4A16LinearKernel] * use_w4 + [StrixW8A16LinearKernel] * use_w8
    for cls in ours:
        linear_kernels.register_linear_kernel(cls, PlatformEnum.ROCM, "mp")
    # register_linear_kernel appends; selection takes the first kernel that
    # can implement a layer, so move ours ahead of the in-tree ones.
    rocm = linear_kernels._POSSIBLE_KERNELS[PlatformEnum.ROCM]
    for cls in ours:
        rocm.insert(0, rocm.pop(rocm.index(cls)))
