# SPDX-License-Identifier: MIT
"""Block-scaled FP8 (float8_e4m3fn, 128x128 scales) W8A16 linears on gfx1151.

vLLM's ROCm path for these checkpoints (Qwen3.8-27B-FP8, `quant_method: fp8`)
is TritonFp8BlockScaledMMKernel, which quantizes the activations to fp8 and
runs a Triton block-scaled GEMM with no tuned config for this GPU: 0.6 tok/s
decode on the 27B. gfx1151 has no fp8 WMMA, so the fp8 weights are decoded
either way; this kernel keeps them as loaded (1 byte per weight) and
leaves the activations in bf16.

- M <= 5: strix::fp8_skinny_gemm. The weight is streamed once (~220 GB/s), the
  e4m3 bytes decoded in registers. The 128x128 block scales are expanded to one
  fp32 row per output channel at load ([N, K/128], 3 % of the weight) so the
  kernel indexes them like a per-row group scale.
- M < DEQUANT_MIN_M: strix::fp8_wmma_gemm_raw up to RAW_MAX_M rows (fp16
  activations, integer-only FP8 decode, the block scale folded into the fp32
  accumulator), strix::fp8_wmma_gemm above (decodes to bf16 as it stages). One
  weight pass up to M = 16 (the same time as the skinny kernel), then one per
  m-block of the tile.
- Larger M: strix::fp8_dequant decodes the weight to bf16 (exact) and
  hipBLASLt runs the GEMM, as for MXFP4: 1.1-1.3x the fused tile from M 768
  on the 27B shapes (bench/kernels/fp8_gemm.py).

Per-output-channel checkpoints (compressed-tensors "channel" weights with
dynamic per-token activations: Sage-A1.5, MiniCPM-V-4.7, gemma-4 FP8-dynamic)
take the same ops with an fp32 [N] scale, applied once per output; vLLM's
choice there is the FP8 _scaled_mm kernel list, so StrixFp8ChannelMMKernel
goes first in that one. K only needs to be a multiple of 64 on that path.
"""

import torch

from strix_vllm import disabled, fp16_activations_ok

MAX_SKINNY_ROWS = 5
DEQUANT_MIN_M = 768
# hipBLASLt drops from ~41 to ~29 TFLOPS on the 27B MLP down (K = 17408) past
# ~1K rows; 1K-row slices keep it up (python/strix_vllm/gemm.py, same rule).
# Longer-K only: at K = 5120/6144 slicing measured slower.
SLICE_MIN_K = 16384
SLICE_ROWS = 1024
# The mid-size tile kernel with fp16 activations and the scale folded into the
# accumulator (fp8_wmma_gemm_raw): 20-29 % faster than the bf16-staging one at
# 24-128 rows, level at 256 and ~6 % slower from 768 (bench/kernels/fp8_gemm.py).
# STRIX_DISABLE=fp8_raw keeps the bf16 one throughout, as do models whose
# activations leave the fp16 range (strix_vllm.FP16_ACT_UNSAFE, gemma-4).
RAW_MAX_M = 256
_RAW = [True]
_BLOCK = 128
# Row strides that are a multiple of this alias in the vector caches on gfx11
# (see linear.py): such rows get one extra cache line.
_WEIGHT_CLIFF_BYTES = 1024
_STRIDE_PAD_BYTES = 128
_ROW_SCALE_ATTR = "strix_fp8_row_scale"
# K granularity of the per-channel path (BK = 64 tiles; the scale is not
# blocked along K).
_CH_K = 64


def _apply_impl(
    x_2d: torch.Tensor,
    w: torch.Tensor,
    block_scale: torch.Tensor,
    row_scale: torch.Tensor,
    bias: torch.Tensor | None,
    cu_count: int,
    raw: bool,
) -> torch.Tensor:
    m = x_2d.size(0)
    if m <= MAX_SKINNY_ROWS:
        return torch.ops.strix.fp8_skinny_gemm(
            w, x_2d.contiguous(), row_scale, bias, cu_count
        )
    if m < DEQUANT_MIN_M:
        if raw and m <= RAW_MAX_M:
            out = torch.ops.strix.fp8_wmma_gemm_raw(
                w, x_2d.to(torch.float16).contiguous(), block_scale
            )
        else:
            out = torch.ops.strix.fp8_wmma_gemm(w, x_2d.contiguous(), block_scale)
    else:
        w_bf16 = torch.ops.strix.fp8_dequant(w, block_scale)
        if w.size(1) >= SLICE_MIN_K and m > SLICE_ROWS:
            out = x_2d.new_empty(m, w.size(0))
            for i in range(0, m, SLICE_ROWS):
                out[i : i + SLICE_ROWS] = torch.nn.functional.linear(
                    x_2d[i : i + SLICE_ROWS], w_bf16
                )
        else:
            out = torch.nn.functional.linear(x_2d, w_bf16)
    return out if bias is None else out + bias


def _apply_fake(x_2d, w, block_scale, row_scale, bias, cu_count, raw):
    return x_2d.new_empty((x_2d.size(0), w.size(0)))


def register() -> None:
    if disabled("fp8"):
        return
    _RAW[0] = not disabled("fp8_raw")
    import vllm.model_executor.kernels.linear as linear_kernels
    from torch.nn.parameter import Parameter
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        Fp8BlockScaledMMLinearKernel,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.utils import replace_parameter
    from vllm.platforms import PlatformEnum
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="strix_fp8_apply",
        op_func=_apply_impl,
        mutates_args=[],
        fake_impl=_apply_fake,
    )

    class StrixFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
        # W8A16: the activations stay bf16, so base apply_weights' dynamic
        # per-group activation quantize is skipped.
        apply_input_quant = False

        @classmethod
        def is_supported(cls, compute_capability=None):
            from vllm.platforms.rocm import on_gfx1151

            if not on_gfx1151():
                return False, "strix FP8 W8A16 is built for gfx1151"
            return True, None

        @classmethod
        def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
            ok, why = super().can_implement(config)
            if not ok:
                return ok, why
            if tuple(config.weight_quant_key.scale.group_shape) != (_BLOCK, _BLOCK):
                return False, "needs 128x128 weight blocks"
            if config.out_dtype != torch.bfloat16:
                return False, f"needs bf16 activations, got {config.out_dtype}"
            if config.weight_shape[-1] % _BLOCK:
                return False, f"K={config.weight_shape[-1]} must be a multiple of 128"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            super().process_weights_after_loading(layer)
            params = self._get_layer_params(layer)
            w, bs = params.weight, params.block_scale
            if w.dtype != torch.float8_e4m3fn:
                raise RuntimeError(f"strix FP8 takes float8_e4m3fn, got {w.dtype}")
            n, k = w.shape
            w = w.contiguous()
            if k % _WEIGHT_CLIFF_BYTES == 0:
                buf = torch.empty(
                    (n, k + _STRIDE_PAD_BYTES), dtype=w.dtype, device=w.device
                )
                buf[:, :k].copy_(w)
                w = buf[:, :k]
            replace_parameter(layer, params.WEIGHT, w)
            bs = bs.float().contiguous()
            replace_parameter(layer, params.block_scale_attr, bs)
            rows = bs.repeat_interleave(_BLOCK, dim=0)[:n].contiguous()
            setattr(layer, _ROW_SCALE_ATTR, Parameter(rows, requires_grad=False))
            from vllm.utils.platform_utils import num_compute_units

            # Read once here: apply_weights is traced by torch.compile.
            self._cu_count = num_compute_units()
            self._raw = _RAW[0] and fp16_activations_ok("fp8_raw")

        def apply_weights(self, layer, x, bias=None, **kwargs):
            params = self._get_layer_params(layer)
            w = params.weight
            x_2d = x.reshape(-1, x.shape[-1])
            out = torch.ops.vllm.strix_fp8_apply(
                x_2d,
                w,
                params.block_scale,
                getattr(layer, _ROW_SCALE_ATTR),
                bias,
                self._cu_count,
                self._raw,
            )
            return out.reshape(*x.shape[:-1], w.size(0))

        def apply_block_scaled_mm(self, A, B, As, Bs):
            raise NotImplementedError("strix FP8 overrides apply_weights")

    kernels = linear_kernels._POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.ROCM]
    if StrixFp8BlockScaledMMKernel not in kernels:
        kernels.insert(0, StrixFp8BlockScaledMMKernel)

    if disabled("fp8_channel"):
        return
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    class StrixFp8ChannelMMKernel(FP8ScaledMMLinearKernel):
        """W8A16 for per-channel FP8 weights: the per-token activation
        quantization the checkpoint asks for is skipped (bf16 in)."""

        @classmethod
        def is_supported(cls, compute_capability=None):
            from vllm.platforms.rocm import on_gfx1151

            if not on_gfx1151():
                return False, "strix FP8 W8A16 is built for gfx1151"
            return True, None

        @classmethod
        def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
            wkey = config.weight_quant_key
            if wkey.dtype != torch.float8_e4m3fn:
                return False, f"needs float8_e4m3fn weights, got {wkey.dtype}"
            if tuple(wkey.scale.group_shape) != tuple(GroupShape.PER_CHANNEL):
                return False, "needs per-channel weight scales"
            if config.out_dtype not in (None, torch.bfloat16):
                return False, f"needs bf16 activations, got {config.out_dtype}"
            if config.weight_shape[-1] % _CH_K:
                return False, f"K={config.weight_shape[-1]} must be a multiple of {_CH_K}"
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            w_name, s_name, _, _ = self.layer_param_names
            w, s = getattr(layer, w_name), getattr(layer, s_name)
            if w.dtype != torch.float8_e4m3fn:
                raise RuntimeError(f"strix FP8 takes float8_e4m3fn, got {w.dtype}")
            # The channel helper hands the weight over as a (K, N) view.
            if w.size(0) != self.config.weight_shape[0]:
                w = w.t()
            n, k = w.shape
            buf_k = k + _STRIDE_PAD_BYTES if k % _WEIGHT_CLIFF_BYTES == 0 else k
            buf = torch.empty((n, buf_k), dtype=w.dtype, device=w.device)
            buf[:, :k].copy_(w)
            replace_parameter(layer, w_name, buf[:, :k])
            s = s.float().reshape(-1).contiguous()
            if s.numel() != n:
                raise RuntimeError(f"per-channel scale has {s.numel()} values for N={n}")
            replace_parameter(layer, s_name, s)
            from vllm.utils.platform_utils import num_compute_units

            self._cu_count = num_compute_units()
            self._raw = _RAW[0] and fp16_activations_ok("fp8_raw")

        def apply_weights(self, layer, x, bias=None, **kwargs):
            w_name, s_name, _, _ = self.layer_param_names
            w, s = getattr(layer, w_name), getattr(layer, s_name)
            x_2d = x.reshape(-1, x.shape[-1])
            out = torch.ops.vllm.strix_fp8_apply(
                x_2d, w, s, s, bias, self._cu_count, self._raw
            )
            return out.reshape(*x.shape[:-1], w.size(0))

        def apply_scaled_mm(self, **kwargs):
            raise NotImplementedError("strix FP8 overrides apply_weights")

    kernels = linear_kernels._POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM]
    if StrixFp8ChannelMMKernel not in kernels:
        kernels.insert(0, StrixFp8ChannelMMKernel)
