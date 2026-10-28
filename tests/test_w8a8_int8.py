# SPDX-License-Identifier: MIT
"""The plugin's int8 dense kernel for compressed-tensors W8A8 checkpoints.

Covers the three silent-failure surfaces: the gate (static input scales or
tensor-wise weight scales must be rejected), the in-place signed-int8 ->
excess-128 flip, and the per-channel scale squeeze, then the GEMM itself
against a torch reference that quantizes activations per 128-element group
the way strix::w8a8_wmma_gemm does.
"""

import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


def build_kernel():
    from strix_vllm import int8 as strix_int8

    strix_int8.register()
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearLayerConfig,
    )

    cfg = Int8ScaledMMLinearLayerConfig(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=True,
    )
    return strix_int8._kernel_class()(
        cfg,
        ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"],
    )


def mock_layer(n: int, k: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8),
        requires_grad=False,
    )
    layer.weight_scale = torch.nn.Parameter(
        (torch.rand((n, 1), device="cuda") * 0.01 + 1e-4).float(),
        requires_grad=False,
    )
    return layer


def reference(x: torch.Tensor, w_i8: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Per-channel int8 weights, per-128-group symmetric int8 activations."""
    m, k = x.shape
    w = w_i8.float() * s.reshape(-1, 1)
    xg = x.float().reshape(m, k // 128, 128)
    sc = xg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    xq = (xg / sc).round().clamp(-127, 127) * sc
    return xq.reshape(m, k) @ w.t()


def test_gate_rejects_unsupported_layouts():
    from strix_vllm import int8 as strix_int8
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        Int8ScaledMMLinearLayerConfig,
    )

    cls = strix_int8._kernel_class()

    def cfg(**kw):
        base = {
            "is_channelwise": True,
            "is_static_input_scheme": False,
            "input_symmetric": True,
        }
        return Int8ScaledMMLinearLayerConfig(**{**base, **kw})

    assert cls.can_implement(cfg())[0]
    assert not cls.can_implement(cfg(is_static_input_scheme=True))[0]
    assert not cls.can_implement(cfg(is_channelwise=False))[0]
    assert not cls.can_implement(cfg(input_symmetric=False))[0]


@pytest.mark.parametrize("n,k", [(2048, 2048), (1024, 2048), (256, 4096)])
@pytest.mark.parametrize("m", [6, 64, 300])
def test_kernel_matches_torch(n, k, m):
    kernel = build_kernel()
    layer = mock_layer(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_i8 = layer.weight.data.clone()
    s = layer.weight_scale.data.clone()

    kernel.process_weights_after_loading(layer)
    # The flip is in place: the storage is shared, only the dtype view changed.
    assert layer.weight.dtype == torch.uint8
    assert layer.weight_scale.shape == (n,)

    out = kernel.apply_weights(layer, x)
    ref = reference(x, w_i8, s)
    torch.testing.assert_close(
        out.float() / ref.abs().amax(), ref / ref.abs().amax(), rtol=0, atol=6e-2
    )


def test_bias_is_added():
    kernel = build_kernel()
    layer = mock_layer(2048, 2048)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    kernel.process_weights_after_loading(layer)
    with_bias = kernel.apply_weights(layer, x, bias)
    without = kernel.apply_weights(layer, x, None)
    torch.testing.assert_close(with_bias, without + bias)
