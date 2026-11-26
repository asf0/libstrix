# SPDX-License-Identifier: Apache-2.0
# Moved from the asf0/strix-vllm fork (dev, f1f5b6038a "quantization:
# dgpp_mixed"): vllm/model_executor/layers/quantization/dgpp_mixed.py.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`dgpp_mixed` checkpoints (e.g. HawkBearPig/flint-FP8-BF16).

The routed experts are weight-only block FP8 (E4M3 with 128x128
`weight_scale_inv`); every other tensor is the source BF16, unchanged. That
is vLLM's block-FP8 format restricted to the routed experts, so this config
is Fp8Config with every linear layer left unquantized.

Registered with vLLM's `register_quantization_config` (which also adds it to
the platform's supported list) before the plugin's STRIX_DISABLE gate: the
checkpoint cannot load without it, so stock (`STRIX_DISABLE=all`) needs it too.
A vLLM that defines `dgpp_mixed` itself keeps its own.
"""

from typing import Any


def register() -> None:
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        register_quantization_config,
    )

    if "dgpp_mixed" in QUANTIZATION_METHODS:
        return
    import torch
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    @register_quantization_config("dgpp_mixed")
    class DgppMixedConfig(Fp8Config):
        """Block-FP8 routed experts, BF16 everywhere else."""

        @classmethod
        def get_name(cls) -> str:
            return "dgpp_mixed"

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "DgppMixedConfig":
            recipe = config.get("recipe")
            if recipe != "fp8":
                raise ValueError(
                    f"dgpp_mixed recipe {recipe!r} is not supported; only 'fp8' "
                    "(block-FP8 routed experts, BF16 elsewhere) is"
                )
            if config.get("activations", "bfloat16") != "bfloat16":
                raise ValueError("dgpp_mixed: only bfloat16 activations are supported")
            return cls(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            )

        def get_quant_method(
            self, layer: torch.nn.Module, prefix: str
        ) -> QuantizeMethodBase | None:
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return super().get_quant_method(layer, prefix)
