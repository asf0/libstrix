# SPDX-License-Identifier: MIT
import torch

from strix_vllm import disabled


def register() -> None:
    from vllm.config import get_cached_compilation_config
    from vllm.model_executor.layers.layernorm import RMSNormGated

    if disabled("rms_norm_gated"):
        return

    @RMSNormGated.register_oot
    class StrixRMSNormGated(RMSNormGated):
        # register_oot renames the class to "RMSNormGated"; keep honoring the
        # in-tree op name so -rms_norm_gated in custom_ops still disables it.
        @classmethod
        def enabled(cls) -> bool:
            ops = get_cached_compilation_config().custom_ops
            return "-rms_norm_gated" not in ops and "-RMSNormGated" not in ops

        def __init__(self, hidden_size: int, *args, **kwargs):
            super().__init__(hidden_size, *args, **kwargs)
            import strix

            self._strix_kernel = strix.select(
                "rms_norm_gated",
                hidden=hidden_size,
                dtype=torch.get_default_dtype(),
                weight_dtype=self.weight.dtype,
                group_size=self.group_size,
                norm_before_gate=self.norm_before_gate,
                activation=self.activation,
            )

        def forward_hip(
            self, x: torch.Tensor, z: torch.Tensor | None = None
        ) -> torch.Tensor:
            if self._strix_kernel is None or z is None or x.dtype != torch.bfloat16:
                return super().forward_hip(x, z)
            act = 1 if self.activation == "sigmoid" else 0
            return torch.ops.strix.gdn_gated_rmsnorm_h128_bf16(
                x, z, self.weight, self.eps, act
            )
