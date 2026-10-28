# SPDX-License-Identifier: MIT
"""Row-sliced bf16 GEMMs for long K on gfx1151.

hipBLASLt holds ~41 TFLOPS on long-K GEMMs (Qwen3.8-27B MLP down, K=17408)
up to ~1K rows and drops to ~30 past that; 1K-row slices keep the
activation panel near the 32 MB infinity cache. vLLM's linear layers fetch
`rocm_unquantized_gemm` by name when they are built, so the plugin rebinds it
to a wrapper that sends long-K layers to an opaque custom op (the slice loop
must not be traced by torch.compile) and every other shape to the original.
"""

import torch

from strix_vllm import disabled

MIN_K = 16384
ROWS = 1024


def _sliced_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    from vllm.logger import init_logger

    # Runs eagerly inside the custom op; the wrapper below is traced.
    init_logger(f"vllm.{__name__}").info_once(
        "libstrix: row-sliced GEMM for K=%d", weight.size(1)
    )
    gemm = torch.ops.vllm.rocm_unquantized_gemm
    x2 = x.reshape(-1, x.size(-1))
    if x2.size(0) <= ROWS:
        return gemm(x, weight, bias)
    out = x2.new_empty(x2.size(0), weight.size(0))
    for start in range(0, x2.size(0), ROWS):
        out[start : start + ROWS] = gemm(x2[start : start + ROWS], weight, bias)
    return out.view(*x.shape[:-1], weight.size(0))


def _sliced_fake(x, weight, bias):
    return x.new_empty(*x.shape[:-1], weight.size(0))


def register() -> None:
    if disabled("gemm_rows"):
        return
    import vllm.model_executor.layers.utils as layer_utils
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="strix_row_sliced_gemm",
        op_func=_sliced_impl,
        mutates_args=[],
        fake_impl=_sliced_fake,
    )
    orig = layer_utils.rocm_unquantized_gemm

    def rocm_unquantized_gemm(layer, x, weight, bias=None):
        if weight.dim() == 2 and weight.size(1) >= MIN_K and x.dtype == torch.bfloat16:
            return torch.ops.vllm.strix_row_sliced_gemm(x, weight, bias)
        return orig(layer, x, weight, bias)

    layer_utils.rocm_unquantized_gemm = rocm_unquantized_gemm
