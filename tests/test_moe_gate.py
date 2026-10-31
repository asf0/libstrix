# SPDX-License-Identifier: MIT
"""The MoE plugin's performance gate and tile table. No GPU or vLLM needed.

python/strix_vllm/moe.py imports only torch and strix_vllm.disabled at module
scope; the vllm imports live inside its registration functions.
"""

import re
from pathlib import Path

from strix_vllm import moe


def test_unknown_geometry_is_never_gated():
    # A geometry with no measurement keeps the strix kernel at any batch size.
    assert moe.strix_moe_ok(1, 1234, 5678)
    assert moe.strix_moe_ok(8192, 1234, 5678)


def test_measured_floor_bounds_the_rows():
    assert moe._MIN_ROWS, "expected at least one measured floor"
    for (n, k), floor in moe._MIN_ROWS.items():
        assert floor > 0, f"({n}, {k}) should have no entry rather than floor 0"
        assert not moe.strix_moe_ok(floor - 1, n, k), f"({n}, {k}) below floor"
        assert moe.strix_moe_ok(floor, n, k), f"({n}, {k}) at floor"


def test_gemma_geometry_is_gated():
    # gemma-4-26B-A4B: w13 N=1408 K=2816, w2 N=2816 K=704. A refactor must not
    # silently drop the one geometry the gate exists for.
    assert (1408, 2816) in moe._MIN_ROWS
    assert (2816, 704) in moe._MIN_ROWS


def test_strix_blocks_are_instantiated():
    source = (
        Path(__file__).parents[1] / "csrc" / "moe" / "moe_gemm_bf16.hip"
    ).read_text()
    instantiated = set(re.findall(r"STRIX_MOE\((\d+), (\d+),", source))
    assert instantiated, "no STRIX_MOE instantiations found"
    for tokens in (1, 2, 16, 1024, 1025, 8192):
        bm, bn = moe.strix_blocks(tokens)
        assert (str(bm), str(bn)) in instantiated, (
            f"strix_blocks({tokens}) = {bm}x{bn} is not instantiated"
        )


def test_fp8_raw_blocks_are_instantiated_and_divide_n():
    source = (
        Path(__file__).parents[1] / "csrc" / "moe" / "moe_gemm_w8a16.hip"
    ).read_text()
    instantiated = set(re.findall(r"STRIX_MOE_W8\((\d+), (\d+),", source))
    assert instantiated, "no STRIX_MOE_W8 instantiations found"
    # (w13 n, w13 k, w2 n) per keyed model, plus an unlisted shape (the default).
    shapes = [(1408, 2816, 2816), (3072, 2048, 2048), (1024, 2048, 2048)]
    for n, k, n2 in shapes:
        for tokens in (1, 4, 8, 9, 64, 65, 256, 257, 1024, 1025, 2048, 2049, 8192):
            bm, bn = moe.fp8_raw_blocks(tokens, n, k)
            assert (str(bm), str(bn)) in instantiated, f"{bm}x{bn} not instantiated"
            assert n % bn == 0 and n2 % bn == 0, f"block_n {bn} does not divide N"


def test_fp8_raw_blocks_default_unchanged():
    # The Qwen3.6-35B-A3B-FP8 table (no keyed entry) and the keyed gemma rows.
    assert moe.fp8_raw_blocks(1, 1024, 2048) == (16, 64)
    assert moe.fp8_raw_blocks(512, 1024, 2048) == (64, 64)
    assert moe.fp8_raw_blocks(4096, 1024, 2048) == (64, 128)
    assert moe.fp8_raw_blocks(4096, 1408, 2816) == (128, 128)
    assert moe.fp8_raw_blocks(1000, 1408, 2816) == (64, 128)


def test_fp16_activation_gate(monkeypatch):
    # gemma-4's activations leave the fp16 range: it must not take the raw FP8
    # kernels (fp16 activations); other architectures keep them.
    import strix_vllm

    monkeypatch.setattr(
        strix_vllm, "_model_architectures", lambda: ["Gemma4ForConditionalGeneration"]
    )
    assert not strix_vllm.fp16_activations_ok("moe_fp8_raw")
    monkeypatch.setattr(
        strix_vllm, "_model_architectures", lambda: ["Qwen3_5MoeForConditionalGeneration"]
    )
    assert strix_vllm.fp16_activations_ok("moe_fp8_raw")
    monkeypatch.setattr(strix_vllm, "_model_architectures", lambda: [])
    assert strix_vllm.fp16_activations_ok("fp8_raw")
