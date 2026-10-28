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
