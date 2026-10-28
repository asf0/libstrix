# SPDX-License-Identifier: MIT
"""The kernel table: coverage of the registered ops, geometry points, ranges."""

import re
from pathlib import Path

import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

# Ops the plugin implements without a torch.ops.strix kernel: a weight layout
# transform, and a Python-side wrapper around hipBLASLt.
_PLUGIN_ONLY = {"moe_pad", "row_sliced_gemm"}


def test_every_registered_op_is_in_the_table():
    # Read the C++ registrations rather than the dispatcher: a new m.def that
    # never reaches the table is exactly the drift this guards against.
    source = (Path(__file__).parents[1] / "csrc" / "torch_ops.cpp").read_text()
    registered = set(re.findall(r'm\.def\(\s*"(\w+)\(', source))
    assert registered, "no m.def found in csrc/torch_ops.cpp"
    described = {k["name"] for k in strix.kernels()} | _PLUGIN_ONLY
    # _wmma_bf16_tile is a layout probe for tests/test_wmma_layout.py.
    assert registered - described == {"_wmma_bf16_tile"}, (
        f"registered but not in the table: {registered - described}"
    )
    assert described - registered == _PLUGIN_ONLY, (
        f"in the table but not registered: {described - registered - _PLUGIN_ONLY}"
    )


def test_table_has_no_duplicate_names_per_geometry():
    seen = set()
    for k in strix.kernels():
        # One entry per (name, geometry); the moe_gemm rows are the same
        # kernel twice, so compare the full constraint set too.
        key = (k["name"], k["op"], tuple(sorted(k["constraints"].items())))
        assert key not in seen, f"duplicate row for {k['name']}"
        seen.add(key)


def test_attention_geometry_comes_from_the_table():
    assert strix.geometries("attn_prefill", "head_dim", "gqa") == {
        (64, 4),
        (256, 2),
        (256, 6),
        (256, 8),
        (512, 8),
    }
    assert strix.geometries("attn_decode", "head_dim", "gqa") == {
        (256, 2),
        (256, 8),
        (512, 8),
    }
    # Every decode geometry takes bf16 and int8 KV.
    for kv in ("bf16", "int8"):
        assert strix.geometries("attn_decode", "head_dim", "gqa", kv_dtype=kv) == {
            (256, 2),
            (256, 8),
            (512, 8),
        }
    # Split-KV (speculative verify) is built for head 256 GQA 6 only.
    assert strix.geometries("attn_prefill", "head_dim", "gqa", split_kv=True) == {
        (256, 6)
    }


def test_select_prefers_the_first_matching_row():
    skinny = {
        "weight_bits": 8,
        "group_size": 128,
        "rows": 1,
        "dtype": torch.bfloat16,
    }
    assert strix.select("w8a16", **skinny) == "w8a16_skinny_gemm"
    assert strix.select("w8a16", **{**skinny, "rows": 512}) == "w8a8_wmma_gemm"
    # fp16 activations are covered by both W8 kernels, fp32 is not.
    assert strix.select("w8a16", **{**skinny, "dtype": torch.float32}) is None


def test_range_reason_names_the_bound():
    geometry = {
        "weight_bits": 4,
        "group_size": 32,
        "rows": 64,
        "dtype": torch.bfloat16,
        "scale_dtype": torch.bfloat16,
    }
    rows = {r["name"]: r for r in strix.explain("w4a16", **geometry)}
    assert rows["w4a16_wmma_gemm"]["ok"] is False
    assert "rows=64, needs >= 129" in rows["w4a16_wmma_gemm"]["reason"]
    # 32-element groups at small M stay on the W4A8 op.
    assert strix.select("w4a16", **geometry) == "w4a8_wmma_gemm"
    assert strix.select("w4a16", **{**geometry, "rows": 512}) == "w4a16_wmma_gemm"


def test_missing_geometry_is_reported():
    # 256 exists at GQA 2 and 8, so head_dim alone is ambiguous and the
    # 512-only row still reports its constraint.
    assert strix.select("attn_decode", head_dim=256) is None
    reasons = [r["reason"] for r in strix.explain("attn_decode", head_dim=256)]
    assert reasons == ["gqa not given", "head_dim=256, needs == 512", "gqa not given"]


def test_unknown_op_explains_nothing():
    assert strix.explain("no_such_op", head_dim=256) == []
    assert strix.select("no_such_op") is None
