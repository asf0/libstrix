# SPDX-License-Identifier: MIT
import pytest
import strix
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


def reference(x, z, w, eps, activation):
    x, z, w = x.float(), z.float(), w.float()
    act = torch.sigmoid if activation == "sigmoid" else F.silu
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w * act(z)
    return out.to(torch.bfloat16)


# 48 / 32 value heads: Qwen3.8-27B / Qwen3.6-35B-A3B GDN layers.
@pytest.mark.parametrize("tokens", [1, 3, 517])
@pytest.mark.parametrize("heads", [48, 32])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation", ["silu", "sigmoid"])
def test_matches_reference(tokens, heads, weight_dtype, activation):
    torch.manual_seed(0)
    x = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x) * 3
    w = (torch.rand(128, device="cuda") + 0.5).to(weight_dtype)
    out = strix.gdn_gated_rmsnorm_h128_bf16(x, z, w, 1e-6, activation)
    torch.testing.assert_close(out, reference(x, z, w, 1e-6, activation))


def test_strided_z_from_projection_split():
    """z arrives as a view into the fused qkvz projection, rows not contiguous."""
    torch.manual_seed(0)
    tokens, heads = 33, 48
    qkvz = torch.randn(tokens, heads * 128 * 3, device="cuda", dtype=torch.bfloat16)
    z = qkvz[:, -heads * 128 :].view(tokens, heads, 128)
    x = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
    out = strix.gdn_gated_rmsnorm_h128_bf16(x, z, w, 1e-6)
    assert out.is_contiguous()
    torch.testing.assert_close(out, reference(x, z, w, 1e-6, "silu"))


def test_rejects_wrong_geometry():
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="contiguous last dim of 128"):
        strix.gdn_gated_rmsnorm_h128_bf16(x, x, w, 1e-6)


def test_select():
    geometry = {
        "hidden": 128,
        "dtype": torch.bfloat16,
        "weight_dtype": torch.float32,
        "group_size": None,
        "norm_before_gate": True,
        "activation": "silu",
    }
    assert strix.select("rms_norm_gated", **geometry) == "gdn_gated_rmsnorm_h128_bf16"
    assert strix.select("rms_norm_gated", **{**geometry, "hidden": 256}) is None
    assert (
        strix.select("rms_norm_gated", **{**geometry, "norm_before_gate": False})
        is None
    )
