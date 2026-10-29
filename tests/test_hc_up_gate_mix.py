# SPDX-License-Identifier: MIT
"""hc_up_gate_mix_bf16: hyper-connection up projection fused with the gated
mix, y = mean_s sigmoid(bf16(a @ w_s.T)) * x_s over 4 streams, against the
unfused torch math (bf16 gate, as the separate GEMM writes it)."""

import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


@pytest.mark.parametrize("rows", [1, 37, 300])
@pytest.mark.parametrize("cols,k", [(2560, 320), (96, 64)])
def test_matches_reference(rows, cols, k):
    torch.manual_seed(0)
    x = torch.randn(rows, 4 * cols, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4 * cols, k, device="cuda", dtype=torch.bfloat16) * 0.1
    y = torch.full((rows, cols), float("nan"), device="cuda", dtype=torch.bfloat16)
    torch.ops.strix.hc_up_gate_mix_bf16(y, x, a, w)
    gate = (a.float() @ w.float().t()).to(torch.bfloat16).float()
    ref = (torch.sigmoid(gate) * x.float()).view(rows, 4, cols).mean(dim=1)
    torch.testing.assert_close(y.float(), ref, rtol=1e-2, atol=1e-2)
