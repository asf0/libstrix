# SPDX-License-Identifier: MIT
"""The fragment layout every WMMA kernel relies on (csrc/strix_common.h)."""

import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


@pytest.mark.parametrize("k", [16, 64, 512])
def test_tile_matches_matmul(k):
    torch.manual_seed(0)
    a = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    d = torch.ops.strix._wmma_bf16_tile(a, b)
    torch.testing.assert_close(d, a.float() @ b.float().t(), rtol=1e-5, atol=1e-4)


def test_layout_is_not_transposed():
    a = torch.zeros(16, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    a[3, 5] = 1
    b[11, 5] = 2
    d = torch.ops.strix._wmma_bf16_tile(a, b)
    assert d[3, 11] == 2 and d.count_nonzero() == 1
