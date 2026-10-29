# SPDX-License-Identifier: MIT
"""qsa_union_attn_bf16: QSA sparse GQA (head_dim 256, GQA 12) over per-row
token selections, against a dense torch reference of the same softmax."""

import os

import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

HKV, G, D = 2, 12, 256


def reference(q, k_cache, v_cache, indices, block_table):
    rows = q.shape[0]
    block = k_cache.shape[1]
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for r in range(rows):
        sel = indices[r]
        sel = torch.unique(sel[sel >= 0])
        if sel.numel() == 0:
            continue
        slots = block_table[sel // block].long() * block + sel % block
        k = k_cache.reshape(-1, HKV, D)[slots].float()  # [n, HKV, D]
        v = v_cache.reshape(-1, HKV, D)[slots].float()
        qr = q[r].float().view(HKV, G, D)
        s = torch.einsum("hgd,nhd->hgn", qr, k) / D**0.5
        p = torch.softmax(s, dim=-1)
        out[r] = torch.einsum("hgn,nhd->hgd", p, v).reshape(HKV * G, D)
    return out


@pytest.mark.parametrize("union_rows", ["4", "8", "16"])
@pytest.mark.parametrize("rows,ctx", [(37, 300), (100, 2048)])
def test_matches_reference(rows, ctx, union_rows):
    # The row count is read once per process; run each in a fresh one.
    if os.environ.get("STRIX_QSA_UNION_ROWS", "16") != union_rows:
        pytest.skip("set STRIX_QSA_UNION_ROWS to run this group size")
    torch.manual_seed(0)
    block = 64
    pages = (ctx + block - 1) // block
    k_cache = torch.randn(pages + 3, block, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.randperm(pages + 3, device="cuda")[:pages].int().contiguous()
    q = torch.randn(rows, HKV * G, D, device="cuda", dtype=torch.bfloat16)
    width = 67
    indices = torch.randint(0, ctx, (rows, width), device="cuda", dtype=torch.int32)
    # Neighbouring rows share some tokens, a row may repeat one, padding is -1,
    # and one row selects nothing.
    indices[1::2, :20] = indices[0::2, :20][: indices[1::2].shape[0]]
    indices[:, 5] = indices[:, 4]
    indices[:, -7:] = -1
    indices[3] = -1
    out = torch.full_like(q, float("nan"))
    torch.ops.strix.qsa_union_attn_bf16(out, q, k_cache, v_cache, indices, block_table)
    ref = reference(q, k_cache, v_cache, indices, block_table)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
