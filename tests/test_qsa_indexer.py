# SPDX-License-Identifier: MIT
"""qsa_indexer_bf16: QSA indexer scores (4 heads, head_dim 128, one compressed
KV head, paged) for one request, against the torch reference."""

import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


@pytest.mark.parametrize("rows,ctx", [(37, 500), (200, 4096), (130, 9000)])
def test_matches_reference(rows, ctx):
    torch.manual_seed(0)
    ratio, block = 4, 64
    n_keys = ctx // ratio
    pages = (n_keys + block - 1) // block + 2
    k_cache = torch.randn(pages + 3, block, 1, 128, device="cuda", dtype=torch.bfloat16)
    block_table = torch.randperm(pages + 3, device="cuda")[:pages].int().contiguous()
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(ctx - rows, ctx, device="cuda", dtype=torch.int64)
    seq = torch.tensor([ctx], device="cuda", dtype=torch.int32)
    columns = pages * block
    logits = torch.full((rows, columns), float("nan"), device="cuda")
    visible = torch.full((rows,), -7, device="cuda", dtype=torch.int32)
    torch.ops.strix.qsa_indexer_bf16(logits, visible, q, k_cache, block_table, pos, seq, ratio, 128**0.5)

    vis_ref = torch.minimum((pos + 1) // ratio, torch.tensor(ctx // ratio, device="cuda"))
    assert (visible.long() == vis_ref).all()
    cols = torch.arange(columns, device="cuda")
    slots = block_table[cols // block].long() * block + cols % block
    k = k_cache.view(-1, 128)[slots].float()  # [columns, 128]
    s = torch.relu(torch.einsum("rhd,cd->rhc", q.float(), k)).sum(1) / 128**0.5
    ref = torch.where(cols[None, :] < vis_ref[:, None], s, float("-inf"))
    assert (torch.isinf(logits) == torch.isinf(ref)).all()
    fin = torch.isfinite(ref)
    torch.testing.assert_close(logits[fin], ref[fin], rtol=1e-3, atol=1e-3)
