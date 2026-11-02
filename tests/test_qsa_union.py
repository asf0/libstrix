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


def packed_cache(blocks, block, int8):
    """vLLM's packed layout, logical [blocks, Hkv, block, content] seen as
    [blocks, block, Hkv, content]: each head's K is followed by its V (and,
    int8_per_token_head, an fp32 scale after each). Returns K / V views, the
    scale views (None for bf16) and the dequantized K / V for the reference."""
    torch.manual_seed(1)
    k = torch.randn(blocks, block, HKV, D, device="cuda")
    v = torch.randn(blocks, block, HKV, D, device="cuda")
    # Per-token magnitudes over 100x, so per-(token, head) scales matter.
    k = k * torch.logspace(-1, 1, block, device="cuda")[None, :, None, None]
    v = v * torch.logspace(1, -1, block, device="cuda")[None, :, None, None]
    if not int8:
        packed = torch.empty(blocks, HKV, block, 2 * D, device="cuda", dtype=torch.bfloat16)
        kv = packed.transpose(1, 2)
        kv[..., :D] = k.bfloat16()
        kv[..., D:] = v.bfloat16()
        kc, vc = kv[..., :D], kv[..., D:]
        return kc, vc, None, None, kc, vc
    packed = torch.zeros(blocks, HKV, block, 2 * D + 8, device="cuda", dtype=torch.int8)
    kv = packed.transpose(1, 2)
    f32 = torch.tensor([], dtype=torch.float32, device="cuda").set_(kv.untyped_storage())
    st = tuple(s // 4 for s in kv.stride()[:3])
    ks = torch.as_strided(f32, kv.shape[:3], st, D // 4)
    vs = torch.as_strided(f32, kv.shape[:3], st, (2 * D + 4) // 4)
    kc, vc = kv[..., :D], kv[..., D + 4 : 2 * D + 4]
    for x, xc, xs in ((k, kc, ks), (v, vc, vs)):
        scale = (x.bfloat16().float().abs().amax(-1) / 127).clamp_min(1e-6)
        xs.copy_(scale)
        xc.copy_((x.bfloat16().float() / scale[..., None]).round().clamp(-128, 127).to(torch.int8))
    kd = (kc.float() * ks[..., None]).bfloat16()
    vd = (vc.float() * vs[..., None]).bfloat16()
    return kc, vc, ks, vs, kd, vd


@pytest.mark.parametrize("int8", [False, True], ids=["bf16-compact", "int8"])
def test_compact_path_on_packed_cache(int8):
    """The compact fp16 path (STRIX_QSA_UNION_F16 for bf16; always for int8)
    on vLLM's packed layout, where a head's K sits 2 * 256 (+ scales) after
    the previous head's: against the reference on the same (dequantized) K/V."""
    if os.environ.get("STRIX_QSA_UNION_ROWS", "16") != "16":
        pytest.skip("the compact path runs 16-row groups")
    rows, ctx, block = 100, 2048, 64
    pages = ctx // block
    kc, vc, ks, vs, kd, vd = packed_cache(pages + 3, block, int8)
    block_table = torch.randperm(pages + 3, device="cuda")[:pages].int().contiguous()
    q = torch.randn(rows, HKV * G, D, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, ctx, (rows, 67), device="cuda", dtype=torch.int32)
    indices[:, -7:] = -1
    tokens = int(indices.max()) + 1
    out = torch.full_like(q, float("nan"))
    torch.ops.strix.qsa_union_attn_bf16(
        out, q, kc, vc, indices, block_table, tokens, ks, vs
    )
    ref = reference(q, kd.contiguous(), vd.contiguous(), indices, block_table)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
