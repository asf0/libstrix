# SPDX-License-Identifier: MIT
"""qsa_rows_attn_bf16: QSA sparse GQA (head_dim 256, GQA 12), one query row
per wave pair, against a dense torch reference of the same softmax; and the
plugin hook (fp8 / int8 caches converted per request) against vLLM's kernel."""

import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

G, D = 12, 256


def reference(q, k_cache, v_cache, indices, block_table, token_to_req):
    hkv = k_cache.shape[2]
    block = k_cache.shape[1]
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for r in range(q.shape[0]):
        sel = indices[r]
        sel = sel[sel >= 0]  # a repeated token counts once per occurrence, as in the kernels
        if sel.numel() == 0:
            continue
        bt = block_table[int(token_to_req[r])]
        slots = bt[sel // block].long() * block + sel % block
        k = k_cache.reshape(-1, hkv, D)[slots].float()
        v = v_cache.reshape(-1, hkv, D)[slots].float()
        qr = q[r].float().view(hkv, G, D)
        s = torch.einsum("hgd,nhd->hgn", qr, k) / D**0.5
        p = torch.softmax(s, dim=-1)
        out[r] = torch.einsum("hgn,nhd->hgd", p, v).reshape(hkv * G, D)
    return out


def setup(rows, ctx, width, reqs=1, hkv=2, block=64, seed=0):
    torch.manual_seed(seed)
    pages = (ctx + block - 1) // block
    k_cache = torch.randn(reqs * pages + 3, block, hkv, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    perm = torch.randperm(reqs * pages + 3, device="cuda")[: reqs * pages].int()
    block_table = perm.view(reqs, pages).contiguous()
    q = torch.randn(rows, hkv * G, D, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, ctx, (rows, width), device="cuda", dtype=torch.int32)
    # Neighbouring rows share tokens, padding is -1 (a row ends early, one
    # selects nothing), and a row may hold a token twice.
    indices[1::2, :20] = indices[0::2, :20][: indices[1::2].shape[0]]
    indices[:, -7:] = -1
    if rows > 3:
        indices[3] = -1
    token_to_req = (torch.arange(rows, device="cuda") * reqs // max(rows, 1)).int()
    return q, k_cache, v_cache, indices, block_table, token_to_req


@pytest.mark.parametrize(
    "rows,ctx,width,reqs,hkv",
    [
        (1, 300, 67, 1, 2),
        (37, 300, 67, 1, 2),
        (100, 2048, 2051, 1, 2),
        (64, 5000, 300, 3, 2),  # batched prefill, three requests
        (40, 262144, 300, 1, 2),  # positions past 131072
        (33, 4000, 130, 1, 1),  # one KV head per block
    ],
)
def test_matches_reference(rows, ctx, width, reqs, hkv):
    q, k_cache, v_cache, indices, block_table, t2r = setup(rows, ctx, width, reqs, hkv)
    out = torch.full_like(q, float("nan"))
    torch.ops.strix.qsa_rows_attn_bf16(
        out, q, k_cache, v_cache, indices, block_table, t2r if reqs > 1 else None
    )
    ref = reference(q, k_cache, v_cache, indices, block_table, t2r)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", ["fp8", "int8"])
@pytest.mark.parametrize("reqs,union_max", [(1, 49152), (1, 0), (2, 49152)])
def test_hook_converts_and_matches_vllm(dtype, reqs, union_max, monkeypatch):
    """The plugin's qsa_sparse_paged_attention on an fp8 / int8 cache against
    vLLM's kernel on the same cache: a short single request takes the union
    kernel (union_max 49152), a longer one (union_max 0) and a batch the
    per-row kernel on a bf16 copy of each request's pages."""
    import inspect

    ops = pytest.importorskip("vllm.models.qwen4_exp.amd.ops.qsa")
    from strix_vllm import qwen4_exp

    if dtype == "int8" and "k_scale" not in inspect.signature(
        ops.qsa_sparse_paged_attention
    ).parameters:
        pytest.skip("this vLLM has no int8 QSA cache")

    monkeypatch.setattr(qwen4_exp, "_MIN_ROWS", 16)
    monkeypatch.setattr(qwen4_exp, "_UNION_MAX_TOKENS", union_max)
    expect = "qsa_union_attn_bf16" if reqs == 1 and union_max else "qsa_rows_attn_bf16"
    rows, ctx = 48, 3000
    q, k16, v16, indices, block_table, t2r = setup(rows, ctx, 257, reqs)
    if dtype == "fp8":
        fp8 = qwen4_exp._FP8[0]
        k_cache, v_cache = (k16 * 0.25).to(fp8), (v16 * 0.25).to(fp8)
        scales = {}
    else:
        k_scale = (k16.float().abs().amax(-1) / 127).clamp_min(1e-6)
        v_scale = (v16.float().abs().amax(-1) / 127).clamp_min(1e-6)
        k_cache = (k16.float() / k_scale[..., None]).round().clamp(-128, 127).to(torch.int8)
        v_cache = (v16.float() / v_scale[..., None]).round().clamp(-128, 127).to(torch.int8)
        scales = {"k_scale": k_scale, "v_scale": v_scale}
    orig = ops.qsa_sparse_paged_attention
    want = orig(q, k_cache, v_cache, indices, block_table, t2r, None, **scales)
    calls = []
    strix_ops = torch.ops.strix

    class Ops:
        def __getattr__(self, name):
            real = getattr(strix_ops, name)
            if name.startswith("qsa_"):
                return lambda *a: (calls.append(name), real(*a))[1]
            return real

    monkeypatch.setattr(qwen4_exp.torch.ops, "strix", Ops(), raising=False)
    try:
        qwen4_exp._register_qsa_rows()
        got = ops.qsa_sparse_paged_attention(
            q, k_cache, v_cache, indices, block_table, t2r, None, **scales
        )
    finally:
        ops.qsa_sparse_paged_attention = orig
    assert calls == [expect], calls
    torch.testing.assert_close(got.float(), want.float(), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("rows,reqs", [(1, 1), (5, 1), (16, 2), (37, 3)])
def test_decode_fp8_matches_reference(rows, reqs):
    """qsa_decode_fp8 (decode / verify batches on the fp8 cache) against the
    reference on the same e4m3 values."""
    if not hasattr(torch.ops.strix, "qsa_decode_fp8"):
        pytest.skip("no qsa_decode_fp8")
    from vllm.platforms import current_platform

    fp8 = current_platform.fp8_dtype()
    q, k16, v16, indices, block_table, t2r = setup(rows, 4000, 2051, reqs, block=256)
    k = (k16.float() * 0.5).to(fp8)
    v = (v16.float() * 0.5).to(fp8)
    out = torch.full_like(q, float("nan"))
    torch.ops.strix.qsa_decode_fp8(
        out, q, k, v, indices, block_table, t2r if reqs > 1 else None
    )
    ref = reference(q, k.bfloat16(), v.bfloat16(), indices, block_table, t2r)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_hook_takes_decode_kernel_for_small_fp8_batches(monkeypatch):
    ops = pytest.importorskip("vllm.models.qwen4_exp.amd.ops.qsa")
    from strix_vllm import qwen4_exp

    rows = 5
    q, k16, v16, indices, block_table, t2r = setup(rows, 3000, 257, 1, block=256)
    fp8 = qwen4_exp._FP8[0]
    k, v = (k16 * 0.25).to(fp8), (v16 * 0.25).to(fp8)
    orig = ops.qsa_sparse_paged_attention
    want = orig(q, k, v, indices, block_table, t2r, None)
    calls = []
    strix_ops = torch.ops.strix

    class Ops:
        def __getattr__(self, name):
            real = getattr(strix_ops, name)
            if name.startswith("qsa_"):
                return lambda *a: (calls.append(name), real(*a))[1]
            return real

    monkeypatch.setattr(qwen4_exp.torch.ops, "strix", Ops(), raising=False)
    try:
        qwen4_exp._register_qsa_rows()
        got = ops.qsa_sparse_paged_attention(q, k, v, indices, block_table, t2r, None)
    finally:
        ops.qsa_sparse_paged_attention = orig
    assert calls == ["qsa_decode_fp8"], calls
    torch.testing.assert_close(got.float(), want.float(), rtol=3e-2, atol=3e-2)
