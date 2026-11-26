# SPDX-License-Identifier: MIT
"""inkling_rel_attn_decode (Inkling 2 relative attention for decode and
verify batches: split-KV, sheared relative logits, gated sink-scaled
epilogue) against a dense fp32 reference, bf16 and int8 KV."""
import pytest
import strix
import torch

from test_attn_prefill import build, quantize
from test_inkling_rel_attn import padded_quantize

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

HQ, HKV, D = 24, 8, 128
# e_pad of the served model (rel_extent 1024 + the table's 256 padding).
E_PAD = 1280


def reference(q, kc, vc, cu, seq_lens, table, rel, gate, sinks, scale, window_left):
    """Each query token sees its own window of the request's keys (causal
    from its position), the keys from key0 on add their sheared relative
    logit (key0: the e_pad keys ending at the request's last 128-key
    boundary), then the gate and the per-token sink scale the softmax
    output: out = o * sigmoid(gate) * sigmoid(lse - sink)."""
    e_pad = rel.size(2)
    bs = kc.size(1)
    out = torch.zeros(q.size(0), HQ * D, device="cuda")
    for n in range(cu.numel() - 1):
        a, b, kv_len = int(cu[n]), int(cu[n + 1]), int(seq_lens[n])
        q_len = b - a
        ctx = kv_len - q_len
        key0 = (kv_len + 127) // 128 * 128 - e_pad
        pos = torch.arange(kv_len, device="cuda")
        blocks = table[n, pos // bs].long()
        k = kc[blocks, pos % bs].float().repeat_interleave(HQ // HKV, 1)
        v = vc[blocks, pos % bs].float().repeat_interleave(HQ // HKV, 1)
        i = torch.arange(q_len, device="cuda")
        qpos = ctx + i
        dist = qpos[:, None] - pos[None, :]
        seen = dist >= 0
        if window_left >= 0:
            seen &= dist <= window_left
        s = torch.einsum("thd,khd->htk", q[a:b].float(), k) * scale
        col = pos[None, :] - key0
        has = seen & (col >= 0)
        bias = rel[a:b].float().gather(
            2, col.clamp(0, e_pad - 1)[:, None, :].expand(q_len, HQ, -1)
        )
        s = s + torch.where(has[:, None, :], bias, 0.0).permute(1, 0, 2)
        s = s.masked_fill(~seen[None], float("-inf"))
        lse = torch.logsumexp(s, -1)  # [H, q_len]
        o = torch.einsum("htk,khd->thd", s.softmax(-1), v)
        g = torch.sigmoid(gate[a:b].float().view(q_len, HQ, D))
        sk = torch.sigmoid(lse.t() - sinks[a:b].float())
        out[a:b] = (o * g * sk[..., None]).reshape(q_len, HQ * D)
    return out


@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [
        ([1], [0]),
        ([1], [77]),
        ([3], [300]),
        ([1, 3, 1], [900, 20, 5000]),
        ([2, 1, 3, 1], [0, 1024, 77, 12000]),
    ],
)
@pytest.mark.parametrize("window_left", [-1, 1023])
@pytest.mark.parametrize("num_splits", [1, 3, 9])
def test_matches_reference(q_lens, ctx_lens, window_left, num_splits):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, HQ, HKV, 16, hd=D)
    t = q.size(0)
    g = torch.Generator(device="cuda").manual_seed(5)
    rel = torch.randn(t + 128, HQ, E_PAD, device="cuda", generator=g).bfloat16()
    proj = torch.randn(t, HQ * D + HQ + 8, device="cuda", generator=g).bfloat16()
    gate, sinks = proj[:, : HQ * D], proj[:, HQ * D : HQ * D + HQ]
    cu, seq_lens = cu.int(), seq_lens.int()
    out = torch.full((t, HQ * D), float("nan"), device="cuda").bfloat16()
    torch.ops.strix.inkling_rel_attn_decode(
        out, q, kc, vc, None, None, rel, gate, sinks, cu, seq_lens, table,
        D**-0.5, num_splits, window_left,
    )
    ref = reference(q, kc, vc, cu, seq_lens, table, rel, gate, sinks, D**-0.5, window_left)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [([1], [0]), ([3], [300]), ([1, 3, 1], [900, 20, 5000])],
)
@pytest.mark.parametrize("window_left", [-1, 1023])
@pytest.mark.parametrize("num_splits", [1, 5])
def test_int8_kv_matches_dequantized_reference(q_lens, ctx_lens, window_left, num_splits):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, HQ, HKV, 16, hd=D)
    t = q.size(0)
    g = torch.Generator(device="cuda").manual_seed(6)
    rel = torch.randn(t + 128, HQ, E_PAD, device="cuda", generator=g).bfloat16()
    proj = torch.randn(t, HQ * D + HQ + 8, device="cuda", generator=g).bfloat16()
    gate, sinks = proj[:, : HQ * D], proj[:, HQ * D : HQ * D + HQ]
    cu, seq_lens = cu.int(), seq_lens.int()
    kq, ks = quantize(kc, 16, HKV)
    vq, vs = quantize(vc, 16, HKV)
    out = torch.full((t, HQ * D), float("nan"), device="cuda").bfloat16()
    torch.ops.strix.inkling_rel_attn_decode(
        out, q, kq, vq, ks, vs, rel, gate, sinks, cu, seq_lens, table,
        D**-0.5, num_splits, window_left,
    )
    ref = reference(
        q,
        (kq.float() * ks[..., None]).bfloat16(),
        (vq.float() * vs[..., None]).bfloat16(),
        cu, seq_lens, table, rel, gate, sinks, D**-0.5, window_left,
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


# The served packed layout: the scales sit inline in the cache, so the fetch
# reads 4 bytes at a time.
@pytest.mark.parametrize("q_lens,ctx_lens", [([3], [300]), ([1, 3, 1], [900, 20, 5000])])
@pytest.mark.parametrize("window_left", [-1, 1023])
@pytest.mark.parametrize("num_splits", [1, 5])
def test_int8_kv_packed_layout_matches_reference(q_lens, ctx_lens, window_left, num_splits):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, HQ, HKV, 16, hd=D)
    t = q.size(0)
    g = torch.Generator(device="cuda").manual_seed(8)
    rel = torch.randn(t + 128, HQ, E_PAD, device="cuda", generator=g).bfloat16()
    proj = torch.randn(t, HQ * D + HQ + 8, device="cuda", generator=g).bfloat16()
    gate, sinks = proj[:, : HQ * D], proj[:, HQ * D : HQ * D + HQ]
    cu, seq_lens = cu.int(), seq_lens.int()
    kq, vq, ks, vs = padded_quantize(kc, vc)
    out = torch.full((t, HQ * D), float("nan"), device="cuda").bfloat16()
    torch.ops.strix.inkling_rel_attn_decode(
        out, q, kq, vq, ks, vs, rel, gate, sinks, cu, seq_lens, table,
        D**-0.5, num_splits, window_left,
    )
    ref = reference(
        q,
        (kq.float() * ks[..., None]).bfloat16(),
        (vq.float() * vs[..., None]).bfloat16(),
        cu, seq_lens, table, rel, gate, sinks, D**-0.5, window_left,
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


SCALE = 0.25  # logits with a standard deviation of 2.8: a few keys matter


def build_verify(q_lens, ctx_lens, page, int8, pad=0, page_pad=1):
    """The arguments of a batch whose last `pad` rows belong to no request,
    on the served cache layouts, and the fp32 K/V a reference reads. The row
    scales of K and V (the int8 scales) spread 10x. With `page_pad` 2 the pages are twice as long, the second half NaN (bf16) or
    NaN scales (int8): a padded page's stride."""
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, HQ, HKV, page, hd=D)
    g = torch.Generator(device="cuda").manual_seed(9)
    t = q.size(0) + pad
    q = torch.cat([q, torch.randn(pad, HQ, D, device="cuda", generator=g).bfloat16()])
    rel = 2 * torch.randn(t + 128, HQ, E_PAD, device="cuda", generator=g).bfloat16()
    proj = torch.randn(t, HQ * D + HQ + 8, device="cuda", generator=g).bfloat16()
    # Sinks near the log-sum-exp of the logits, so that their gate matters.
    sinks = 3 * proj[:, HQ * D : HQ * D + HQ] + 8
    gate = proj[:, : HQ * D]
    kc, vc = (
        (x * 10 ** (torch.rand(x.shape[:3], device="cuda", generator=g)[..., None] - 0.5))
        .bfloat16()
        .repeat_interleave(page_pad, 0)
        for x in (kc, vc)
    )
    if int8:
        k, v, ks, vs = padded_quantize(kc, vc)
        poison = (ks, vs)
    else:
        packed = torch.cat([kc, vc], -1).transpose(1, 2).contiguous()
        k, v = packed.transpose(1, 2).split(D, -1)
        ks = vs = None
        poison = (k, v)
    for x in poison if page_pad > 1 else ():
        x[1::2] = float("nan")
    k, v, ks, vs = (None if x is None else x[::page_pad] for x in (k, v, ks, vs))
    kf, vf = (k.float(), v.float()) if ks is None else (
        k.float() * ks[..., None], v.float() * vs[..., None])
    args = (q, k, v, ks, vs, rel, gate, sinks, cu.int(), seq_lens.int(), table)
    return args, (kf, vf)


def decode(args, num_splits, window_left, q_rows):
    out = torch.full((args[0].size(0), HQ * D), float("nan"), device="cuda").bfloat16()
    torch.ops.strix.inkling_rel_attn_decode(
        out, *args, SCALE, num_splits, window_left, q_rows
    )
    return out


def verify_reference(args, kv, window_left):
    q, _, _, _, _, rel, gate, sinks, cu, seq_lens, table = args
    return reference(q, *kv, cu, seq_lens, table, rel, gate, sinks, SCALE, window_left)


def head_error(out, ref):
    """The largest relative error of a (token, head): NaN or infinite when a
    row of no request (zero in `ref`) is not zero."""
    err = (out.float() - ref).view(-1, HQ, D).norm(dim=-1)
    return (err / (ref.view(-1, HQ, D).norm(dim=-1) + 1e-20)).max().item()


# Verify batches with 1, 3 or 5 tokens of a request per block: single
# requests whose tokens cross a 128-token boundary (the sheared table's
# key0) or whose windows end in the context, mixed requests (one without
# tokens), rows of no request. The splits leave some segments without keys.
@pytest.mark.parametrize("q_rows", [1, 3, 5])
@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize(
    "q_lens,ctx_lens,pad",
    [([n], [c], 0) for n in (1, 2, 3, 4, 5, 8, 16) for c in (0, 126, 1500)]
    + [
        ([3], [127], 5),
        ([1, 3, 1], [900, 20, 5000], 1),
        ([4, 1, 5, 2, 16], [0, 127, 1030, 77, 250], 0),
        ([3, 3, 3, 3], [124, 125, 126, 1151], 4),
        ([2, 0, 3], [50, 10, 60], 5),
    ],
)
def test_verify_rows_match_reference(q_rows, int8, q_lens, ctx_lens, pad):
    args, kv = build_verify(q_lens, ctx_lens, 64, int8, pad)
    for window_left in (-1, 31, 1023):
        ref = verify_reference(args, kv, window_left)
        for num_splits in (1,) if q_rows == 1 else (1, 5, 20):  # 1 does not split
            out = decode(args, num_splits, window_left, q_rows)
            assert head_error(out, ref) < 1e-2, (window_left, num_splits)


@pytest.mark.parametrize("q_rows", [1, 3, 5])
@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("page", [32, 128])
def test_verify_rows_on_padded_pages(q_rows, int8, page):
    args, kv = build_verify([3, 1, 5], [300, 1100, 126], page, int8, 2, page_pad=2)
    for window_left in (-1, 1023):
        ref = verify_reference(args, kv, window_left)
        for num_splits in (1,) if q_rows == 1 else (1, 7):
            out = decode(args, num_splits, window_left, q_rows)
            assert head_error(out, ref) < 1e-2, (window_left, num_splits)
