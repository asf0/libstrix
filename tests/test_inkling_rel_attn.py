# SPDX-License-Identifier: MIT
"""inkling_rel_attn_prefill (Inkling 2 relative attention with the gated,
sink-scaled epilogue) against a dense fp32 reference, bf16 and int8
(vLLM's int8_per_token_head) KV."""
import pytest
import strix
import torch

from test_attn_prefill import build, quantize

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

HQ, HKV, D = 24, 8, 128


def padded_quantize(kc, vc):
    """kc / vc [blocks, bs, Hkv, D] bf16 -> the served packed layout: a
    (blocks, Hkv, bs, 2 * (D + 4)) int8 tensor with one fp32 scale after
    each (head, slot) content half, returned as the key / value byte views (V's
    rows D + 4 bytes after K's) and the fp32 scale views."""
    b, bs, h, d = kc.shape

    def q8(t):
        s = t.abs().amax(-1).float().clamp_min(1e-8) / 127.0
        q = (t.float() / s[..., None]).round().clamp(-127, 127).to(torch.int8)
        return q, s

    kq, ks = q8(kc)
    vq, vs = q8(vc)
    packed = torch.zeros(b, h, bs, 2 * (d + 4), dtype=torch.int8, device=kc.device)
    packed[:, :, :, :d] = kq.permute(0, 2, 1, 3)
    packed[:, :, :, d : d + 4] = (
        ks.permute(0, 2, 1).contiguous().view(torch.int8).view(b, h, bs, 4)
    )
    packed[:, :, :, d + 4 : 2 * d + 4] = vq.permute(0, 2, 1, 3)
    packed[:, :, :, 2 * d + 4 :] = (
        vs.permute(0, 2, 1).contiguous().view(torch.int8).view(b, h, bs, 4)
    )
    t = packed.transpose(1, 2)
    key = t[:, :, :, :d]
    value = t[:, :, :, d + 4 : 2 * d + 4]
    base = packed.view(torch.float32)  # [b, h, bs, 2 * (d + 4) / 4]
    sc = (base.stride(0), base.stride(2), base.stride(1))  # block, slot, head
    k_scale = base.as_strided((b, bs, h), sc, d // 4)
    v_scale = base.as_strided((b, bs, h), sc, (d + 4 + d) // 4)
    return key, value, k_scale, v_scale


def _run(q, kc, vc, cu, seq_lens, table, rel_arg, relp, gate, sinks, window_left,
         q_tiles, block_n, scales):
    t = q.size(0)
    tile = 16 * q_tiles
    offsets = torch.zeros_like(cu)
    torch.cumsum((cu[1:] - cu[:-1] + tile - 1) // tile, 0, out=offsets[1:])
    max_tiles = (t + tile - 1) // tile + len(cu) - 1
    out = torch.full((t, HQ * D), float("nan"), device="cuda").bfloat16()
    torch.ops.strix.inkling_rel_attn_prefill(
        out, q, kc, vc, *scales, rel_arg, gate, sinks, cu, seq_lens, table,
        offsets, max_tiles, D**-0.5, window_left, q_tiles, block_n, relp,
    )
    return out


def reference(q, kc, vc, cu, seq_lens, table, rel, gate, sinks, scale, window_left):
    """Each query sees keys up to its own (at most window_left back); the keys
    from key0 on add their sheared relative logit (key0: the e_pad keys ending
    at the 128-key boundary past the query's 128-query tile)."""
    e_pad = rel.size(2)
    bs = kc.size(1)
    out = torch.zeros(q.size(0), HQ * D, device="cuda")
    for n in range(cu.numel() - 1):
        a, b, kv_len = int(cu[n]), int(cu[n + 1]), int(seq_lens[n])
        q_len, ctx = b - a, kv_len - (b - a)
        pos = torch.arange(kv_len, device="cuda")
        blocks = table[n, pos // bs].long()
        k = kc[blocks, pos % bs].float().repeat_interleave(HQ // HKV, 1)
        v = vc[blocks, pos % bs].float().repeat_interleave(HQ // HKV, 1)
        s = torch.einsum("thd,khd->htk", q[a:b].float(), k) * scale
        i = torch.arange(q_len, device="cuda")
        tile_end = torch.clamp((i // 128 + 1) * 128, max=q_len)
        key0 = (ctx + tile_end + 127) // 128 * 128 - e_pad  # [q_len]
        col = pos[None, :] - key0[:, None]  # [q_len, kv_len]
        has = col >= 0
        bias = rel[a:b].float().gather(2, col.clamp(0, e_pad - 1)[:, None, :].expand(-1, HQ, -1))
        s = s + torch.where(has[:, None, :], bias, 0.0).permute(1, 0, 2)
        qpos = ctx + i
        dist = qpos[:, None] - pos[None, :]
        seen = dist >= 0
        if window_left >= 0:
            seen &= dist <= window_left
        s = s.masked_fill(~seen[None], float("-inf"))
        lse = torch.logsumexp(s, -1)  # [H, q_len]
        o = torch.einsum("htk,khd->thd", s.softmax(-1), v)
        g = torch.sigmoid(gate[a:b].float().view(q_len, HQ, D))
        sk = torch.sigmoid(lse.t() - sinks[a:b].float())
        out[a:b] = (o * g * sk[..., None]).reshape(q_len, HQ * D)
    return out


@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [([64], [0]), ([1], [500]), ([300], [77]), ([100, 1, 17, 260], [0, 900, 300, 1400])],
)
@pytest.mark.parametrize("window_left", [-1, 1023])
@pytest.mark.parametrize("q_tiles,block_n", [(1, 16), (2, 32), (4, 32)])
def test_matches_reference(q_lens, ctx_lens, window_left, q_tiles, block_n):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, HQ, HKV, 64, hd=D)
    t = q.size(0)
    g = torch.Generator(device="cuda").manual_seed(1)
    rel = torch.randn(t + 128, HQ, 1280, device="cuda", generator=g).bfloat16()
    # gate and sinks as row slices of one projection, as in the model
    proj = torch.randn(t, HQ * D + HQ + 8, device="cuda", generator=g).bfloat16()
    gate, sinks = proj[:, : HQ * D], proj[:, HQ * D : HQ * D + HQ]
    cu, seq_lens = cu.int(), seq_lens.int()
    tile = 16 * q_tiles
    offsets = torch.zeros_like(cu)
    torch.cumsum((cu[1:] - cu[:-1] + tile - 1) // tile, 0, out=offsets[1:])
    max_tiles = (t + tile - 1) // tile + len(q_lens)
    out = torch.full((t, HQ * D), float("nan"), device="cuda").bfloat16()
    scale = D**-0.5
    torch.ops.strix.inkling_rel_attn_prefill(
        out, q, kc, vc, None, None, rel, gate, sinks, cu, seq_lens, table,
        offsets, max_tiles, scale, window_left, q_tiles, block_n,
    )
    ref = reference(q, kc, vc, cu, seq_lens, table, rel, gate, sinks, scale, window_left)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def reference_fused(q, kc, vc, cu, seq_lens, table, r, relp, gate, sinks, scale, window_left):
    """As `reference`, with key j of the query at p getting r . relp[:, p - j]
    for p - j < extent."""
    extent = relp.size(1)
    bs = kc.size(1)
    out = torch.zeros(q.size(0), HQ * D, device="cuda")
    for n in range(cu.numel() - 1):
        a, b, kv_len = int(cu[n]), int(cu[n + 1]), int(seq_lens[n])
        q_len, ctx = b - a, kv_len - (b - a)
        pos = torch.arange(kv_len, device="cuda")
        blocks = table[n, pos // bs].long()
        k = kc[blocks, pos % bs].float().repeat_interleave(HQ // HKV, 1)
        v = vc[blocks, pos % bs].float().repeat_interleave(HQ // HKV, 1)
        s = torch.einsum("thd,khd->htk", q[a:b].float(), k) * scale
        qpos = ctx + torch.arange(q_len, device="cuda")
        dist = qpos[:, None] - pos[None, :]
        logits = torch.einsum("thc,ce->the", r[a:b].float(), relp.float())  # [q, H, extent]
        bias = logits.gather(2, dist.clamp(0, extent - 1)[:, None, :].expand(-1, HQ, -1))
        s = s + torch.where(((dist >= 0) & (dist < extent))[:, None, :], bias, 0.0).permute(1, 0, 2)
        seen = dist >= 0
        if window_left >= 0:
            seen &= dist <= window_left
        s = s.masked_fill(~seen[None], float("-inf"))
        lse = torch.logsumexp(s, -1)
        o = torch.einsum("htk,khd->thd", s.softmax(-1), v)
        g = torch.sigmoid(gate[a:b].float().view(q_len, HQ, D))
        sk = torch.sigmoid(lse.t() - sinks[a:b].float())
        out[a:b] = (o * g * sk[..., None]).reshape(q_len, HQ * D)
    return out


@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [([64], [0]), ([1], [500]), ([300], [77]), ([100, 1, 17, 260], [0, 900, 300, 1400])],
)
@pytest.mark.parametrize("window_left", [-1, 1023])
@pytest.mark.parametrize("q_tiles,block_n", [(1, 16), (2, 32), (4, 32)])
def test_fused_logits_match_reference(q_lens, ctx_lens, window_left, q_tiles, block_n):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, HQ, HKV, 64, hd=D)
    t = q.size(0)
    g = torch.Generator(device="cuda").manual_seed(2)
    r = (torch.randn(t, HQ, 16, device="cuda", generator=g) * 0.5).bfloat16()
    relp = (torch.randn(16, 1024, device="cuda", generator=g) * 0.5).bfloat16()
    proj = torch.randn(t, HQ * D + HQ + 8, device="cuda", generator=g).bfloat16()
    gate, sinks = proj[:, : HQ * D], proj[:, HQ * D : HQ * D + HQ]
    cu, seq_lens = cu.int(), seq_lens.int()
    tile = 16 * q_tiles
    offsets = torch.zeros_like(cu)
    torch.cumsum((cu[1:] - cu[:-1] + tile - 1) // tile, 0, out=offsets[1:])
    max_tiles = (t + tile - 1) // tile + len(q_lens)
    out = torch.full((t, HQ * D), float("nan"), device="cuda").bfloat16()
    scale = D**-0.5
    torch.ops.strix.inkling_rel_attn_prefill(
        out, q, kc, vc, None, None, r, gate, sinks, cu, seq_lens, table, offsets,
        max_tiles, scale, window_left, q_tiles, block_n, relp.t().contiguous(),
    )
    ref = reference_fused(q, kc, vc, cu, seq_lens, table, r, relp, gate, sinks, scale, window_left)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


# int8 KV (vLLM's int8_per_token_head): the bytes widen in LDS without the
# scale, the K scale folds into the score and the V scale into P; the
# relative table, the gate and the sinks stay bf16. Pages of 32 keys, the
# kernel's step.
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [([64], [0]), ([300], [77]), ([100, 1, 17, 260], [0, 900, 300, 1400])],
)
@pytest.mark.parametrize("window_left", [-1, 1023])
@pytest.mark.parametrize("q_tiles,block_n", [(2, 32), (4, 32)])
def test_int8_kv_matches_dequantized_reference(
    q_lens, ctx_lens, window_left, q_tiles, block_n
):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, HQ, HKV, 32, hd=D)
    t = q.size(0)
    g = torch.Generator(device="cuda").manual_seed(3)
    rel = torch.randn(t + 128, HQ, 1280, device="cuda", generator=g).bfloat16()
    proj = torch.randn(t, HQ * D + HQ + 8, device="cuda", generator=g).bfloat16()
    gate, sinks = proj[:, : HQ * D], proj[:, HQ * D : HQ * D + HQ]
    cu, seq_lens = cu.int(), seq_lens.int()
    kq, ks = quantize(kc, 32, HKV)
    vq, vs = quantize(vc, 32, HKV)
    out = _run(q, kq, vq, cu, seq_lens, table, rel, None, gate, sinks,
               window_left, q_tiles, block_n, (ks, vs))
    ref = reference(
        q,
        (kq.float() * ks[..., None]).bfloat16(),
        (vq.float() * vs[..., None]).bfloat16(),
        cu, seq_lens, table, rel, gate, sinks, D**-0.5, window_left,
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


# The served packed layout: the scales sit inline in the cache, so the fetch
# reads 4 bytes at a time.
@pytest.mark.parametrize(
    "q_lens,ctx_lens", [([64], [0]), ([100, 1, 17, 260], [0, 900, 300, 1400])]
)
@pytest.mark.parametrize("window_left", [-1, 1023])
@pytest.mark.parametrize("q_tiles,block_n", [(2, 32), (4, 32)])
def test_int8_kv_packed_layout_matches_reference(
    q_lens, ctx_lens, window_left, q_tiles, block_n
):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, HQ, HKV, 32, hd=D)
    t = q.size(0)
    g = torch.Generator(device="cuda").manual_seed(7)
    rel = torch.randn(t + 128, HQ, 1280, device="cuda", generator=g).bfloat16()
    proj = torch.randn(t, HQ * D + HQ + 8, device="cuda", generator=g).bfloat16()
    gate, sinks = proj[:, : HQ * D], proj[:, HQ * D : HQ * D + HQ]
    cu, seq_lens = cu.int(), seq_lens.int()
    kq, vq, ks, vs = padded_quantize(kc, vc)
    out = _run(q, kq, vq, cu, seq_lens, table, rel, None, gate, sinks,
               window_left, q_tiles, block_n, (ks, vs))
    ref = reference(
        q,
        (kq.float() * ks[..., None]).bfloat16(),
        (vq.float() * vs[..., None]).bfloat16(),
        cu, seq_lens, table, rel, gate, sinks, D**-0.5, window_left,
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("window_left", [1023])
@pytest.mark.parametrize("q_tiles,block_n", [(2, 32)])
def test_int8_kv_fused_logits(q_tiles, block_n, window_left):
    q, kc, vc, cu, seq_lens, table = build([100, 1, 17, 260], [0, 900, 300, 1400],
                                           HQ, HKV, 32, hd=D)
    t = q.size(0)
    g = torch.Generator(device="cuda").manual_seed(4)
    r = (torch.randn(t, HQ, 16, device="cuda", generator=g) * 0.5).bfloat16()
    relp = (torch.randn(16, 1024, device="cuda", generator=g) * 0.5).bfloat16()
    proj = torch.randn(t, HQ * D + HQ + 8, device="cuda", generator=g).bfloat16()
    gate, sinks = proj[:, : HQ * D], proj[:, HQ * D : HQ * D + HQ]
    cu, seq_lens = cu.int(), seq_lens.int()
    kq, ks = quantize(kc, 32, HKV)
    vq, vs = quantize(vc, 32, HKV)
    out = _run(q, kq, vq, cu, seq_lens, table, r, relp.t().contiguous(), gate,
               sinks, window_left, q_tiles, block_n, (ks, vs))
    ref = reference_fused(
        q,
        (kq.float() * ks[..., None]).bfloat16(),
        (vq.float() * vs[..., None]).bfloat16(),
        cu, seq_lens, table, r, relp, gate, sinks, D**-0.5, window_left,
    )
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
