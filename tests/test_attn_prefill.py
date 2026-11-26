# SPDX-License-Identifier: MIT
import itertools
import math
from types import SimpleNamespace

import pytest
import strix
import torch

DEV = "cuda" if strix.is_supported_device() else "cpu"


@pytest.fixture(autouse=True)
def needs_gfx1151(request):
    # The checks of the checker itself build no kernel input.
    if DEV == "cpu" and not request.node.name.startswith("test_checker"):
        pytest.skip("needs gfx1151")


def build(q_lens, ctx_lens, hq, hkv, block_size, seed=0, hd=256):
    """Random paged KV, heads before slots ([blocks, Hkv, bs, 2*hs], vLLM's
    LBHNC) viewed as [blocks, bs, Hkv, hs] K/V caches, with shuffled block
    tables."""
    torch.manual_seed(seed)
    seq_lens = [q + c for q, c in zip(q_lens, ctx_lens)]
    nblk = [(s + block_size - 1) // block_size for s in seq_lens]
    total = sum(nblk) + 3
    kv = torch.randn(total, hkv, block_size, 2 * hd, device=DEV).bfloat16()
    key_cache, value_cache = kv.transpose(1, 2).split(hd, dim=-1)
    perm = torch.randperm(total, device=DEV)
    table = torch.zeros(len(q_lens), max(nblk), dtype=torch.int32, device=DEV)
    used = 0
    for i, nb in enumerate(nblk):
        table[i, :nb] = perm[used : used + nb]
        used += nb
    t = sum(q_lens)
    q = torch.randn(t, hq, hd, device=DEV).bfloat16()
    cu = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device=DEV)
    return q, key_cache, value_cache, cu, torch.tensor(seq_lens, device=DEV), table


def reference(
    q,
    key_cache,
    value_cache,
    cu,
    seq_lens,
    table,
    scale,
    window_left=-1,
    sinks=None,
    drop=(),
    ahead=0,
    p_dtype=torch.float32,
):
    """Dense fp32 attention. `drop` ((sequence, key) pairs not attended) and
    `ahead` (keys past the query that are) make it wrong on purpose; `p_dtype`
    rounds the softmax weights."""
    hq, hkv = q.size(1), key_cache.size(2)
    bs = key_cache.size(1)
    out = torch.empty_like(q, dtype=torch.float32)
    for n in range(cu.numel() - 1):
        a, b, kv_len = int(cu[n]), int(cu[n + 1]), int(seq_lens[n])
        pos = torch.arange(kv_len, device=q.device)
        blocks = table[n, pos // bs].long()
        # [kv_len, Hkv, D]; freed pages hold NaN, which nothing may attend to.
        k = key_cache[blocks, pos % bs].float().nan_to_num(0.0)
        v = value_cache[blocks, pos % bs].float().nan_to_num(0.0)
        k = k.repeat_interleave(hq // hkv, 1)
        v = v.repeat_interleave(hq // hkv, 1)
        qs = q[a:b].float()
        s = torch.einsum("thd,khd->htk", qs, k) * scale
        qpos = kv_len - (b - a) + torch.arange(b - a, device=q.device)
        dist = qpos[:, None] - pos[None, :]
        hidden = dist < -ahead
        if window_left >= 0:
            hidden = hidden | (dist > window_left)
        for m, key in drop:
            if m == n:
                hidden[:, key] = True
        s.masked_fill_(hidden[None], float("-inf"))
        if sinks is not None:
            # One extra logit per head (unscaled), dropped after the softmax.
            sink = sinks.float()[:, None, None].expand(-1, b - a, 1)
            p = torch.cat([s, sink], -1).softmax(-1)[..., :-1]
        else:
            p = s.softmax(-1)
        out[a:b] = torch.einsum("htk,khd->thd", p.to(p_dtype).float(), v)
    return out


# Bounds on a (token, head) row's error over the RMS of that row of the
# reference. The kernels' rounding, emulated on the planted and dense cases
# below (bf16 output; in prefill also bf16 or fp16 softmax weights and fp16
# score partials), stays under 0.0052 RMS and 0.019 at any element (decode:
# 0.0022 RMS). One lost key out of n of equal weight costs 1 / sqrt(n).
ROW_RMS_TOL = 0.01
ROW_MAX_TOL = 0.04


def row_errors(out, ref):
    """(RMS error, largest error) of each (token, head) row over its RMS."""
    rms = ref.square().mean(-1).sqrt().clamp_min(1e-6)
    err = (out.float() - ref).abs()
    return err.square().mean(-1).sqrt() / rms, err.amax(-1) / rms


def mismatch(out, ref):
    """What is wrong with `out`, or None."""
    if not out.isfinite().all():
        return f"{int((~out.isfinite()).any(-1).sum())} rows not finite"
    rms, worst = row_errors(out, ref)
    if rms.max() <= ROW_RMS_TOL and worst.max() <= ROW_MAX_TOL:
        return None
    token, head = divmod(int(rms.argmax()), rms.size(1))
    return (
        f"row error {float(rms.max()):.4f} (max element {float(worst.max()):.4f}) at "
        f"token {token} head {head}, {int((rms > ROW_RMS_TOL).sum())} bad rows"
    )


def assert_matches(out, ref):
    problem = mismatch(out, ref)
    assert problem is None, problem


def landmarks(qpos, first, page):
    """Visible keys [tokens, 8] a kernel is most likely to lose: the first and
    last of the window, both sides of the first and the last page edge in it,
    and the first key of the last 16- and 32-key chunk."""
    inner = (first // page + 1) * page
    last_edge = qpos // page * page
    marks = [first, qpos, inner, inner - 1, last_edge, last_edge - 1]
    marks += [first + (qpos - first) // c * c for c in (16, 32)]
    marks = torch.stack(marks, -1)
    return torch.minimum(torch.maximum(marks, first[:, None]), qpos[:, None])


def served(
    q_lens,
    ctx_lens,
    hq,
    hkv,
    hd,
    page,
    int8=False,
    window=-1,
    planted=True,
    seed=0,
    v_gain=1.0,
):
    """A case in the layout vLLM serves (LBNHC): [block][slot][head][K | V], on
    int8 [K | K scale | V | V scale] with scales spread over 10x, every block
    padded by one slot. Everything no query may read is NaN (int8: the scales),
    and pages a sliding window has left are the null block 0.

    planted: each (token, head) attends to up to four keys, two of them
    landmarks, with logits about 6 above the log of the key count, so a lost or
    wrong key moves its row by far more than the tolerance. The keys just
    outside the window and just past the query get 12, from every fourth row.
    """
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=gen)

    def randn(*shape):
        return torch.randn(*shape, generator=gen)

    lens = [q + c for q, c in zip(q_lens, ctx_lens)]
    content = 2 * (hd + 4) if int8 else 2 * hd
    slot = hkv * content
    stride = (page + 1) * slot
    first_key = [max(0, c - window) if window >= 0 else 0 for c in ctx_lens]
    live = [range(f // page, -(-n // page)) for f, n in zip(first_key, lens)]
    total = 1 + sum(len(r) for r in live) + 2
    table = torch.zeros(len(lens), max(r.stop for r in live), dtype=torch.int32)
    perm = torch.randperm(total - 1, generator=gen).int() + 1
    for row, r in zip(table, live):
        row[r.start : r.stop], perm = perm[: len(r)], perm[len(r) :]
    nan = float("nan")
    if int8:
        raw = torch.full((total * stride // 4,), nan).view(torch.int8)
    else:
        raw = torch.full((total * stride,), nan, dtype=torch.bfloat16)

    def carve(raw):
        cache = raw.as_strided((total, hkv, page, content), (stride, content, slot, 1))
        kc, vc = cache.transpose(1, 2).split(content // 2, dim=-1)
        if not int8:
            return kc, vc, None, None
        shape = (total, page, hkv)
        strides = (stride // 4, slot // 4, content // 4)
        words = raw.view(torch.float32)
        ks = words.as_strided(shape, strides, hd // 4)
        vs = words.as_strided(shape, strides, (content // 2 + hd) // 4)
        return kc, vc, ks, vs

    kc, vc, ks, vs = carve(raw)
    queries, marks = [], []
    for n, (ql, kl) in enumerate(zip(q_lens, lens)):
        q = randn(ql, hq, hd).bfloat16().float()
        k, v = randn(kl, hkv, hd), randn(kl, hkv, hd) * v_gain
        qpos = kl - ql + torch.arange(ql)
        first = (qpos - window).clamp_min(0) if window >= 0 else torch.zeros_like(qpos)
        mark = landmarks(qpos, first, page)
        marks.append(mark)
        if planted:
            t, h = torch.meshgrid(torch.arange(ql), torch.arange(hq), indexing="ij")
            spots = [mark[t, (2 * h + j + t) % 8] for j in range(2)]
            span = (qpos - first + 1)[:, None]
            spots += [first[:, None] + (rand(ql, hq) * span).long() for _ in range(2)]
            logit = [6 + 1.4 * rand(ql, hq) - 0.7 for _ in spots]
            for s in (first[:, None] - 1, qpos[:, None] + 1):
                spots.append(torch.where((t + h) % 4 == 0, s, -1))
                logit.append(torch.full((ql, hq), 12.0))
            # At most three rows share a key, and a row counts a key once.
            rows = [torch.stack(x, -1).flatten() for x in ([t] * 6, [h] * 6, spots)]
            logit = torch.stack(logit, -1).flatten()
            order = torch.randperm(len(logit), generator=gen)
            rows, logit = [x[order] for x in rows], logit[order]
            used, keep = {}, []
            for i, (tok, head, s) in enumerate(zip(*(x.tolist() for x in rows))):
                at_key = used.setdefault((s, head // (hq // hkv)), [])
                if 0 <= s < kl and len(at_key) < 3 and (tok, head) not in at_key:
                    at_key.append((tok, head))
                    keep.append(i)
            tok, head, s = (x[keep] for x in rows)
            qs = q[tok, head]
            gain = (math.log(kl) + logit[keep]) * hd**0.5 / qs.square().sum(-1)
            where = s, head // (hq // hkv)
            k.index_put_(where, gain[:, None] * qs, accumulate=True)
        at = torch.arange(live[n].start * page, kl)
        where = table[n, at // page].long(), at % page
        for view, scales, x in ((kc, ks, k[at]), (vc, vs, v[at])):
            if int8:
                s = x.abs().amax(-1) / 127 * 10 ** rand(*x.shape[:2])
                x = (x / s[..., None]).round().to(torch.int8)
                scales[where] = s
            view[..., :hd][where] = x.to(view.dtype)
        queries.append(q.bfloat16())
    raw, table = raw.to(DEV), table.to(DEV)
    kc, vc, ks, vs = carve(raw)
    case = SimpleNamespace(
        q=torch.cat(queries).to(DEV),
        kc=kc,
        vc=vc,
        ks=ks,
        vs=vs,
        cu=torch.tensor([0, *itertools.accumulate(q_lens)], device=DEV),
        seq_lens=torch.tensor(lens, device=DEV),
        table=table,
        scale=hd**-0.5,
        window=window,
        marks=marks,
    )
    case.fk = kc[..., :hd].float() * (ks[..., None] if int8 else 1)
    case.fv = vc[..., :hd].float() * (vs[..., None] if int8 else 1)
    return case


def expected(c, sinks=None, **wrong):
    """The reference of a case; `wrong` replaces its inputs or arguments."""
    args = dict(
        q=c.q,
        key_cache=c.fk,
        value_cache=c.fv,
        cu=c.cu,
        seq_lens=c.seq_lens,
        table=c.table,
        scale=c.scale,
        window_left=c.window,
        sinks=sinks,
    )
    return reference(**{**args, **wrong})


def prefill(c, splits=1, sinks=None):
    out = torch.full_like(c.q, float("nan"))
    strix.attn_prefill(
        out, c.q, c.kc, c.vc, c.cu, c.seq_lens, c.table, c.scale, splits, c.window,
        c.ks, c.vs, sinks,
    )
    return out


def decode(c, splits=1, sinks=None):
    out = torch.full_like(c.q, float("nan"))
    strix.attn_decode(
        out, c.q, c.kc, c.vc, c.seq_lens, c.table, c.scale, splits, c.window, c.ks,
        c.vs, sinks,
    )
    return out


# 27B: 24 q / 4 kv heads (G=6); 35B: 16 / 2 (G=8). Ragged q lens, prefix
# context (chunked prefill), block sizes 16 and a hybrid-style large block.
@pytest.mark.parametrize("hq,hkv", [(24, 4), (16, 2)])
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [([64], [0]), ([1], [0]), ([37], [0]), ([100, 5, 17], [0, 40, 300]), ([200], [77])],
)
@pytest.mark.parametrize("block_size", [16, 544])
def test_matches_reference(hq, hkv, q_lens, ctx_lens, block_size):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, hq, hkv, block_size)
    scale = 256**-0.5
    out = torch.full_like(q, float("nan"))
    strix.attn_prefill(out, q, kc, vc, cu, seq_lens, table, scale)
    ref = reference(q, kc, vc, cu, seq_lens, table, scale)
    assert_matches(out, ref)


# Split-KV for decode / speculative-verify shapes over long contexts; 13
# splits leaves some segments empty for the shorter rows.
@pytest.mark.parametrize("hq,hkv", [(24, 4), (16, 2)])
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [([1, 1], [3000, 77]), ([3] * 3, [5000, 20, 1200]), ([16], [2500])],
)
@pytest.mark.parametrize("num_splits", [4, 13])
def test_split_kv_matches_reference(hq, hkv, q_lens, ctx_lens, num_splits):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, hq, hkv, 784)
    scale = 256**-0.5
    out = torch.full_like(q, float("nan"))
    strix.attn_prefill(out, q, kc, vc, cu, seq_lens, table, scale, num_splits)
    ref = reference(q, kc, vc, cu, seq_lens, table, scale)
    assert_matches(out, ref)


# Gemma-4 31B full-attention layers: head_dim 512, 32 q / 4 kv heads (G=8).
@pytest.mark.parametrize(
    "q_lens,ctx_lens", [([64], [0]), ([37], [0]), ([100, 5, 17], [0, 40, 300])]
)
@pytest.mark.parametrize("block_size", [16, 272])
@pytest.mark.parametrize("num_splits", [1, 3])
def test_h512_matches_reference(q_lens, ctx_lens, block_size, num_splits):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, 32, 4, block_size, hd=512)
    scale = 512**-0.5
    out = torch.full_like(q, float("nan"))
    strix.attn_prefill(out, q, kc, vc, cu, seq_lens, table, scale, num_splits)
    ref = reference(q, kc, vc, cu, seq_lens, table, scale)
    assert_matches(out, ref)


# LFM2.5-24B-A2B attention layers: head_dim 64, 32 q / 8 kv heads (G=4).
@pytest.mark.parametrize(
    "q_lens,ctx_lens", [([64], [0]), ([37], [0]), ([100, 5, 17], [0, 40, 300])]
)
@pytest.mark.parametrize("block_size", [16, 784])
@pytest.mark.parametrize("num_splits", [1, 3])
def test_h64_matches_reference(q_lens, ctx_lens, block_size, num_splits):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, 32, 8, block_size, hd=64)
    scale = 64**-0.5
    out = torch.full_like(q, float("nan"))
    strix.attn_prefill(out, q, kc, vc, cu, seq_lens, table, scale, num_splits)
    ref = reference(q, kc, vc, cu, seq_lens, table, scale)
    assert_matches(out, ref)


# Gemma-4 sliding layers: head_dim 256, 32 q / 16 kv heads (G=2), window
# shrunk to 32 so short sequences cross it; decode-style 1-query rows too.
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [([64], [0]), ([100, 5, 17], [0, 40, 300]), ([1, 1, 1], [70, 200, 3])],
)
@pytest.mark.parametrize("num_splits", [1, 4])
@pytest.mark.parametrize("window_left", [-1, 31])
def test_g2_window_matches_reference(q_lens, ctx_lens, num_splits, window_left):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, 32, 16, 16)
    scale = 256**-0.5
    out = torch.full_like(q, float("nan"))
    strix.attn_prefill(
        out, q, kc, vc, cu, seq_lens, table, scale, num_splits, window_left
    )
    ref = reference(q, kc, vc, cu, seq_lens, table, scale, window_left)
    assert_matches(out, ref)


# attn_decode: one query per sequence (Gemma-4 sliding and full layers, the
# 35B's head 256 GQA 8, the 27B's head 256 GQA 6, which serving decodes on the
# vendored Triton kernel, Muse-Glimmer's head 128 GQA 16 on its full and
# windowed layers).
@pytest.mark.parametrize(
    "hq,hkv,hd,window_left",
    [
        (32, 16, 256, -1),
        (32, 16, 256, 31),
        (16, 2, 256, -1),
        (24, 4, 256, -1),
        (32, 4, 512, -1),
        (32, 2, 128, -1),
        (32, 2, 128, 31),
    ],
)
@pytest.mark.parametrize("ctx_lens", [[0], [70, 200, 3], [500, 31, 1000, 64]])
@pytest.mark.parametrize("block_size", [16, 272])
def test_decode_matches_reference(hq, hkv, hd, window_left, ctx_lens, block_size):
    q_lens = [1] * len(ctx_lens)
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, hq, hkv, block_size, hd=hd)
    scale = hd**-0.5
    ref = reference(q, kc, vc, cu, seq_lens, table, scale, window_left)
    for splits in (1, 4):
        out = torch.full_like(q, float("nan"))
        strix.attn_decode(out, q, kc, vc, seq_lens, table, scale, splits, window_left)
        assert_matches(out, ref)


# Decode on the int8_per_token_head cache: same geometries, bytes widened in
# LDS and the per-key scale folded into the score and P.
@pytest.mark.parametrize(
    "hq,hkv,hd",
    [(32, 16, 256), (16, 2, 256), (24, 4, 256), (32, 4, 512), (32, 2, 128)],
)
@pytest.mark.parametrize("ctx_lens", [[0], [70, 200, 3], [500, 31, 1000, 64]])
@pytest.mark.parametrize("block_size", [16, 272])
@pytest.mark.parametrize("window_left", [-1, 31])
def test_decode_int8_matches_reference(hq, hkv, hd, ctx_lens, block_size, window_left):
    q_lens = [1] * len(ctx_lens)
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, hq, hkv, block_size, hd=hd)
    scale = hd**-0.5
    kq, ks = quantize(kc, block_size, hkv)
    vq, vs = quantize(vc, block_size, hkv)
    kc, vc = kq.float() * ks[..., None], vq.float() * vs[..., None]
    ref = reference(q, kc, vc, cu, seq_lens, table, scale, window_left)
    for splits in (1, 4):
        out = torch.full_like(q, float("nan"))
        strix.attn_decode(
            out, q, kq, vq, seq_lens, table, scale, splits, window_left, ks, vs
        )
        assert_matches(out, ref)


# int8 KV (vLLM's int8_per_token_head): byte caches plus one fp32 dequant
# scale per (key, head). The kernel widens the bytes without the scale and
# applies it once per score, so this checks the folded path against a
# dequantized reference.
def quantize(q8_tensor, block_size, hkv):
    """[blocks, bs, Hkv, D] bf16 -> int8 bytes + fp32 [blocks, bs, Hkv]."""
    s = q8_tensor.abs().amax(-1).float().clamp_min(1e-8) / 127.0
    q = (q8_tensor.float() / s[..., None]).round().clamp(-127, 127).to(torch.int8)
    return q, s.contiguous()


@pytest.mark.parametrize("hq,hkv", [(24, 4), (16, 2)])
@pytest.mark.parametrize(
    "q_lens,ctx_lens", [([64], [0]), ([37], [0]), ([100, 5, 17], [0, 40, 300])]
)
@pytest.mark.parametrize("num_splits", [1, 4])
def test_int8_kv_matches_dequantized_reference(hq, hkv, q_lens, ctx_lens, num_splits):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, hq, hkv, 16)
    k_q, k_s = quantize(kc, 16, hkv)
    v_q, v_s = quantize(vc, 16, hkv)
    scale = 256**-0.5
    out = torch.full_like(q, float("nan"))
    strix.attn_prefill(
        out,
        q,
        k_q,
        v_q,
        cu,
        seq_lens,
        table,
        scale,
        num_splits,
        -1,
        k_s,
        v_s,
    )
    k_dq = k_q.float() * k_s[..., None]
    v_dq = v_q.float() * v_s[..., None]
    ref = reference(q, k_dq, v_dq, cu, seq_lens, table, scale)
    assert_matches(out, ref)


# Flint: head_dim 128, 32 q / 4 kv heads (G=8), learned attention sinks, and a
# 513-key sliding window (window_left 512) on most layers; shrunk to 40 here
# so short sequences cross it. bf16 and int8 KV.
@pytest.mark.parametrize(
    "q_lens,ctx_lens",
    [([64], [0]), ([37], [0]), ([100, 5, 17], [0, 40, 300]), ([200], [700])],
)
@pytest.mark.parametrize("window_left", [-1, 40, 512])
@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("int8", [False, True])
def test_h128_g8_sinks_window(q_lens, ctx_lens, window_left, use_sinks, int8):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, 32, 4, 16, hd=128)
    scale = 128**-0.5
    sinks = (torch.randn(32, device=DEV) * 2).float() if use_sinks else None
    out = torch.full_like(q, float("nan"))
    if int8:
        k_q, k_s = quantize(kc, 16, 4)
        v_q, v_s = quantize(vc, 16, 4)
        strix.attn_prefill(
            out, q, k_q, v_q, cu, seq_lens, table, scale, 1, window_left, k_s, v_s,
            sinks=sinks,
        )
        kc = k_q.float() * k_s[..., None]
        vc = v_q.float() * v_s[..., None]
    else:
        strix.attn_prefill(
            out, q, kc, vc, cu, seq_lens, table, scale, 1, window_left, sinks=sinks
        )
    ref = reference(q, kc, vc, cu, seq_lens, table, scale, window_left, sinks)
    assert_matches(out, ref)


# Muse-Glimmer: head 128, 32 q / 2 kv heads (G=16), no sinks, full layers and
# a 2048-key window (window_left 2047) on most layers; shrunk to 40 here so
# short sequences cross it. bf16 and int8 KV, split-KV.
@pytest.mark.parametrize(
    "q_lens,ctx_lens", [([37], [0]), ([100, 5, 17], [0, 40, 300]), ([200], [700])]
)
@pytest.mark.parametrize("block_size", [16, 272])
@pytest.mark.parametrize("window_left", [-1, 40])
@pytest.mark.parametrize("int8", [False, True])
def test_h128_g16_matches_reference(q_lens, ctx_lens, block_size, window_left, int8):
    c = served(
        q_lens, ctx_lens, 32, 2, 128, block_size, int8, window_left, planted=False
    )
    ref = expected(c)
    for splits in (1, 4, 7):
        assert_matches(prefill(c, splits), ref)


# Prefill tiles and speculative-verify shapes (4-16 query tokens, unequal
# lengths, so short sequences leave segments empty) with planted keys.
# Page 24 is past the 16-key step and short of the 32-key one; 1223 puts page
# edges inside steps.
PREFILL_SHAPES = [
    ([100, 5, 17], [0, 40, 1300]),
    ([8, 4, 16, 1], [2111, 20, 1300, 333]),
]


@pytest.mark.parametrize(
    "hq,hkv,hd,use_sinks",
    [
        (12, 2, 256, False),
        (16, 2, 256, False),
        (4, 2, 256, False),
        (16, 2, 512, False),
        (8, 2, 64, False),
        (32, 2, 128, False),
        (16, 2, 128, False),
        (16, 2, 128, True),
        (8, 2, 128, False),
    ],
)
@pytest.mark.parametrize("q_lens,ctx_lens", PREFILL_SHAPES)
@pytest.mark.parametrize("page", [24, 1223])
@pytest.mark.parametrize("window", [-1, 40])
@pytest.mark.parametrize("int8", [False, True])
def test_prefill_planted(
    hq, hkv, hd, use_sinks, q_lens, ctx_lens, page, window, int8
):
    c = served(q_lens, ctx_lens, hq, hkv, hd, page, int8, window)
    sinks = torch.linspace(10, 16, hq, device=DEV) if use_sinks else None
    ref = expected(c, sinks)
    for splits in (1, 4, 7):
        assert_matches(prefill(c, splits, sinks), ref)


# Decode. Pages: below, equal to and between the 16- and 32-key chunks (a
# 32-key chunk spans three 16- or 20-token pages), served sizes, and 1223,
# whose edges fall inside chunks of a full-attention layer too. Lengths end
# mid-chunk and mid-page, and fill the last page of 1223 and 2112.
DECODE_PAGES = [8, 16, 20, 1223, 1600, 2112]
DECODE_WINDOWS = [-1, 1003]
DECODE_CTX = [2111, 2600, 4307, 30, 1222]


@pytest.mark.parametrize(
    "hq,hkv,hd",
    [(4, 2, 256), (12, 2, 256), (16, 2, 256), (16, 2, 512), (16, 2, 128), (32, 2, 128), (8, 2, 64)],
)
@pytest.mark.parametrize("page", DECODE_PAGES)
@pytest.mark.parametrize("window", DECODE_WINDOWS)
@pytest.mark.parametrize("int8", [False, True])
def test_decode_planted(hq, hkv, hd, page, window, int8):
    c = served([1] * len(DECODE_CTX), DECODE_CTX, hq, hkv, hd, page, int8, window)
    ref = expected(c)
    for splits in (1, 7, 60):
        assert_matches(decode(c, splits), ref)


# Values of ~1e-3 under a flat softmax: the int8 tiles round P times V's scale
# to fp16, where the weights of these keys would be subnormal.
@pytest.mark.parametrize(
    "hq,hkv,hd",
    [(4, 2, 256), (12, 2, 256), (16, 2, 256), (16, 2, 512), (16, 2, 128), (32, 2, 128), (8, 2, 64)],
)
@pytest.mark.parametrize("v_gain", [2**-10, 2**-13])
def test_decode_small_values(hq, hkv, hd, v_gain):
    ctx = [2111, 700, 30]
    c = served([1] * 3, ctx, hq, hkv, hd, 256, True, planted=False, v_gain=v_gain)
    ref = expected(c)
    for splits in (1, 7):
        assert_matches(decode(c, splits), ref)


# Blocks of several query tiles (head groups of 2 and 4), with ragged tails, on
# pages that steps do and do not cross, and prefill batches with one long and
# many one-query sequences.
PREFILL_GEOMETRIES = [
    (12, 2, 256),
    (16, 2, 256),
    (4, 2, 256),
    (16, 2, 512),
    (8, 2, 64),
    (32, 2, 128),
    (16, 2, 128),
    (8, 2, 128),
]


@pytest.mark.parametrize("hq,hkv,hd", PREFILL_GEOMETRIES)
@pytest.mark.parametrize("q_lens", [[130, 64, 33, 1], [1, 1, 150, 1]])
@pytest.mark.parametrize("page", [64, 272])
@pytest.mark.parametrize("window", [-1, 1003])
@pytest.mark.parametrize("int8", [False, True])
def test_prefill_query_tiles(hq, hkv, hd, q_lens, page, window, int8):
    c = served(q_lens, [500, 0, 1411, 77], hq, hkv, hd, page, int8, window)
    assert_matches(prefill(c), expected(c))


# Verify batches of few queries: rows of several heads share a tile, and a
# sequence with more queries takes more blocks.
@pytest.mark.parametrize("hq,hkv,hd", PREFILL_GEOMETRIES)
@pytest.mark.parametrize(
    "q_lens", [[1, 1, 1], [2, 2, 2], [3, 5, 7], [8, 8, 8], [16, 1, 9], [4, 40, 2, 1]]
)
@pytest.mark.parametrize("window", [-1, 1003])
@pytest.mark.parametrize("int8", [False, True])
def test_verify_few_queries(hq, hkv, hd, q_lens, window, int8):
    c = served(q_lens, [2111, 77, 1300, 20][: len(q_lens)], hq, hkv, hd, 64, int8, window)
    ref = expected(c)
    for splits in (3, 7):
        assert_matches(prefill(c, splits), ref)


# Draft blocks (DFlash / DSpark): every query sees every key of its sequence,
# the ones past it included. window_left -2 is non-causal.
@pytest.mark.parametrize("hq,hkv,hd", PREFILL_GEOMETRIES)
@pytest.mark.parametrize("q_lens", [[16, 1, 9], [5, 5, 5], [8, 16, 3]])
@pytest.mark.parametrize("ctx_lens", [[2111, 77, 1300], [3, 0, 15]])
@pytest.mark.parametrize("window", [-1, 40, 1003])
@pytest.mark.parametrize("int8", [False, True])
def test_non_causal(hq, hkv, hd, q_lens, ctx_lens, window, int8):
    c = served(q_lens, ctx_lens, hq, hkv, hd, 64, int8, window)
    ref = expected(c, ahead=1 << 20)
    c.window = -2 if window < 0 else -3 - window
    for splits in (1, 3, 7):
        assert_matches(prefill(c, splits), ref)


# The served draft window (head 128 GQA 4): a 2048-key window, q_lens of a
# block and of a partial one.
@pytest.mark.parametrize("page", [64, 256])
@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("q_lens", [[8, 16, 3, 5], [4, 4, 4, 4]])
def test_draft_window(page, int8, q_lens):
    ctx = [2046, 2047, 2048, 4131]
    c = served(q_lens, ctx, 8, 2, 128, page, int8, 2047)
    ref = expected(c)
    for splits in (1, 5, 7):
        assert_matches(prefill(c, splits), ref)


# As test_decode_small_values.
@pytest.mark.parametrize("hq,hkv,hd", PREFILL_GEOMETRIES)
@pytest.mark.parametrize("v_gain", [2**-10, 2**-12])
def test_prefill_small_values(hq, hkv, hd, v_gain):
    ctx = [2111, 700, 30]
    c = served([40, 3, 1], ctx, hq, hkv, hd, 256, True, planted=False, v_gain=v_gain)
    ref = expected(c)
    for splits in (1, 7):
        assert_matches(prefill(c, splits), ref)


# The served windows (Muse-Glimmer, Gemma-4 sliding, Flint) around their edge,
# on the served pages. A window one key too long or too short is wrong here.
@pytest.mark.parametrize(
    "hq,hkv,hd,window,page",
    [(32, 2, 128, 2047, 256), (4, 2, 256, 1023, 64), (16, 2, 128, 512, 128)],
)
@pytest.mark.parametrize("op", [prefill, decode])
@pytest.mark.parametrize("int8", [False, True])
def test_served_window(hq, hkv, hd, window, page, op, int8):
    ctx = [window - 1, window, window + 1, 2 * window + 37]
    q_lens = [40, 16, 3, 5] if op is prefill else [1] * 4
    if op is prefill:
        ctx[0] -= 20
    c = served(q_lens, ctx, hq, hkv, hd, page, int8, window)
    ref = expected(c)
    for splits in (1, 7):
        assert_matches(op(c, splits), ref)


# Sinks join the softmax once, after the segments combine. These are as strong
# as all keys together, so a fold per segment is far outside the tolerance.
@pytest.mark.parametrize("op", [prefill, decode])
@pytest.mark.parametrize("window", [-1, 40])
@pytest.mark.parametrize("int8", [False, True])
def test_sinks_split_kv(op, window, int8):
    c, sinks = strong_sinks([4, 2, 16] if op is prefill else [1, 1, 1], window, int8)
    ref = expected(c, sinks)
    for splits in (1, 4, 13):
        assert_matches(op(c, splits, sinks), ref)


def strong_sinks(q_lens, window, int8):
    ctx = [600, 70, 135]
    c = served(q_lens, ctx, 32, 4, 128, 16, int8, window, planted=False)
    keys = min(600, window + 1) if window >= 0 else 600
    return c, math.log(keys) + torch.linspace(-0.5, 2.5, 32, device=DEV)


# Flint decode: head 128 GQA 8, sinks folded in on the single-pass epilogue
# and, with split-KV, once in the segment reduce; 513-key window shrunk to 40.
@pytest.mark.parametrize("ctx_lens", [[0], [70, 200, 3], [500, 31, 1000, 64]])
@pytest.mark.parametrize("num_splits", [1, 4])
@pytest.mark.parametrize("window_left", [-1, 40])
@pytest.mark.parametrize("use_sinks", [False, True])
@pytest.mark.parametrize("int8", [False, True])
def test_h128_g8_decode_sinks(ctx_lens, num_splits, window_left, use_sinks, int8):
    q_lens = [1] * len(ctx_lens)
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, 32, 4, 16, hd=128)
    scale = 128**-0.5
    sinks = (torch.randn(32, device=DEV) * 2).float() if use_sinks else None
    out = torch.full_like(q, float("nan"))
    if int8:
        kq, ks = quantize(kc, 16, 4)
        vq, vs = quantize(vc, 16, 4)
        strix.attn_decode(
            out, q, kq, vq, seq_lens, table, scale, num_splits, window_left, ks, vs,
            sinks=sinks,
        )
        kc, vc = kq.float() * ks[..., None], vq.float() * vs[..., None]
    else:
        strix.attn_decode(
            out, q, kc, vc, seq_lens, table, scale, num_splits, window_left, sinks=sinks
        )
    ref = reference(q, kc, vc, cu, seq_lens, table, scale, window_left, sinks)
    assert_matches(out, ref)


# --- The checker: what it must accept and what it must flag, on the same cases.


def check_case(planted=True, int8=False, window=-1, q_lens=(1, 1, 1)):
    ctx = [4308 - q_lens[0], 1223 - q_lens[1], 61 - q_lens[2]]
    return served(list(q_lens), ctx, 16, 2, 256, 611, int8, window, planted)


CHECKED = pytest.mark.parametrize(
    "planted,int8,window",
    [
        (True, False, -1),
        (True, True, -1),
        (True, True, 1003),
        (False, False, -1),
        (False, True, 1003),
    ],
)


@CHECKED
def test_checker_accepts_rounding(planted, int8, window):
    c = check_case(planted, int8, window, (16, 5, 1))
    ref = expected(c)
    for dtype in (torch.bfloat16, torch.float16):
        assert_matches(expected(c, p_dtype=dtype).bfloat16(), ref)


@CHECKED
def test_checker_flags_lost_keys(planted, int8, window):
    c = check_case(planted, int8, window)
    ref = expected(c)
    for n, marks in enumerate(c.marks):
        # One of 4308 keys of equal weight is below the tolerance.
        for key in set(marks.flatten().tolist()) if planted or n else ():
            assert mismatch(expected(c, drop=[(n, key)]), ref), (n, key)
        tail = int(c.seq_lens[n]) - 1
        keys = range(tail // 16 * 16, tail + 1)
        assert mismatch(expected(c, drop=[(n, k) for k in keys]), ref)


@CHECKED
def test_checker_flags_wrong_addresses(planted, int8, window):
    c = check_case(planted, int8, window)
    ref = expected(c)

    def flagged(**wrong):
        return mismatch(expected(c, **wrong), ref)

    # A page table entry of one sequence off by one page.
    table = c.table.clone()
    last = (int(c.seq_lens[0]) - 1) // 611
    table[0, last] = c.table[0, last - 1]
    assert flagged(table=table)
    # The first keys of that page read from the page before it.
    for keys in (1, 4) if planted else (4, 16):
        fk, fv = c.fk.clone(), c.fv.clone()
        here, before = int(c.table[0, last]), int(c.table[0, last - 1])
        fk[here, :keys] = c.fk[before, :keys]
        fv[here, :keys] = c.fv[before, :keys]
        assert flagged(key_cache=fk, value_cache=fv)
    # The other KV head, and one slot later.
    assert flagged(key_cache=c.fk.flip(2))
    assert flagged(value_cache=c.fv.flip(2))
    assert flagged(value_cache=c.fv.roll(1, 1))
    if int8:
        for name, x, s in (("key_cache", c.fk, c.ks), ("value_cache", c.fv, c.vs)):
            bytes_ = x / s[..., None]
            assert flagged(**{name: bytes_ * s.roll(1, 1)[..., None]})
            # Sixteen wrong scales in the last page.
            wrong = s.clone()
            wrong[int(c.table[0, last]), :16] = s[int(c.table[0, last]), 16:32]
            assert flagged(**{name: bytes_ * wrong[..., None]})


@pytest.mark.parametrize("q_lens", [(1, 1, 1), (16, 5, 1)])
@pytest.mark.parametrize("int8", [False, True])
def test_checker_flags_mask_errors(q_lens, int8):
    for window in (63, 255, 1003):
        c = check_case(True, int8, window, q_lens)
        ref = expected(c)
        assert mismatch(expected(c, window_left=window + 1), ref)
        assert mismatch(expected(c, window_left=window - 1), ref)
        assert q_lens == (1, 1, 1) or mismatch(expected(c, ahead=1), ref)
    assert mismatch(ref + torch.nan, ref)


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("window", [-1, 40])
def test_checker_flags_sinks_per_segment(int8, window):
    c, sinks = strong_sinks([4, 2, 16], window, int8)
    ref = expected(c, sinks)
    for splits in (4, 13):
        folded = expected(c, sinks + math.log(splits))
        assert row_errors(folded, ref)[0].min() > 20 * ROW_RMS_TOL
    assert mismatch(expected(c), ref)


# Pages that a chunk of the decode kernel touches, over the cases of
# test_decode_planted: those it is there for must be among them.
def test_checker_decode_cases_cross_pages():
    touched = {}
    for page, window, ctx, splits, kc in itertools.product(
        DECODE_PAGES, DECODE_WINDOWS, DECODE_CTX, (1, 7, 60), (16, 32)
    ):
        first = max(0, ctx - window) if window >= 0 else 0
        seg = (-(-(ctx + 1 - first) // splits) + kc - 1) // kc * kc
        ends = range(first, ctx + 1, seg)
        for k0 in itertools.chain(*(range(e, min(e + seg, ctx + 1), kc) for e in ends)):
            seg_end = first + (k0 - first) // seg * seg + seg
            last = min(k0 + kc, ctx + 1, seg_end) - 1
            spans = last // page - k0 // page + 1
            key = (kc, page, window)
            touched[key] = max(touched.get(key, 0), spans)
    for kc, page, window in touched:
        most = touched[kc, page, window]
        if page < 16:
            assert most == kc // page + (window >= 0)
        elif kc == 32 and page < 32:
            assert most == (3 if page == 20 or window >= 0 else 2)
        elif page in (20, 1223) or window >= 0:
            assert most == 2
