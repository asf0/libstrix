# SPDX-License-Identifier: MIT
import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


def build(q_lens, ctx_lens, hq, hkv, block_size, seed=0, hd=256):
    """Random paged KV in vLLM's fork layout [blocks, Hkv, bs, 2*hs] viewed as
    [blocks, bs, Hkv, hs] K/V caches, with shuffled block tables."""
    torch.manual_seed(seed)
    seq_lens = [q + c for q, c in zip(q_lens, ctx_lens)]
    nblk = [(s + block_size - 1) // block_size for s in seq_lens]
    total = sum(nblk) + 3
    kv = torch.randn(total, hkv, block_size, 2 * hd, device="cuda").bfloat16()
    key_cache, value_cache = kv.transpose(1, 2).split(hd, dim=-1)
    perm = torch.randperm(total, device="cuda")
    table = torch.zeros(len(q_lens), max(nblk), dtype=torch.int32, device="cuda")
    used = 0
    for i, nb in enumerate(nblk):
        table[i, :nb] = perm[used : used + nb]
        used += nb
    t = sum(q_lens)
    q = torch.randn(t, hq, hd, device="cuda").bfloat16()
    cu = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda")
    return q, key_cache, value_cache, cu, torch.tensor(seq_lens, device="cuda"), table


def reference(q, key_cache, value_cache, cu, seq_lens, table, scale, window_left=-1):
    hq, hkv = q.size(1), key_cache.size(2)
    bs = key_cache.size(1)
    out = torch.empty_like(q, dtype=torch.float32)
    for n in range(cu.numel() - 1):
        a, b, kv_len = int(cu[n]), int(cu[n + 1]), int(seq_lens[n])
        pos = torch.arange(kv_len, device="cuda")
        blocks = table[n, pos // bs].long()
        k = key_cache[blocks, pos % bs].float()  # [kv_len, Hkv, D]
        v = value_cache[blocks, pos % bs].float()
        k = k.repeat_interleave(hq // hkv, 1)
        v = v.repeat_interleave(hq // hkv, 1)
        qs = q[a:b].float()
        s = torch.einsum("thd,khd->htk", qs, k) * scale
        qpos = kv_len - (b - a) + torch.arange(b - a, device="cuda")
        s.masked_fill_(pos[None, None, :] > qpos[None, :, None], float("-inf"))
        if window_left >= 0:
            s.masked_fill_(
                qpos[None, :, None] - pos[None, None, :] > window_left, float("-inf")
            )
        out[a:b] = torch.einsum("htk,khd->thd", s.softmax(-1), v)
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
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


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
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


# Gemma-4 31B full-attention layers: head_dim 512, 32 q / 4 kv heads (G=8).
@pytest.mark.parametrize(
    "q_lens,ctx_lens", [([64], [0]), ([37], [0]), ([100, 5, 17], [0, 40, 300])]
)
@pytest.mark.parametrize("block_size", [16, 272])
@pytest.mark.parametrize("num_splits", [1, 3])
def test_h512_matches_reference(q_lens, ctx_lens, block_size, num_splits):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, 32, 4, block_size, hd=512)
    scale = 512**-0.5
    out = torch.empty_like(q)
    strix.attn_prefill(out, q, kc, vc, cu, seq_lens, table, scale, num_splits)
    ref = reference(q, kc, vc, cu, seq_lens, table, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


# LFM2.5-24B-A2B attention layers: head_dim 64, 32 q / 8 kv heads (G=4).
@pytest.mark.parametrize(
    "q_lens,ctx_lens", [([64], [0]), ([37], [0]), ([100, 5, 17], [0, 40, 300])]
)
@pytest.mark.parametrize("block_size", [16, 784])
@pytest.mark.parametrize("num_splits", [1, 3])
def test_h64_matches_reference(q_lens, ctx_lens, block_size, num_splits):
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, 32, 8, block_size, hd=64)
    scale = 64**-0.5
    out = torch.empty_like(q)
    strix.attn_prefill(out, q, kc, vc, cu, seq_lens, table, scale, num_splits)
    ref = reference(q, kc, vc, cu, seq_lens, table, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


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
    out = torch.empty_like(q)
    strix.attn_prefill(
        out, q, kc, vc, cu, seq_lens, table, scale, num_splits, window_left
    )
    ref = reference(q, kc, vc, cu, seq_lens, table, scale, window_left)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


# attn_decode: one query per sequence (Gemma-4 sliding and full layers, the
# 35B's head 256 GQA 8).
@pytest.mark.parametrize(
    "hq,hkv,hd,window_left",
    [(32, 16, 256, -1), (32, 16, 256, 31), (16, 2, 256, -1), (32, 4, 512, -1)],
)
@pytest.mark.parametrize("ctx_lens", [[0], [70, 200, 3], [500, 31, 1000, 64]])
@pytest.mark.parametrize("num_splits", [1, 4])
@pytest.mark.parametrize("block_size", [16, 272])
def test_decode_matches_reference(
    hq, hkv, hd, window_left, ctx_lens, num_splits, block_size
):
    q_lens = [1] * len(ctx_lens)
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, hq, hkv, block_size, hd=hd)
    scale = hd**-0.5
    out = torch.empty_like(q)
    strix.attn_decode(out, q, kc, vc, seq_lens, table, scale, num_splits, window_left)
    ref = reference(q, kc, vc, cu, seq_lens, table, scale, window_left)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


# Decode on the int8_per_token_head cache: same geometries, bytes widened in
# LDS and the per-key scale folded into the score and P.
@pytest.mark.parametrize("hq,hkv,hd", [(32, 16, 256), (16, 2, 256), (32, 4, 512)])
@pytest.mark.parametrize("ctx_lens", [[0], [70, 200, 3], [500, 31, 1000, 64]])
@pytest.mark.parametrize("num_splits", [1, 4])
@pytest.mark.parametrize("block_size", [16, 272])
@pytest.mark.parametrize("window_left", [-1, 31])
def test_decode_int8_matches_reference(
    hq, hkv, hd, ctx_lens, num_splits, block_size, window_left
):
    q_lens = [1] * len(ctx_lens)
    q, kc, vc, cu, seq_lens, table = build(q_lens, ctx_lens, hq, hkv, block_size, hd=hd)
    scale = hd**-0.5
    kq, ks = quantize(kc, block_size, hkv)
    vq, vs = quantize(vc, block_size, hkv)
    out = torch.empty_like(q)
    strix.attn_decode(
        out, q, kq, vq, seq_lens, table, scale, num_splits, window_left, ks, vs
    )
    ref = reference(
        q,
        kq.float() * ks[..., None],
        vq.float() * vs[..., None],
        cu,
        seq_lens,
        table,
        scale,
        window_left,
    )
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


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
    k_dq = (k_q.float() * k_s[..., None]).bfloat16()
    v_dq = (v_q.float() * v_s[..., None]).bfloat16()
    ref = reference(q, k_dq, v_dq, cu, seq_lens, table, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
