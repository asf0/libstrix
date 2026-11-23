# SPDX-License-Identifier: MIT
"""inkling_rel_attn_prefill (Inkling 2 relative attention with the gated,
sink-scaled epilogue) against a dense fp32 reference."""
import pytest
import strix
import torch

from test_attn_prefill import build

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

HQ, HKV, D = 24, 8, 128


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
        out, q, kc, vc, rel, gate, sinks, cu, seq_lens, table, offsets, max_tiles,
        scale, window_left, q_tiles, block_n,
    )
    ref = reference(q, kc, vc, cu, seq_lens, table, rel, gate, sinks, scale, window_left)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
