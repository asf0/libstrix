# SPDX-License-Identifier: MIT
"""strix::inkling2_kv_conv against the port's Triton attention_prologue (K/V
part) on a chunked-prefill batch: a request continuing from cached positions,
a fresh one, and padding tokens."""
import pytest
import torch

import strix  # noqa: F401
from vllm.models.inkling2.amd.ops import sconv as S

dev = torch.device("cuda")
H, D, W, N, PAGE, HQ = 8, 128, 4, 4, 64, 24
DPACK = 1024


def batch(q_lens, ctx, pad, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    T = sum(q_lens) + pad
    pos, seq, qstart, cslot, kvslot = [], [], [], [], []
    nblk = [(c + q + N - 1) // N for c, q in zip(ctx, q_lens)]
    total = sum(nblk) + 2
    perm = torch.randperm(total, device=dev, generator=g).int()
    table = torch.zeros(len(q_lens), max(nblk), dtype=torch.int32, device=dev)
    used, row = 0, 0
    for r, (q, c) in enumerate(zip(q_lens, ctx)):
        table[r, : nblk[r]] = perm[used : used + nblk[r]]
        used += nblk[r]
        for i in range(q):
            p = c + i
            pos.append(p); seq.append(r); qstart.append(row)
            cslot.append(int(table[r, p // N]) * N + p % N)
        row += q
    kv_perm = torch.randperm(T * 2, device=dev, generator=g)[:T]
    for i in range(sum(q_lens)):
        kvslot.append(int(kv_perm[i]))
    for _ in range(pad):  # CUDA-graph padding rows
        pos.append(0); seq.append(0); qstart.append(0); cslot.append(-1); kvslot.append(-1)
    md = dict(
        positions=torch.tensor(pos, dtype=torch.int64, device=dev),
        seq_idx=torch.tensor(seq, dtype=torch.int32, device=dev),
        query_start=torch.tensor(qstart, dtype=torch.int32, device=dev),
        slot_mapping=torch.tensor(cslot, dtype=torch.int64, device=dev),
        block_table=table,
    )
    cache = torch.randn(total, H, N, DPACK, device=dev, generator=g).bfloat16()
    return T, md, cache, torch.tensor(kvslot, dtype=torch.int64, device=dev), g


@pytest.mark.parametrize("dynamic", [False, True])
def test_kv_conv_matches_prologue(dynamic):
    T, md, cache, kv_slot, g = batch([600, 900], [5, 0], 3)
    proj = torch.randn(T, 2 * H * D + 2 * H * W + 8, device=dev, generator=g).bfloat16()
    k, v = proj[:, : H * D], proj[:, H * D : 2 * H * D]
    if dynamic:
        kw = (torch.randn(W, H * D, device=dev, generator=g) * 0.3).bfloat16()
        vw = (torch.randn(W, H * D, device=dev, generator=g) * 0.3).bfloat16()
        kd = proj[:, 2 * H * D : 2 * H * D + H * W] * 0.2
        vd = proj[:, 2 * H * D + H * W : 2 * H * D + 2 * H * W] * 0.2
    else:
        kw = (torch.randn(H * D, W, device=dev, generator=g) * 0.3).bfloat16()
        vw = (torch.randn(H * D, W, device=dev, generator=g) * 0.3).bfloat16()
        kd = vd = None
    kv = torch.zeros(2 * T // PAGE + 2, H, PAGE, 2 * D, device=dev).bfloat16()
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    kv2 = kv.clone()
    kc2, vc2 = kv2.transpose(1, 2).split(D, dim=-1)
    cache2 = cache.clone()

    q = torch.randn(T, HQ, D, device=dev, generator=g).bfloat16()
    r = torch.randn(T, HQ, 16, device=dev, generator=g).bfloat16()
    rel_proj = S.shift_rel_proj(torch.randn(16, 1024, device=dev, generator=g).bfloat16(), True)
    cu = torch.tensor([0, 600, 1500], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([605, 900], dtype=torch.int32, device=dev)
    S.attention_prologue(
        q, r, k, v, torch.ones(HQ, D, device=dev).bfloat16(),
        torch.ones(HQ, 16, device=dev).bfloat16(), None, rel_proj, kw, vw, kd, vd, cache,
        md["positions"], md["block_table"], md["seq_idx"], md["slot_mapping"],
        md["query_start"], kc, vc, kv_slot, torch.empty_like(q),
        q.new_empty(T + 128, HQ, 1024 + 256), cu, seq_lens,
        clip_threshold=5.0, rel_threshold=5.0, norm_eps=1e-6,
    )
    if dynamic:
        wk, wv = kw.float().t().contiguous(), vw.float().t().contiguous()
    else:
        wk, wv = kw.float().contiguous(), vw.float().contiguous()
    scale = (W * (2 if dynamic else 1) + 1) ** -0.5
    torch.ops.strix.inkling2_kv_conv(
        k, v, wk, wv, kd, vd, cache2, kc2, vc2, kv_slot, md["positions"], md["seq_idx"],
        md["slot_mapping"], md["block_table"], md["query_start"], scale, 5.0, 1e-6, 1e-6)
    torch.testing.assert_close(cache2, cache, rtol=0, atol=0)
    # bf16 outputs; the per-head sum of squares is summed in another order.
    torch.testing.assert_close(kv2.float(), kv.float(), rtol=1e-2, atol=1e-2)
    assert (kv2 != kv).float().mean() < 1e-3


def _quantize_ref(t):
    """[blocks, page, H, D] bf16 -> (int8, fp32 [blocks, page, H]) as the
    kernel stores it: scale amax / 127 (floor 1e-6), round half away from
    zero, clamp [-128, 127]."""
    s = (t.abs().amax(-1).float() / 127.0).clamp_min(1e-6)
    z = t.float() / s[..., None]
    z = torch.where(z >= 0, torch.floor(z + 0.5), torch.ceil(z - 0.5))
    return z.clamp(-128, 127).to(torch.int8), s


@pytest.mark.parametrize("dynamic", [False, True])
def test_kv_conv_int8_matches_quantized_prologue(dynamic):
    """The int8 write against the port's bf16 prologue, quantized: the port
    has no int8 path, so its bf16 output (dequantized from the same
    convention) is the reference."""
    T, md, cache, kv_slot, g = batch([600, 900], [5, 0], 3)
    proj = torch.randn(T, 2 * H * D + 2 * H * W + 8, device=dev, generator=g).bfloat16()
    k, v = proj[:, : H * D], proj[:, H * D : 2 * H * D]
    if dynamic:
        kw = (torch.randn(W, H * D, device=dev, generator=g) * 0.3).bfloat16()
        vw = (torch.randn(W, H * D, device=dev, generator=g) * 0.3).bfloat16()
        kd = proj[:, 2 * H * D : 2 * H * D + H * W] * 0.2
        vd = proj[:, 2 * H * D + H * W : 2 * H * D + 2 * H * W] * 0.2
    else:
        kw = (torch.randn(H * D, W, device=dev, generator=g) * 0.3).bfloat16()
        vw = (torch.randn(H * D, W, device=dev, generator=g) * 0.3).bfloat16()
        kd = vd = None
    kv = torch.zeros(2 * T // PAGE + 2, H, PAGE, 2 * D, device=dev).bfloat16()
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    q = torch.randn(T, HQ, D, device=dev, generator=g).bfloat16()
    r = torch.randn(T, HQ, 16, device=dev, generator=g).bfloat16()
    rel_proj = S.shift_rel_proj(torch.randn(16, 1024, device=dev, generator=g).bfloat16(), True)
    cu = torch.tensor([0, 600, 1500], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([605, 900], dtype=torch.int32, device=dev)
    S.attention_prologue(
        q, r, k, v, torch.ones(HQ, D, device=dev).bfloat16(),
        torch.ones(HQ, 16, device=dev).bfloat16(), None, rel_proj, kw, vw, kd, vd, cache,
        md["positions"], md["block_table"], md["seq_idx"], md["slot_mapping"],
        md["query_start"], kc, vc, kv_slot, torch.empty_like(q),
        q.new_empty(T + 128, HQ, 1024 + 256), cu, seq_lens,
        clip_threshold=5.0, rel_threshold=5.0, norm_eps=1e-6,
    )
    if dynamic:
        wk, wv = kw.float().t().contiguous(), vw.float().t().contiguous()
    else:
        wk, wv = kw.float().contiguous(), vw.float().contiguous()
    scale = (W * (2 if dynamic else 1) + 1) ** -0.5
    # The served packed layout: [K | K_scale | V | V_scale] per (head, slot).
    nblk = kv.shape[0]
    packed = torch.zeros(nblk, H, PAGE, 2 * (D + 4), dtype=torch.int8, device=dev)
    t = packed.transpose(1, 2)
    kq, vq = t[:, :, :, :D], t[:, :, :, D + 4 : 2 * D + 4]
    torch.ops.strix.inkling2_kv_conv(
        k, v, wk, wv, kd, vd, cache.clone(), kq, vq, kv_slot, md["positions"],
        md["seq_idx"], md["slot_mapping"], md["block_table"], md["query_start"],
        scale, 5.0, 1e-6, 1e-6,
    )
    k_ref, k_s = _quantize_ref(kc)
    v_ref, v_s = _quantize_ref(vc)
    # The per-head sum of squares is summed in another order: a value at a
    # rounding boundary may sit one code off.
    assert (kq.float() - k_ref.float()).abs().max() <= 1.0
    assert (vq.float() - v_ref.float()).abs().max() <= 1.0
    # The scale follows the amax of the unrounded fp32 output, a bf16 ULP
    # from the reference's (the other-order sum of squares), hence rtol 1e-2.
    k_scale = t[:, :, :, D : D + 4].contiguous().view(torch.float32)[..., 0]
    v_scale = t[:, :, :, 2 * D + 4 : 2 * D + 8].contiguous().view(torch.float32)[..., 0]
    torch.testing.assert_close(k_scale, k_s, rtol=1e-2, atol=1e-6)
    torch.testing.assert_close(v_scale, v_s, rtol=1e-2, atol=1e-6)
    # One code (the reference amax / 127) of the two conv sum orders'
    # rounding, plus bf16 noise.
    step_k = 1.02 * float(k_s.max())
    step_v = 1.02 * float(v_s.max())
    torch.testing.assert_close(
        kq.float() * k_scale[..., None], k_ref.float() * k_s[..., None],
        rtol=2e-2, atol=2e-2 + step_k)
    torch.testing.assert_close(
        vq.float() * v_scale[..., None], v_ref.float() * v_s[..., None],
        rtol=2e-2, atol=2e-2 + step_v)
