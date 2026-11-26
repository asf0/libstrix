# SPDX-License-Identifier: MIT
"""inkling2_kv_conv (the Inkling 2 attention-prologue K / V short convs,
the K clip and V RMS norm, the conv-state and KV-cache writes) against a
torch reference, static and dynamic taps, bf16 and int8 (vLLM's
int8_per_token_head, the scale inline after each head's 128 values) KV."""
import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

H, D = 8, 128
W = 4
N_CONV = 16  # conv-state block (a power of two)
PAGE = 16    # KV page (a power of two)


def reference(k, v, wk, wv, k_dyn, v_dyn, cache, positions, seq_idx, qstart,
              kv_slot, conv_slot, block_table, scale, clip, clip_eps, norm_eps,
              n_conv, page):
    """The processed K / V per (token, head) (fp32, after the bf16 rounding
    the kernel does before the clip / norm)."""
    t = positions.size(0)
    out_k = torch.zeros(t, H, D, device="cuda")
    out_v = torch.zeros(t, H, D, device="cuda")
    for i in range(t):
        p, req, qs = int(positions[i]), int(seq_idx[i]), int(qstart[i])
        for h in range(H):
            for x, dyn, wt, is_v in (
                (k, k_dyn, wk, False),
                (v, v_dyn, wv, True),
            ):
                # The K / V input and weight rows are [H * D, ...] each (a
                # separate tensor per side); the conv cache holds both sides
                # in one [H, N, 2 * D] head row.
                xoff, woff = h * D, h * D
                coff = D if is_v else 0
                # xs[lag] = the input row at token i - lag (zero for a
                # padding token or before position 0, the conv state for
                # rows of an earlier chunk of the request).
                xs = []
                for lag in range(W):
                    row, src = i - lag, p - lag
                    if kv_slot[i] < 0 or src < 0:
                        xs.append(torch.zeros(D, device="cuda"))
                    elif row >= qs:
                        xs.append(x[row, xoff : xoff + D].float())
                    else:
                        b = int(block_table[req, src // n_conv])
                        xs.append(cache[b, h, src % n_conv, coff : coff + D].float())
                if dyn is not None:
                    dl = dyn[i, h * W : (h + 1) * W].float()
                    acc = torch.zeros(D, device="cuda")
                    for lag in range(W):
                        acc = acc + xs[lag] * (wt[woff : woff + D, lag].float() + dl[lag])
                else:
                    acc = torch.zeros(D, device="cuda")
                    for kk in range(W):
                        acc = acc + xs[W - 1 - kk] * wt[woff : woff + D, kk].float()
                y = ((acc + x[i, xoff : xoff + D].float()) * scale).bfloat16().float()
                ss = (y * y).sum()
                if is_v:
                    f = torch.rsqrt(ss / D + norm_eps)
                else:
                    norm = ss * torch.rsqrt(ss) if ss > 0 else 0.0
                    f = (clip / (norm + clip_eps)).clamp(max=1.0)
                (out_v if is_v else out_k)[i, h] = y * f
    return out_k, out_v


def _quantize(y):
    """One (token, head) row to vLLM's int8: scale amax / 127 (floor 1e-6),
    round half away from zero, clamp [-128, 127]."""
    s = (y.abs().amax() / 127.0).clamp_min(1e-6)
    z = y / s
    z = torch.where(z >= 0, torch.floor(z + 0.5), torch.ceil(z - 0.5))
    return z.clamp(-128, 127).to(torch.int8), s.float()


def make_case(seed, dynamic):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = 10
    # Requests: 0 has 6 tokens at positions 24..29 (24 of earlier context in
    # the conv state), 1 has 2 tokens at 0..1, 2 has 1 token at 3; row 9 is
    # a padding row.
    positions = torch.tensor([24, 25, 26, 27, 28, 29, 0, 1, 3, -5], device="cuda")
    seq_idx = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=torch.int32, device="cuda")
    qstart = torch.tensor([0, 0, 0, 0, 0, 0, 6, 6, 8, 9], dtype=torch.int32, device="cuda")
    conv_slot = torch.cat(
        [torch.arange(0, 9, device="cuda"), torch.tensor([-1], device="cuda")]
    )
    kv_slot = torch.tensor([40, 41, 42, 43, 44, 45, 5, 6, 20, -1], device="cuda")
    block_table = torch.tensor(
        [[3, 5, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]], dtype=torch.int32, device="cuda"
    )
    cache = torch.randn(8, H, N_CONV, 2 * D, device="cuda", generator=g).bfloat16()
    packed = torch.zeros(8, H, PAGE, 264, dtype=torch.int8, device="cuda")
    k = torch.randn(t, H * D, device="cuda", generator=g).bfloat16()
    v = torch.randn(t, H * D, device="cuda", generator=g).bfloat16()
    wk = torch.randn(H * D, W, device="cuda", generator=g).bfloat16()
    wv = torch.randn(H * D, W, device="cuda", generator=g).bfloat16()
    if dynamic:
        k_dyn = (torch.randn(t, H * W, device="cuda", generator=g) * 0.5).bfloat16()
        v_dyn = (torch.randn(t, H * W, device="cuda", generator=g) * 0.5).bfloat16()
    else:
        k_dyn = v_dyn = None
    scale = (W * (2 if dynamic else 1) + 1) ** -0.5
    clip, clip_eps, norm_eps = 22.6, 1e-6, 1e-6
    return dict(
        t=t, positions=positions, seq_idx=seq_idx, qstart=qstart, conv_slot=conv_slot,
        kv_slot=kv_slot, block_table=block_table, cache=cache, packed=packed, k=k,
        v=v, wk=wk, wv=wv, k_dyn=k_dyn, v_dyn=v_dyn, scale=scale, clip=clip,
        clip_eps=clip_eps, norm_eps=norm_eps,
    )


def run(case, int8):
    packed = case["packed"].clone()
    cache = case["cache"].clone()
    if int8:
        key_cache = packed.transpose(1, 2)[:, :, :, :128]
        value_cache = packed.transpose(1, 2)[:, :, :, 132:260]
    else:
        key_cache = torch.zeros(8, PAGE, H, D, dtype=torch.bfloat16, device="cuda")
        value_cache = torch.zeros_like(key_cache)
    torch.ops.strix.inkling2_kv_conv(
        case["k"], case["v"], case["wk"].float().contiguous(),
        case["wv"].float().contiguous(), case["k_dyn"], case["v_dyn"], cache,
        key_cache, value_cache, case["kv_slot"], case["positions"],
        case["seq_idx"], case["conv_slot"], case["block_table"], case["qstart"],
        case["scale"], case["clip"], case["clip_eps"], case["norm_eps"],
    )
    torch.cuda.synchronize()
    return key_cache, value_cache, packed, cache


def check(case, int8):
    key_cache, value_cache, packed, cache = run(case, int8)
    ref_k, ref_v = reference(
        case["k"], case["v"], case["wk"], case["wv"], case["k_dyn"], case["v_dyn"],
        case["cache"], case["positions"], case["seq_idx"], case["qstart"],
        case["kv_slot"], case["conv_slot"], case["block_table"], case["scale"],
        case["clip"], case["clip_eps"], case["norm_eps"], N_CONV, PAGE,
    )
    for i in range(case["t"]):
        if case["kv_slot"][i] < 0:
            continue
        s = int(case["kv_slot"][i])
        blk, off = s // PAGE, s % PAGE
        for h in range(H):
            if int8:
                kq, ks = _quantize(ref_k[i, h])
                vq, vs = _quantize(ref_v[i, h])
                # The conv's fp32 sum order differs from torch's by ULPs, so
                # a quantized value may sit at the other side of a rounding
                # boundary: one code off is fine.
                assert (key_cache[blk, off, h].float() - kq.float()).abs().max() <= 1.0, (i, h)
                assert (value_cache[blk, off, h].float() - vq.float()).abs().max() <= 1.0, (i, h)
                sk = packed[blk, h, off, 128:132].view(torch.float32).item()
                sv = packed[blk, h, off, 260:264].view(torch.float32).item()
                # The conv's fp32 sum order differs from torch's by ULPs, so
                # the amax value may sit on the other side of a bf16 rounding
                # boundary: the scale is off by one bf16 ulp at most.
                assert abs(sk - ks) <= ks * 2 ** -8, (i, h)
                assert abs(sv - vs) <= vs * 2 ** -8, (i, h)
                # What the attention will read, dequantized. One code (the
                # reference amax / 127) of the two sum orders' rounding, plus
                # bf16 noise.
                step_k = 1.02 * float(ref_k[i, h].abs().amax()) / 127.0
                step_v = 1.02 * float(ref_v[i, h].abs().amax()) / 127.0
                torch.testing.assert_close(
                    key_cache[blk, off, h].float() * sk,
                    ref_k[i, h].bfloat16().float(), atol=2e-2 + step_k, rtol=2e-2,
                )
                torch.testing.assert_close(
                    value_cache[blk, off, h].float() * sv,
                    ref_v[i, h].bfloat16().float(), atol=2e-2 + step_v, rtol=2e-2,
                )
            else:
                torch.testing.assert_close(
                    key_cache[blk, off, h].float(), ref_k[i, h].bfloat16().float(),
                    atol=2e-2, rtol=2e-2,
                )
                torch.testing.assert_close(
                    value_cache[blk, off, h].float(), ref_v[i, h].bfloat16().float(),
                    atol=2e-2, rtol=2e-2,
                )
    # Every non-padding token wrote its input to its conv-state slot.
    for i in range(case["t"]):
        s = int(case["conv_slot"][i])
        if s < 0:
            continue
        blk, off = s // N_CONV, s % N_CONV
        # The conv cache row per head is [K(D) | V(D)] of that head.
        ref = torch.stack(
            [
                torch.cat(
                    [case["k"][i, h * D : (h + 1) * D], case["v"][i, h * D : (h + 1) * D]]
                )
                for h in range(H)
            ]
        ).bfloat16().float()
        torch.testing.assert_close(
            cache[blk, :, off, :].float(), ref, atol=0, rtol=0,
        )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_reference(dynamic, int8, seed):
    case = make_case(seed, dynamic)
    check(case, int8)
