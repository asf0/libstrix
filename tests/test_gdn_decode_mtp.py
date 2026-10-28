# SPDX-License-Identifier: MIT
"""strix port of vLLM's fused GDN MTP decode kernel vs a torch reference."""

import pytest
import strix
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

D = 128


def reference(
    mixed, a, b, a_log, dt_bias, si, cu, acc, state, gate, norm_w, scale, eps
):
    state = state.clone()
    hv, h = state.size(1), (mixed.size(1) - state.size(1) * D) // (2 * D)
    out = torch.zeros(mixed.size(0), hv, D, device=mixed.device)
    for r in range(si.size(0)):
        bos, eos = int(cu[r]), int(cu[r + 1])
        src = int(si[r, int(acc[r]) - 1]) if 0 < int(acc[r]) <= si.size(1) else 0
        if eos <= bos or src <= 0:
            continue
        for v in range(hv):
            kh = v // (hv // h)
            hs = state[src, v].float().clone()  # [V, K]
            for t in range(eos - bos):
                tok = bos + t
                row = mixed[tok].float()
                q = (
                    row[kh * D : (kh + 1) * D]
                    / (row[kh * D : (kh + 1) * D].pow(2).sum() + 1e-6).sqrt()
                    * scale
                )
                kk = row[h * D + kh * D : h * D + (kh + 1) * D]
                kk = kk / (kk.pow(2).sum() + 1e-6).sqrt()
                vv = row[2 * h * D + v * D : 2 * h * D + (v + 1) * D]
                g = -a_log[v].exp() * F.softplus(a[tok, v].float() + dt_bias[v].float())
                beta = torch.sigmoid(b[tok, v].float())
                hs = hs * g.exp()
                delta = (vv - hs @ kk) * beta
                hs = hs + torch.outer(delta, kk)
                o = hs @ q
                dst = int(si[r, t])
                if dst > 0:
                    state[dst, v] = hs.to(state.dtype)
                o = o.bfloat16().float()
                rstd = (o.pow(2).mean() + eps).rsqrt()
                gi = gate[tok, v].float()
                out[tok, v] = o * rstd * norm_w.float() * (gi * torch.sigmoid(gi))
    return out, state


@pytest.mark.parametrize("h,hv", [(16, 48), (16, 32)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_matches_reference(h, hv, state_dtype):
    torch.manual_seed(0)
    slots = 12
    lens = [3, 2, 3]
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda"
    )
    n_tok = sum(lens)
    mixed = torch.randn(n_tok, (2 * h + hv) * D, device="cuda").bfloat16()
    ab = torch.randn(n_tok, 2 * hv, device="cuda").bfloat16()
    a, b = ab[:, :hv], ab[:, hv:]
    a_log = torch.rand(hv, device="cuda")
    dt_bias = torch.randn(hv, device="cuda")
    si = torch.tensor(
        [[1, 2, 3], [4, 5, 0], [0, 0, 0]], dtype=torch.int32, device="cuda"
    )
    si[2] = torch.tensor([6, 7, 8])
    acc = torch.tensor([2, 1, 3], dtype=torch.int32, device="cuda")
    state = (torch.randn(slots, hv, D, D, device="cuda") * 0.1).to(state_dtype)
    gate = torch.randn(n_tok, hv, D, device="cuda").bfloat16()
    norm_w = torch.rand(D, device="cuda")
    scale, eps = D**-0.5, 1e-6
    ref_out, ref_state = reference(
        mixed, a, b, a_log, dt_bias, si, cu, acc, state, gate, norm_w, scale, eps
    )
    out = torch.empty(n_tok, hv, D, device="cuda").bfloat16()
    st = state.clone()
    torch.ops.strix.gdn_decode_mtp_bf16(
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        si,
        cu,
        acc,
        st,
        gate,
        norm_w,
        out,
        scale,
        eps,
        "silu",
    )
    torch.testing.assert_close(out.float(), ref_out, atol=3e-2, rtol=3e-2)
    tol = 1e-3 if state_dtype == torch.float32 else 2e-2
    torch.testing.assert_close(st.float(), ref_state.float(), atol=tol, rtol=tol)
