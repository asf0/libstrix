# SPDX-License-Identifier: MIT
"""strix's ReplaySSM verify kernel vs vLLM's Triton one and a torch reference."""

import pytest
import strix
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

D = 128
DEV = "cuda"


def inputs(h, hv, lens, state_dtype, seed=0):
    torch.manual_seed(seed)
    t = sum(lens)
    q = torch.randn(1, t, h, D, device=DEV).bfloat16()
    k = torch.randn(1, t, h, D, device=DEV).bfloat16()
    v = torch.randn(1, t, hv, D, device=DEV).bfloat16()
    ab = torch.randn(t, 2 * hv, device=DEV).bfloat16()
    n = len(lens)
    state = (torch.randn(n + 3, hv, D, D, device=DEV) * 0.1).to(state_dtype)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device=DEV)
    # Block 0 is vLLM's null block: request 1 has no checkpoint when n > 1.
    idx = torch.arange(1, n + 1, device=DEV)
    if n > 1:
        idx[1] = 0
    return q, k, v, ab[:, :hv], ab[:, hv:], torch.rand(hv, device=DEV), torch.randn(
        hv, device=DEV
    ), state, cu, idx


def records(hv, h, blocks, s):
    # Filled with a value no kernel writes.
    return (
        torch.full((blocks, hv, s, D), -7.0, device=DEV),
        torch.full((blocks, h, s, D), -7.0, device=DEV),
        torch.full((blocks, hv, s), -7.0, device=DEV),
    )


def reference(q, k, v, a, b, a_log, dt_bias, state, cu, idx, s):
    h, hv = q.size(2), v.size(2)
    corr, keys, decays = records(hv, h, state.size(0), s)
    out = torch.zeros_like(v).float()
    for r in range(idx.numel()):
        bos, eos, slot = int(cu[r]), int(cu[r + 1]), int(idx[r])
        if slot <= 0:
            continue
        for vh in range(hv):
            kh = vh // (hv // h)
            hs = state[slot, vh].float().clone()
            for t in range(eos - bos):
                tok = bos + t
                qq, kk = q[0, tok, kh].float(), k[0, tok, kh].float()
                qq = qq * (qq.pow(2).sum() + 1e-6).rsqrt() * D**-0.5
                kk = kk * (kk.pow(2).sum() + 1e-6).rsqrt()
                g = -a_log[vh].exp() * F.softplus(a[tok, vh].float() + dt_bias[vh])
                beta = torch.sigmoid(b[tok, vh].float())
                hs = hs * g.exp()
                delta = (v[0, tok, vh].float() - hs @ kk) * beta
                hs = hs + torch.outer(delta, kk)
                out[0, tok, vh] = hs @ qq
                corr[slot, vh, t], decays[slot, vh, t] = delta, g.exp()
                if vh % (hv // h) == 0:
                    keys[slot, kh, t] = kk
    return out, corr, keys, decays


CASES = [(16, 32), (16, 48), (8, 32), (16, 16), (4, 32)]  # HV / H 2, 3, 4, 1, 8


@pytest.mark.parametrize("h,hv", CASES)
@pytest.mark.parametrize("lens", [[8], [3, 2, 8], [1, 5]])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_matches_reference(h, hv, lens, state_dtype):
    q, k, v, a, b, a_log, dt, state, cu, idx = inputs(h, hv, lens, state_dtype)
    s = 8
    ref_out, ref_c, ref_k, ref_d = reference(q, k, v, a, b, a_log, dt, state, cu, idx, s)
    corr, keys, decays = records(hv, h, state.size(0), s)
    out = torch.full_like(v, float("nan"))
    before = state.clone()
    torch.ops.strix.gdn_recoverssm_verify_bf16(
        q[0], k[0], v[0], a, b, a_log, dt, state, corr, keys, decays,
        cu.to(torch.int32), idx.to(torch.int64), out[0], D**-0.5, 0,
    )
    assert torch.equal(state, before)  # the checkpoint is read only
    for r in range(idx.numel()):
        if int(idx[r]) <= 0:
            assert not out[0, int(cu[r]) : int(cu[r + 1])].any()
    torch.testing.assert_close(out.float(), ref_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(corr, ref_c, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(keys, ref_k, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(decays, ref_d, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("h,hv", CASES)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_matches_triton(h, hv, state_dtype):
    gdn = pytest.importorskip("vllm.model_executor.layers.mamba.ops.gdn_recoverssm")
    lens = [4, 8, 2]
    q, k, v, a, b, a_log, dt, state, cu, idx = inputs(h, hv, lens, state_dtype, 1)
    s = 8
    cu32 = cu.to(torch.int32)
    tri = records(hv, h, state.size(0), s)
    ours = records(hv, h, state.size(0), s)
    want = gdn.gdn_recoverssm_verify(
        q, k, v, a, b, a_log, dt, state, *tri, cu32, idx.to(torch.int64), s
    )
    out = torch.full_like(v, float("nan"))
    torch.ops.strix.gdn_recoverssm_verify_bf16(
        q[0], k[0], v[0], a, b, a_log, dt, state, *ours, cu32, idx.to(torch.int64),
        out[0], D**-0.5, 0,
    )
    torch.testing.assert_close(out, want, atol=3e-2, rtol=3e-2)
    for got, ref in zip(ours, tri):
        torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)


def test_wrapper_routes_and_falls_back():
    from strix_vllm.gdn import _wrap_recoverssm_verify

    calls = []

    def triton_verify(*args):
        calls.append(args)
        return torch.zeros_like(args[2])

    verify = _wrap_recoverssm_verify(triton_verify)
    h, hv, lens, s = 16, 32, [3, 8], 8
    q, k, v, a, b, a_log, dt, state, cu, idx = inputs(h, hv, lens, torch.float32)
    corr, keys, decays = records(hv, h, state.size(0), s)
    # vLLM passes an int32 column of its state-index table and int32 offsets.
    table = torch.stack([idx, idx], 1).to(torch.int32)
    out = verify(q, k, v, a, b, a_log, dt, state, corr, keys, decays,
                 cu.to(torch.int32), table[:, 0], s)
    assert not calls
    ref, *_ = reference(q, k, v, a, b, a_log, dt, state, cu, idx, s)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)
    # A window past the kernel's 8 tokens, and records of another length, go
    # to the Triton kernel.
    corr9, keys9, decays9 = records(hv, h, state.size(0), 9)
    verify(q, k, v, a, b, a_log, dt, state, corr9, keys9, decays9,
           cu.to(torch.int32), idx, 9)
    verify(q, k, v, a, b, a_log, dt, state, corr, keys, decays,
           cu.to(torch.int32), idx, 7)
    assert len(calls) == 2
