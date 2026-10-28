# SPDX-License-Identifier: MIT
import pytest
import strix
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


def recurrent_reference(q, k, v, g, beta, state, cu, scale):
    """Token-by-token gated delta rule in fp32; state is [N, H, V, K]."""
    q, k, v = q.float(), k.float(), v.float()
    hv, hk = v.size(1), q.size(1)
    rep = hv // hk
    o = torch.empty_like(v)
    final = state.clone().float()
    for n in range(cu.numel() - 1):
        s = final[n].transpose(-1, -2).clone()  # [H, K, V]
        for t in range(int(cu[n]), int(cu[n + 1])):
            kt = k[t].repeat_interleave(rep, 0)  # [H, K]
            qt = q[t].repeat_interleave(rep, 0)
            s = s * g[t].exp()[:, None, None]
            vnew = (v[t] - torch.einsum("hkv,hk->hv", s, kt)) * beta[t][:, None]
            s = s + kt[:, :, None] * vnew[:, None, :]
            o[t] = scale * torch.einsum("hkv,hk->hv", s, qt)
        final[n] = s.transpose(-1, -2)
    return o, final


def make_inputs(lens, hk, hv, seed=0, state_dtype=torch.float32):
    torch.manual_seed(seed)
    t = sum(lens)
    q = F.normalize(torch.randn(t, hk, 128, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(t, hk, 128, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(t, hv, 128, device="cuda").bfloat16()
    g = F.logsigmoid(torch.randn(t, hv, device="cuda") + 3)
    beta = torch.rand(t, hv, device="cuda").sigmoid()
    state = (torch.randn(len(lens), hv, 128, 128, device="cuda") * 0.1).to(state_dtype)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda")
    return q, k, v, g, beta, state, cu


# Qwen3.8-27B: Hk=16, Hv=48; Qwen3.6-35B-A3B: Hk=16, Hv=32. Ragged lengths
# cover partial last chunks, a 1-token sequence and multi-chunk carries.
@pytest.mark.parametrize("hk,hv", [(16, 48), (16, 32)])
@pytest.mark.parametrize("lens", [[64], [1], [200], [37, 130, 64, 5]])
# bf16 state is what --mamba-ssm-cache-dtype bfloat16 sets; the carry is read
# and written in that dtype, so the reference carries in fp32 from a bf16 start
# and the result is compared at bf16 tolerance.
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_matches_recurrence(hk, hv, lens, state_dtype):
    q, k, v, g, beta, state, cu = make_inputs(lens, hk, hv, state_dtype=state_dtype)
    scale = 128**-0.5
    o, final = strix.gdn_chunk_fwd(q, k, v, g, beta, state, cu, scale)
    ref_o, ref_final = recurrent_reference(q, k, v, g, beta, state.float(), cu, scale)
    assert final.dtype == state_dtype
    tol = 3e-2 if state_dtype == torch.float32 else 5e-2
    torch.testing.assert_close(o.float(), ref_o, rtol=tol, atol=tol)
    torch.testing.assert_close(final.float(), ref_final, rtol=tol, atol=tol)


def test_strided_qkv_views():
    """vLLM passes q/k/v as views into the fused projection."""
    lens, hk, hv = [100], 16, 32
    q, k, v, g, beta, state, cu = make_inputs(lens, hk, hv)
    packed = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=1).contiguous()
    qv = packed[:, : hk * 128].view(-1, hk, 128)
    kv = packed[:, hk * 128 : 2 * hk * 128].view(-1, hk, 128)
    vv = packed[:, 2 * hk * 128 :].view(-1, hv, 128)
    o, final = strix.gdn_chunk_fwd(qv, kv, vv, g, beta, state, cu)
    ref_o, ref_final = strix.gdn_chunk_fwd(q, k, v, g, beta, state, cu)
    torch.testing.assert_close(o, ref_o, rtol=0, atol=0)
    torch.testing.assert_close(final, ref_final, rtol=0, atol=0)
