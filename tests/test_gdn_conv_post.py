# SPDX-License-Identifier: MIT
"""GDN prefill front end: causal conv (with conv-state cache) and post-conv
split/L2 norm/gating, against an fp32 torch reference and vLLM's Triton ops."""

import pytest
import strix
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

# (hk, hv): Qwen3.8-27B, Qwen3.6-35B-A3B
SHAPES = [(16, 48), (16, 32)]


def conv_reference(x, w, bias, states, cu, idx, has_init):
    out = torch.empty_like(x, dtype=torch.float32)
    new_states = states.clone()
    for n in range(cu.numel() - 1):
        s, e, ci = int(cu[n]), int(cu[n + 1]), int(idx[n])
        valid = ci not in (-1, 0)
        init = (
            states[ci].float().t()
            if valid and has_init is not None and bool(has_init[n])
            else x.new_zeros(3, x.size(1), dtype=torch.float32)
        )
        full = torch.cat([init, x[s:e].float()])
        for j in range(e - s):
            acc = (full[j : j + 4] * w.float().t()).sum(0)
            out[s + j] = F.silu(acc + (bias.float() if bias is not None else 0))
        if valid:
            new_states[ci] = full[-3:].t().to(states.dtype)
    return out, new_states


def make_conv_case(lens, dim, transposed_state, seed=0):
    torch.manual_seed(seed)
    t = sum(lens)
    # A slice of a wider projection, like vLLM's mixed_qkv.
    x = torch.randn(t, dim + 64, device="cuda").bfloat16()[:, :dim]
    w = (torch.randn(dim, 4, device="cuda") * 0.5).bfloat16()
    lines = len(lens) + 3
    if transposed_state:  # vLLM's layout: [lines, 3, dim] viewed as [lines, dim, 3]
        states = torch.randn(lines, 3, dim, device="cuda").bfloat16().transpose(1, 2)
    else:
        states = torch.randn(lines, dim, 3, device="cuda").bfloat16()
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    idx = torch.randperm(lines - 1)[: len(lens)].to(torch.int32) + 1
    if len(lens) > 2:
        idx[1] = 0  # null block: no state read or write
    has_init = torch.rand(len(lens)) > 0.4
    return x, w, states, cu.cuda(), idx.cuda(), has_init.cuda()


@pytest.mark.parametrize(
    "lens",
    [[1], [2], [3], [37], [64, 1, 2, 300], [5, 0, 9], [1000, 17, 3, 4, 2048]],
)
@pytest.mark.parametrize("transposed_state", [False, True])
@pytest.mark.parametrize("hk,hv", SHAPES[:1])
def test_conv_matches_reference(lens, transposed_state, hk, hv):
    dim = (2 * hk + hv) * 128
    x, w, states, cu, idx, has_init = make_conv_case(lens, dim, transposed_state)
    ref_out, ref_states = conv_reference(x, w, None, states, cu, idx, has_init)
    out = strix.gdn_causal_conv1d(x, w, None, states, cu, idx, has_init)
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(states, ref_states, atol=0, rtol=0)


def test_conv_matches_vllm():
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    lens, dim = [500, 3, 1, 1200], (2 * 16 + 32) * 128
    x, w, states, cu, idx, has_init = make_conv_case(lens, dim, True, seed=1)
    states_ref = states.clone()
    ref = causal_conv1d_fn(
        x.t(),
        w,
        None,
        activation="silu",
        conv_states=states_ref,
        has_initial_state=has_init,
        cache_indices=idx,
        query_start_loc=cu,
    ).t()
    out = strix.gdn_causal_conv1d(x, w, None, states, cu, idx, has_init)
    keep = torch.ones(x.size(0), dtype=torch.bool)
    keep[int(cu[1]) : int(cu[2])] = False  # null-block sequence: Triton skips it
    torch.testing.assert_close(
        out[keep].float(), ref[keep].float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(states, states_ref, atol=0, rtol=0)


@pytest.mark.parametrize("hk,hv", SHAPES)
@pytest.mark.parametrize("t", [1, 77, 4096])
def test_post_conv_matches_vllm(hk, hv, t):
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
        fused_post_conv_prep,
    )

    torch.manual_seed(0)
    dim = (2 * hk + hv) * 128
    conv = torch.randn(t, dim + 128, device="cuda").bfloat16()[:, :dim]
    ab = torch.randn(t, 2 * hv, device="cuda").bfloat16()
    a, b = ab[:, :hv], ab[:, hv:]
    a_log = torch.rand(hv, device="cuda") * 2 - 1
    dt_bias = torch.randn(hv, device="cuda").bfloat16()
    ref = fused_post_conv_prep(
        conv.contiguous(), a, b, a_log, dt_bias, hk, 128, 128, apply_l2norm=True
    )
    got = strix.gdn_post_conv(conv, a, b, a_log, dt_bias, hk)
    for name, r, o in zip("qkvgb", ref, got):
        assert o.shape == r.shape and o.dtype == r.dtype, name
        torch.testing.assert_close(o.float(), r.float(), atol=1e-2, rtol=1e-2)
