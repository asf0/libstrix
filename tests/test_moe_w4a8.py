# SPDX-License-Identifier: MIT
"""moe_act_quant_i8 + moe_gemm_w4a8: int8 activations x int4 experts (vLLM's
Triton WNA16 layout, group 128) at Qwen3.8-Flash-Next shapes (top-10, w13
N=1280 K=2560, w2 N=2560 K=640; 64 of the 512 experts)."""

import pytest
import strix
import torch
import torch.nn.functional as F
from test_moe_wna16 import align, pack, reference

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

E, TOP_K, GROUP = 64, 10, 128


def act_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    sa = torch.empty(x.size(0), x.size(1) // GROUP, device=x.device, dtype=torch.float32)
    torch.ops.strix.moe_act_quant_i8(x, q, sa)
    return q, sa


def test_act_quant_matches_reference():
    torch.manual_seed(0)
    x = torch.randn(37, 640, device="cuda", dtype=torch.bfloat16) * 3
    x[5, 128:256] = 0  # an all-zero group takes scale 1
    q, sa = act_quant(x)
    g = x.float().view(37, -1, GROUP)
    s = g.abs().amax(-1) / 127
    s = torch.where(s > 0, s, torch.ones_like(s))
    torch.testing.assert_close(sa, s, rtol=1e-6, atol=0)
    q_ref = torch.clamp(torch.round(g / s[..., None]), -127, 127).view(37, 640)
    # x * (1/s) and x / s can round a .5 tie differently: off by one at most.
    diff = (q.float() - q_ref).abs()
    assert diff.max() <= 1 and (diff > 0).float().mean() < 2e-3


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 37, 300])
@pytest.mark.parametrize("block_m,block_n", [(64, 256), (128, 128), (64, 128)])
@pytest.mark.parametrize(
    "gemm,activation", [("w13", "silu"), ("w13", "gelu_tanh"), ("w2", "silu")]
)
def test_matches_reference(tokens, block_m, block_n, gemm, activation, scale_dtype):
    n, k = (1280, 2560) if gemm == "w13" else (2560, 640)
    torch.manual_seed(0)
    q4 = torch.randint(0, 16, (E, n, k), device="cuda", dtype=torch.uint8)
    w_u8 = pack(q4)
    scales = (torch.rand(E, n, k // GROUP, device="cuda") * 0.02 + 0.001).to(scale_dtype)
    b = (q4.float() - 8) * scales.float().repeat_interleave(GROUP, dim=-1)
    topk_ids = torch.stack([torch.randperm(E, device="cuda")[:TOP_K] for _ in range(tokens)])
    if gemm == "w13":
        a = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
        a_topk, weights, out_cols = TOP_K, None, n // 2
    else:
        a = torch.randn(tokens * TOP_K, k, device="cuda", dtype=torch.bfloat16)
        a_topk, weights, out_cols = 1, torch.rand(tokens, TOP_K, device="cuda"), n
    qa, sa = act_quant(a)
    sorted_ids, expert_ids, num_post = align(topk_ids, block_m, E)
    c = torch.full((tokens * TOP_K, out_cols), float("nan"), device="cuda", dtype=torch.bfloat16)
    torch.ops.strix.moe_gemm_w4a8(
        c,
        qa,
        sa,
        w_u8,
        scales,
        sorted_ids,
        expert_ids,
        num_post,
        None if weights is None else weights.flatten(),
        a_topk,
        block_m,
        block_n,
        gemm == "w13",
        activation,
    )
    # The kernel's own math: the dequantized int8 activations, exact weights.
    a_deq = qa.float() * sa.repeat_interleave(GROUP, dim=-1)
    ref = reference(a_deq, b, topk_ids, a_topk, weights)
    if gemm == "w13":
        gate, up = ref[:, : n // 2], ref[:, n // 2 :]
        act = F.silu(gate) if activation == "silu" else F.gelu(gate, approximate="tanh")
        ref = act * up
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)
