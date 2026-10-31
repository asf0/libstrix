# SPDX-License-Identifier: MIT
"""moe_gemm_fp8: float8_e4m3fn experts with 128x128 block scales (fp32
[E, N/128, K/128], vLLM's block-quantized FP8 MoE layout) or one scale per
output channel, at Qwen3.6-35B-A3B shapes (w13 N=1024 K=2048, w2 N=2048 K=512)."""

import pytest
import strix
import torch
import torch.nn.functional as F
from test_moe_w8a16 import align, reference

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

E, TOP_K = 256, 8


def make_weights(n, k, scheme, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((E, n, k), device="cuda", generator=g) * 40).clamp(-448, 448)
    w = w.to(torch.float8_e4m3fn)
    if scheme == "block":
        scales = torch.rand((E, n // 128, k // 128), device="cuda", generator=g)
        scales = scales * 0.02 + 0.001
        full = scales.repeat_interleave(128, 1).repeat_interleave(128, 2)
    else:
        scales = torch.rand((E, n), device="cuda", generator=g) * 0.02 + 0.001
        full = scales.reshape(E, n, 1)
    # Dequantize the long way, so a scale-index bug fails.
    return w, scales, (w.float() * full).to(torch.bfloat16)


@pytest.mark.parametrize("scheme", ["block", "channel"])
@pytest.mark.parametrize("tokens", [1, 5, 40])
@pytest.mark.parametrize("block_m,block_n", [(16, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize(
    "gemm,activation", [("w13", "silu"), ("w13", "gelu_tanh"), ("w2", "silu")]
)
def test_matches_reference(tokens, block_m, block_n, gemm, activation, scheme):
    n, k = (1024, 2048) if gemm == "w13" else (2048, 512)
    torch.manual_seed(0)
    w, scales, b = make_weights(n, k, scheme)
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:TOP_K] for _ in range(tokens)]
    )
    if gemm == "w13":
        a = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
        a_topk, weights = TOP_K, None
        out_cols = n // 2
    else:
        a = torch.randn(tokens * TOP_K, k, device="cuda", dtype=torch.bfloat16)
        a_topk = 1
        weights = torch.rand(tokens, TOP_K, device="cuda")
        out_cols = n
    sorted_ids, expert_ids, num_post = align(topk_ids, block_m, E)
    c = torch.full(
        (tokens * TOP_K, out_cols), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    torch.ops.strix.moe_gemm_fp8(
        c,
        a,
        w,
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
    ref = reference(a, b, topk_ids, a_topk, weights)
    if gemm == "w13":
        gate, up = ref[:, : n // 2], ref[:, n // 2 :]
        act = F.silu(gate) if activation == "silu" else F.gelu(gate, approximate="tanh")
        ref = act * up
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)


def test_channel_scale_shapes_and_dtypes_agree():
    n, k = 1024, 2048
    w, scales, _ = make_weights(n, k, "channel")
    topk_ids = torch.stack([torch.randperm(E, device="cuda")[:TOP_K] for _ in range(3)])
    a = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    sorted_ids, expert_ids, num_post = align(topk_ids, 16, E)
    outs = []
    for s in (scales, scales.unsqueeze(-1), scales.to(torch.bfloat16)):
        c = torch.empty((3 * TOP_K, n // 2), device="cuda", dtype=torch.bfloat16)
        torch.ops.strix.moe_gemm_fp8(
            c, a, w, s, sorted_ids, expert_ids, num_post, None, TOP_K, 16, 64, True, "silu"
        )
        outs.append(c)
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)
    # bf16 scales round each channel by up to 2^-9; compare to the output's peak.
    peak = outs[0].float().abs().amax()
    torch.testing.assert_close(
        outs[2].float() / peak, outs[0].float() / peak, rtol=0, atol=1e-2
    )
