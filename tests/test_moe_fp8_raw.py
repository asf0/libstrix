# SPDX-License-Identifier: MIT
"""moe_gemm_fp8_raw: float8_e4m3fn experts, fp16 activations, the scale applied to
the fp32 accumulator. The weights are exact in fp16, so the reference keeps the
dequantized matrix in fp32 (no bf16 rounding) and the tolerance is tight."""

import pytest
import strix
import torch
import torch.nn.functional as F
from test_moe_w8a16 import align

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
    return w, scales, w.float() * full


def reference(a, b, topk_ids, top_k, weights):
    flat = topk_ids.flatten()
    out = torch.empty(flat.numel(), b.size(1), device=a.device)
    for e in flat.unique().tolist():
        slots = (flat == e).nonzero().flatten()
        out[slots] = a[slots // top_k].float() @ b[e].t()
    if weights is not None:
        out = out * weights.flatten()[:, None]
    return out


@pytest.mark.parametrize("scheme", ["block", "channel"])
@pytest.mark.parametrize("tokens", [1, 5, 40])
@pytest.mark.parametrize(
    "block_m,block_n", [(16, 64), (32, 64), (64, 64), (64, 128), (128, 128)]
)
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
        a = torch.randn(tokens, k, device="cuda", dtype=torch.float16)
        a_topk, weights = TOP_K, None
        out_cols, out_dtype = n // 2, torch.float16
    else:
        a = torch.randn(tokens * TOP_K, k, device="cuda", dtype=torch.float16)
        a_topk = 1
        weights = torch.rand(tokens, TOP_K, device="cuda")
        out_cols, out_dtype = n, torch.bfloat16
    sorted_ids, expert_ids, num_post = align(topk_ids, block_m, E)
    c = torch.full((tokens * TOP_K, out_cols), float("nan"), device="cuda", dtype=out_dtype)
    torch.ops.strix.moe_gemm_fp8_raw(
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
    peak = ref.abs().amax()
    # fp16 / bf16 output rounding is the only error left: ~2^-10 / 2^-8 of the peak.
    tol = 2e-3 if out_dtype == torch.float16 else 1e-2
    torch.testing.assert_close(c.float() / peak, ref / peak, rtol=0, atol=tol)


def test_all_e4m3_codes():
    """Every finite e4m3 code through the kernel (one expert, identity-ish A)."""
    n, k = 128, 128
    codes = torch.arange(256, device="cuda", dtype=torch.uint8)
    codes = codes[(codes & 0x7F) != 0x7F]  # no NaN
    w = torch.zeros(E, n, k, dtype=torch.uint8, device="cuda")
    flat = codes.repeat((n * k) // codes.numel() + 1)[: n * k].reshape(n, k)
    w[0] = flat
    w = w.view(torch.float8_e4m3fn)
    scales = torch.ones(E, n // 128, k // 128, device="cuda")
    topk_ids = torch.zeros(1, TOP_K, dtype=torch.int64, device="cuda")
    a = torch.zeros(1, k, device="cuda", dtype=torch.float16)
    a[0, :] = torch.linspace(-1, 1, k, device="cuda")
    sorted_ids, expert_ids, num_post = align(topk_ids, 16, E)
    c = torch.empty(TOP_K, n, device="cuda", dtype=torch.bfloat16)
    torch.ops.strix.moe_gemm_fp8_raw(
        c, a.repeat(TOP_K, 1), w, scales, sorted_ids, expert_ids, num_post,
        None, 1, 16, 64, False, "silu"
    )
    ref = a.float() @ w[0].float().t()
    peak = ref.abs().amax()
    torch.testing.assert_close(c[0].float() / peak, ref[0] / peak, rtol=0, atol=1e-2)


@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("tokens", [1, 6, 70])
@pytest.mark.parametrize("block_m,block_n", [(16, 64), (64, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize("gemm", ["w13", "w2"])
def test_channel_k64(tokens, block_m, block_n, gemm, raw):
    """gemma-4-26B-A4B shapes, per-channel scales: w2 has K = 704 (not a
    multiple of 128), w13 N = 1408 with the fused gelu."""
    n, k = (1408, 2816) if gemm == "w13" else (2816, 704)
    torch.manual_seed(0)
    w, scales, b = make_weights(n, k, "channel")
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:TOP_K] for _ in range(tokens)]
    )
    adt = torch.float16 if raw else torch.bfloat16
    if gemm == "w13":
        a = torch.randn(tokens, k, device="cuda", dtype=adt)
        a_topk, weights = TOP_K, None
        out_cols = n // 2
        out_dtype = torch.float16 if raw else torch.bfloat16
    else:
        a = torch.randn(tokens * TOP_K, k, device="cuda", dtype=adt)
        a_topk = 1
        weights = torch.rand(tokens, TOP_K, device="cuda")
        out_cols, out_dtype = n, torch.bfloat16
    sorted_ids, expert_ids, num_post = align(topk_ids, block_m, E)
    c = torch.full((tokens * TOP_K, out_cols), float("nan"), device="cuda", dtype=out_dtype)
    op = torch.ops.strix.moe_gemm_fp8_raw if raw else torch.ops.strix.moe_gemm_fp8
    op(
        c, a, w, scales, sorted_ids, expert_ids, num_post,
        None if weights is None else weights.flatten(),
        a_topk, block_m, block_n, gemm == "w13", "gelu_tanh",
    )
    ref = reference(a, b, topk_ids, a_topk, weights)
    if gemm == "w13":
        gate, up = ref[:, : n // 2], ref[:, n // 2 :]
        ref = F.gelu(gate, approximate="tanh") * up
    peak = ref.abs().amax()
    tol = 2e-3 if out_dtype == torch.float16 else 1e-2
    torch.testing.assert_close(c.float() / peak, ref / peak, rtol=0, atol=tol)


@pytest.mark.parametrize("scheme", ["block", "channel"])
@pytest.mark.parametrize("tokens", [1, 40, 300])
@pytest.mark.parametrize("block_m,block_n", [(16, 64), (64, 64), (64, 128), (128, 128)])
def test_w13_bf16_intermediate(tokens, block_m, block_n, scheme):
    """A bf16 c keeps the fused silu(gate) * up bf16 (FP16_INTER_UNSAFE
    models): here scaled so that it passes the fp16 maximum, which the fp16
    output would turn into inf (Flint's reached 90,112)."""
    n, k = 1024, 2048
    torch.manual_seed(0)
    w, scales, b = make_weights(n, k, scheme)
    topk_ids = torch.stack(
        [torch.randperm(E, device="cuda")[:TOP_K] for _ in range(tokens)]
    )
    a = torch.randn(tokens, k, device="cuda", dtype=torch.float16) * 64
    sorted_ids, expert_ids, num_post = align(topk_ids, block_m, E)
    args = (sorted_ids, expert_ids, num_post, None, TOP_K, block_m, block_n, True, "silu")
    c = torch.full((tokens * TOP_K, n // 2), float("nan"), device="cuda", dtype=torch.bfloat16)
    torch.ops.strix.moe_gemm_fp8_raw(c, a, w, scales, *args)
    ref = reference(a, b, topk_ids, TOP_K, None)
    ref = F.silu(ref[:, : n // 2]) * ref[:, n // 2 :]
    assert ref.abs().amax() > 65504, "the case must leave the fp16 range"
    assert torch.isfinite(c).all()
    peak = ref.abs().amax()
    torch.testing.assert_close(c.float() / peak, ref / peak, rtol=0, atol=1e-2)
    # The fp16 output overflows on the same inputs.
    c16 = torch.empty_like(c, dtype=torch.float16)
    torch.ops.strix.moe_gemm_fp8_raw(c16, a, w, scales, *args)
    assert not torch.isfinite(c16).all()
