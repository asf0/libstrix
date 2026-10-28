# SPDX-License-Identifier: MIT
import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


def align(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    """moe_align_block_size's contract: ids grouped by expert, each expert's run
    padded to block_m with the invalid id topk_ids.numel()."""
    flat = topk_ids.flatten().cpu()
    pad = flat.numel()
    sorted_ids, expert_ids = [], []
    for e in range(num_experts):
        ids = (flat == e).nonzero().flatten().tolist()
        if not ids:
            continue
        ids += [pad] * (-len(ids) % block_m)
        sorted_ids += ids
        expert_ids += [e] * (len(ids) // block_m)
    total = len(sorted_ids)
    # Like vLLM: a spare, partially sized block past num_post_padded.
    sorted_ids += [pad] * (block_m // 2 + 3)
    expert_ids += [0]
    dev = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
        torch.tensor(expert_ids, dtype=torch.int32, device=dev),
        torch.tensor([total], dtype=torch.int32, device=dev),
    )


def reference(a, b, topk_ids, top_k, weights):
    flat = topk_ids.flatten()
    out = torch.empty(flat.numel(), b.size(1), device=a.device)
    for e in flat.unique().tolist():
        slots = (flat == e).nonzero().flatten()
        out[slots] = a[slots // top_k].float() @ b[e].float().t()
    if weights is not None:
        out = out * weights.flatten()[:, None]
    return out


# 35B-A3B: E=256, top-8; w13 N=1024 K=2048, w2 N=2048 K=512.
@pytest.mark.parametrize("tokens", [1, 5, 64, 700])
@pytest.mark.parametrize(
    "block_m,block_n",
    [(16, 64), (32, 64), (64, 64), (64, 128), (128, 64), (128, 128), (64, 256)],
)
@pytest.mark.parametrize("gemm", ["w13", "w2"])
def test_matches_reference(tokens, block_m, block_n, gemm):
    torch.manual_seed(0)
    e, top_k = 256, 8
    n, k = (1024, 2048) if gemm == "w13" else (2048, 512)
    b = torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    topk_ids = torch.stack(
        [torch.randperm(e, device="cuda")[:top_k] for _ in range(tokens)]
    )
    if gemm == "w13":
        a = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
        a_top_k, weights = top_k, None
    else:
        a = torch.randn(tokens * top_k, k, device="cuda", dtype=torch.bfloat16)
        a_top_k = 1
        weights = torch.rand(tokens, top_k, device="cuda")
    sorted_ids, expert_ids, num_post = align(topk_ids, block_m, e)
    c = torch.full(
        (tokens * top_k, n), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    torch.ops.strix.moe_gemm_bf16(
        c,
        a,
        b,
        sorted_ids,
        expert_ids,
        num_post,
        None if weights is None else weights.flatten(),
        a_top_k,
        block_m,
        block_n,
    )
    ref = reference(a, b, topk_ids, a_top_k, weights)
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)


# silu_mul fuses vLLM's silu_and_mul into w13 (rows [gate; up]).
# 35B-A3B (E256 top-8, N=1024) and LFM2.5-24B-A2B (E64 top-4, N=3072) shapes.
@pytest.mark.parametrize("tokens", [1, 5, 700])
@pytest.mark.parametrize(
    "block_m,block_n", [(16, 64), (32, 64), (64, 128), (128, 128), (64, 256)]
)
@pytest.mark.parametrize("e,top_k,n", [(256, 8, 1024), (64, 4, 3072)])
def test_silu_mul_matches_reference(tokens, block_m, block_n, e, top_k, n):
    torch.manual_seed(0)
    k = 2048
    b = torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    topk_ids = torch.stack(
        [torch.randperm(e, device="cuda")[:top_k] for _ in range(tokens)]
    )
    a = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    sorted_ids, expert_ids, num_post = align(topk_ids, block_m, e)
    c = torch.full(
        (tokens * top_k, n // 2), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    torch.ops.strix.moe_gemm_bf16(
        c,
        a,
        b,
        sorted_ids,
        expert_ids,
        num_post,
        None,
        top_k,
        block_m,
        block_n,
        True,
    )
    h = reference(a, b, topk_ids, top_k, None)
    ref = torch.nn.functional.silu(h[:, : n // 2]) * h[:, n // 2 :]
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)


# gelu_tanh_mul is the same epilogue with GELU-tanh in place of SiLU, for
# gemma-4's MoE (hidden_activation gelu_pytorch_tanh). Same [gate; up] rows.
# gemma-4-26B-A4B: E128 top-8, N=1408 (2 x moe_intermediate 704), K=2816;
# 35B-A3B: E256 top-8, N=1024, K=2048.
@pytest.mark.parametrize("tokens", [1, 5, 700])
@pytest.mark.parametrize(
    "block_m,block_n",
    [(16, 64), (32, 64), (64, 64), (64, 128), (128, 64), (128, 128), (64, 256)],
)
@pytest.mark.parametrize("e,top_k,n,k", [(128, 8, 1408, 2816), (256, 8, 1024, 2048)])
def test_gelu_tanh_mul_matches_reference(tokens, block_m, block_n, e, top_k, n, k):
    if n % block_n:
        # The op needs N % block_n == 0; 1408 has no 256-column tile.
        pytest.skip(f"N={n} is not a multiple of block_n={block_n}")
    torch.manual_seed(0)
    b = torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    topk_ids = torch.stack(
        [torch.randperm(e, device="cuda")[:top_k] for _ in range(tokens)]
    )
    a = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    sorted_ids, expert_ids, num_post = align(topk_ids, block_m, e)
    c = torch.full(
        (tokens * top_k, n // 2), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    torch.ops.strix.moe_gemm_bf16(
        c,
        a,
        b,
        sorted_ids,
        expert_ids,
        num_post,
        None,
        top_k,
        block_m,
        block_n,
        True,
        "gelu_tanh",
    )
    h = reference(a, b, topk_ids, top_k, None)
    ref = torch.nn.functional.gelu(h[:, : n // 2], approximate="tanh") * h[:, n // 2 :]
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)
