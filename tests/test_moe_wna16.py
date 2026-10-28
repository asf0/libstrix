# SPDX-License-Identifier: MIT
"""moe_gemm_w4a16: int4 experts in vLLM's Triton WNA16 layout, gemma-4-26B
shapes (w13 N=1408 K=2816, w2 N=2816 K=704, group 32)."""

import pytest
import strix
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

E, TOP_K, GROUP = 128, 8, 32


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
    sorted_ids += [pad] * (block_m // 2 + 3)
    expert_ids += [0]
    dev = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=dev),
        torch.tensor(expert_ids, dtype=torch.int32, device=dev),
        torch.tensor([total], dtype=torch.int32, device=dev),
    )


def pack(q: torch.Tensor) -> torch.Tensor:
    """[E, N, K] uint4 -> [E, N, K/2] uint8, low nibble = even k."""
    return (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()


def dequant(w_u8: torch.Tensor, scales: torch.Tensor, group: int) -> torch.Tensor:
    """Decode the packed bytes the long way, so a nibble-order bug fails."""
    q = torch.stack(
        [(w_u8 & 0xF).float() - 8.0, ((w_u8 >> 4) & 0xF).float() - 8.0], dim=-1
    ).reshape(*w_u8.shape[:-1], -1)
    return (q * scales.float().repeat_interleave(group, dim=-1)).to(torch.bfloat16)


def reference(a, b, topk_ids, top_k, weights):
    flat = topk_ids.flatten()
    out = torch.empty(flat.numel(), b.size(1), device=a.device)
    for e in flat.unique().tolist():
        slots = (flat == e).nonzero().flatten()
        out[slots] = a[slots // top_k].float() @ b[e].float().t()
    if weights is not None:
        out = out * weights.flatten()[:, None]
    return out


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize(
    "block_m,block_n", [(16, 64), (64, 128), (128, 128), (64, 256)]
)
@pytest.mark.parametrize(
    "gemm,activation", [("w13", "silu"), ("w13", "gelu_tanh"), ("w2", "silu")]
)
def test_matches_reference(tokens, block_m, block_n, gemm, activation, scale_dtype):
    n, k = (1408, 2816) if gemm == "w13" else (2816, 704)
    if n % block_n:
        pytest.skip(f"N={n} is not a multiple of block_n={block_n}")
    torch.manual_seed(0)
    q = torch.randint(0, 16, (E, n, k), device="cuda", dtype=torch.uint8)
    w_u8 = pack(q)
    scales = (
        torch.rand(E, n, k // GROUP, device="cuda", dtype=torch.float32) * 0.02 + 0.001
    ).to(scale_dtype)
    b = dequant(w_u8, scales, GROUP)
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
    torch.ops.strix.moe_gemm_w4a16(
        c,
        a,
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
        GROUP,
    )
    ref = reference(a, b, topk_ids, a_topk, weights)
    if gemm == "w13":
        gate, up = ref[:, : n // 2], ref[:, n // 2 :]
        act = F.silu(gate) if activation == "silu" else F.gelu(gate, approximate="tanh")
        ref = act * up
    torch.testing.assert_close(c.float(), ref, rtol=2e-2, atol=2e-2)
