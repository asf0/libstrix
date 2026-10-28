# SPDX-License-Identifier: MIT
import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


def reference(x, w_u8, scales, group_size):
    w = (w_u8.float() - 128.0).view(w_u8.size(0), -1)
    k = w.size(1)
    s = scales.float()
    if group_size:
        s = s.repeat_interleave(group_size, dim=1)[:, :k]
    else:
        s = s.reshape(-1, 1)
    return x.float() @ (w * s).t()


# 27B W8A16 shapes (qkvz 16384x5120, down 5120x17408), decode and prefill M.
@pytest.mark.parametrize("n,k", [(16384, 5120), (5120, 17408), (256, 1024)])
@pytest.mark.parametrize("m", [1, 3, 5, 64, 300])
@pytest.mark.parametrize("group_size", [0, 128])
def test_w8_paths_match_reference(n, k, m, group_size):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_u8 = torch.randint(0, 256, (n, k), device="cuda", dtype=torch.uint8)
    g = k // group_size if group_size else 1
    s = (torch.rand(n, g, device="cuda") * 0.01).to(torch.bfloat16)
    if group_size == 0:
        s = s.reshape(n)
    if m <= 5:
        out = torch.ops.strix.w8a16_skinny_gemm(w_u8, x, s, None, 40, group_size)
    else:
        sf = s.float().reshape(n) if group_size == 0 else s.float().t().contiguous()
        out = torch.ops.strix.w8a8_wmma_gemm(w_u8, x, sf)
    ref = reference(x, w_u8, s, group_size)
    # W8A8 quantizes activations to int8 per 128-group: looser tolerance.
    tol = 3e-2 if m <= 5 else 6e-2
    torch.testing.assert_close(
        out.float() / ref.abs().amax(), ref / ref.abs().amax(), rtol=0, atol=tol
    )


# W4A8 WMMA op on vLLM's skinny int4 layout (RDNAHybridW4A16: ExLlama-shuffled
# int32 words viewed as int8 [N, K/2], rows padded on gfx1151), uint4b8 g128.
@pytest.mark.parametrize("n,k", [(16384, 5120), (5120, 17408), (256, 1024)])
@pytest.mark.parametrize("m", [6, 64, 300])
@pytest.mark.parametrize("group_size", [0, 128, 32])
def test_w4a8_matches_reference(n, k, m, group_size):
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        pack_skinny_int4,
    )

    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_u4 = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    g = k // group_size if group_size else 1
    s = (torch.rand(n, g, device="cuda") * 0.05 + 0.01).to(torch.bfloat16)
    w_packed = pack_skinny_int4(w_u4)
    w = (w_u4.float() - 8.0) * (
        s.float().repeat_interleave(group_size, 1) if group_size else s.float()
    )
    ref = x.float() @ w.t()
    sf = s.float().reshape(n) if group_size == 0 else s.float().t().contiguous()
    out = torch.ops.strix.w4a8_wmma_gemm(w_packed, x, sf)
    # Activations are quantized to int8 per 128-group inside the op.
    torch.testing.assert_close(
        out.float() / ref.abs().amax(), ref / ref.abs().amax(), rtol=0, atol=6e-2
    )


# W4A16 bf16-dequant WMMA GEMM on the same packed layout; bf16 activations.
@pytest.mark.parametrize("n,k", [(16384, 5120), (5376, 21504), (256, 1024)])
@pytest.mark.parametrize("m", [6, 64, 300])
@pytest.mark.parametrize("group_size", [32, 128])
def test_w4a16_matches_reference(n, k, m, group_size):
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        pack_skinny_int4,
    )

    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_u4 = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    s = (torch.rand(n, k // group_size, device="cuda") * 0.05 + 0.01).to(torch.bfloat16)
    w = (w_u4.float() - 8.0) * s.float().repeat_interleave(group_size, 1)
    ref = x.float() @ w.t()
    out = torch.ops.strix.w4a16_wmma_gemm(
        x, pack_skinny_int4(w_u4), s, None, group_size
    )
    torch.testing.assert_close(
        out.float() / ref.abs().amax(), ref / ref.abs().amax(), rtol=0, atol=1e-2
    )


# Whole-matrix W4A16 decode (prefill M runs hipBLASLt on it): the exact bf16
# rounding of (u4 - 8) * scale, on the same packed layout as the GEMM.
@pytest.mark.parametrize("n,k", [(256, 1024), (5376, 2048)])
@pytest.mark.parametrize("group_size", [32, 128])
def test_w4a16_dequant_matches_reference(n, k, group_size):
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        pack_skinny_int4,
    )

    torch.manual_seed(0)
    w_u4 = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.int32)
    s = (torch.rand(n, k // group_size, device="cuda") * 0.05 + 0.01).to(torch.bfloat16)
    ref = ((w_u4.float() - 8.0) * s.float().repeat_interleave(group_size, 1)).to(
        torch.bfloat16
    )
    out = torch.ops.strix.w4a16_dequant(pack_skinny_int4(w_u4), s, group_size)
    assert torch.equal(out, ref)
