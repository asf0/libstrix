# SPDX-License-Identifier: MIT
"""Native MXFP4 (e2m1 + e8m0) W4A16 GEMM against Quark's own dequantizer.

The reference dequantization is quark.torch.kernel.mx.dq_mxfp4, i.e. the
function AMD's Quark ships and the one that produced these checkpoints, so the
LUT and the nibble order verified here are the checkpoint's, not an assumption.
"""

import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")

MX_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dq(w_u8, s_u8):
    from quark.torch.kernel.mx import dq_mxfp4

    return dq_mxfp4(w_u8, s_u8, torch.bfloat16)


def reference(x, w_u8, s_u8):
    return x.float() @ dq(w_u8, s_u8).float().t()


def torch_dequant(w_u8, s_u8, low_first=True):
    """Byte-exact expectation for one layout: low nibble = even k (or not)."""
    lut = torch.tensor(MX_VALUES, device=w_u8.device, dtype=torch.float32)
    w = w_u8.to(torch.int32)
    even = (w >> (0 if low_first else 4)) & 15
    odd = (w >> (4 if low_first else 0)) & 15

    def value(nib):
        v = lut[(nib & 7).long()]
        return torch.where((nib & 8) != 0, -v, v)

    out = torch.empty(
        w_u8.size(0), w_u8.size(1) * 2, device=w_u8.device, dtype=torch.float32
    )
    out[:, 0::2] = value(even)
    out[:, 1::2] = value(odd)
    scale = torch.pow(2.0, s_u8.float() - 127.0)
    return out * scale.repeat_interleave(32, dim=1)


def test_nibble_order_matches_quark():
    """Even k is the low nibble; the odd-nibble-first layout must differ."""
    torch.manual_seed(0)
    w = torch.randint(0, 256, (4, 64), dtype=torch.uint8, device="cuda")
    s = torch.full((4, 4), 127, dtype=torch.uint8, device="cuda")
    ref = dq(w, s).float()
    torch.testing.assert_close(torch_dequant(w, s, low_first=True), ref)
    assert not torch.equal(torch_dequant(w, s, low_first=False), ref)


@pytest.mark.parametrize("low_first", [True, False])
def test_kernel_dequant_matches_quark(low_first):
    """One-hot activations make the GEMM the dequantizer: C.T == dq(W)."""
    torch.manual_seed(0)
    n, k = 256, 128
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(120, 135, (n, k // 32), dtype=torch.uint8, device="cuda")
    if not low_first:
        # Swap the nibbles of every byte so the kernel's (low-first) decode
        # sees the reversed order this test asks its reference for.
        w = ((w & 15) << 4) | (w >> 4)
    x = torch.zeros(k, k, dtype=torch.bfloat16, device="cuda")
    x.fill_diagonal_(1.0)
    out = torch.ops.strix.mxfp4_wmma_gemm(w, x, s, None)
    torch.testing.assert_close(out.t().float(), dq(w, s).float())


def test_zero_block_scale_is_zero():
    """An e8m0 byte of 0 is the zero block: this kernel writes exact zeros.

    (Quark's HIP dequantizer instead leaves 2^-127-scaled bf16 subnormals
    there - 1.8e-38 - so the reference cannot be used for this check.)
    """
    torch.manual_seed(0)
    n, k = 64, 128
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    s = torch.zeros((n, k // 32), dtype=torch.uint8, device="cuda")
    s[:, 1] = 127
    x0 = torch.zeros(8, k, dtype=torch.bfloat16, device="cuda")
    x0[:, :32] = 1.0  # only the zero-scale block
    x1 = torch.zeros(8, k, dtype=torch.bfloat16, device="cuda")
    x1[:, 32:] = 1.0  # only the scaled block
    assert torch.count_nonzero(torch.ops.strix.mxfp4_wmma_gemm(w, x0, s, None)) == 0
    assert torch.count_nonzero(torch.ops.strix.mxfp4_wmma_gemm(w, x1, s, None)) > 0
    x = torch.ones(8, k, dtype=torch.bfloat16, device="cuda")
    out = torch.ops.strix.mxfp4_wmma_gemm(w, x, s, None)
    ref = reference(x, w, s)
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=1e-2 * ref.abs().amax())


@pytest.mark.parametrize("n,k", [(256, 1024), (2048, 2688), (5376, 10752 // 4)])
@pytest.mark.parametrize("m", [1, 5, 32, 513])
def test_gemm_matches_quark_dequant(n, k, m):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(118, 134, (n, k // 32), dtype=torch.uint8, device="cuda")
    out = torch.ops.strix.mxfp4_wmma_gemm(w, x, s, None)
    ref = reference(x, w, s)
    # bf16 operands, fp32 accumulate: the only error is the bf16 rounding of
    # the activations (the weights dequantize exactly).
    torch.testing.assert_close(
        out.float() / ref.abs().amax(), ref / ref.abs().amax(), rtol=0, atol=1e-2
    )


@pytest.mark.parametrize("n,k", [(256, 1024), (2048, 2688), (96, 5376)])
def test_dequant_is_byte_exact(n, k):
    # The whole-matrix decode the plugin feeds hipBLASLt at prefill M: the
    # same values as the reference dequant, exactly (e2m1 times a power of
    # two is representable in bf16), zero blocks included.
    torch.manual_seed(0)
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(118, 134, (n, k // 32), dtype=torch.uint8, device="cuda")
    s[0, :3] = 0
    out = torch.ops.strix.mxfp4_dequant(w, s)
    assert out.dtype == torch.bfloat16 and out.shape == (n, k)
    # A zero scale byte is a zero block (as in the GEMM), not 2^-127 * v.
    ref = torch_dequant(w, s) * (s != 0).repeat_interleave(32, dim=1)
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("n,k", [(256, 1024), (1024, 2688), (512, 5376)])
@pytest.mark.parametrize("m", [1, 2, 3, 5, 8])
def test_gemv_matches_quark_dequant(n, k, m):
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(118, 134, (n, k // 32), dtype=torch.uint8, device="cuda")
    out = torch.ops.strix.mxfp4_gemv(x, w, s)
    ref = reference(x, w, s)
    torch.testing.assert_close(
        out.float() / ref.abs().amax(), ref / ref.abs().amax(), rtol=0, atol=1e-2
    )


@pytest.mark.parametrize("bias_dim", [1, 2])
def test_bias(bias_dim):
    torch.manual_seed(0)
    n, k, m = 512, 1024, 33
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(118, 134, (n, k // 32), dtype=torch.uint8, device="cuda")
    bias = torch.randn(
        n if bias_dim == 1 else m * n, dtype=torch.bfloat16, device="cuda"
    )
    if bias_dim == 2:
        bias = bias.reshape(m, n)
    out = torch.ops.strix.mxfp4_wmma_gemm(w, x, s, bias)
    ref = reference(x, w, s) + bias.float()
    torch.testing.assert_close(
        out.float() / ref.abs().amax(), ref / ref.abs().amax(), rtol=0, atol=1e-2
    )


def test_rejects_unbuilt_shapes():
    x = torch.zeros(4, 96, dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(8, 48, dtype=torch.uint8, device="cuda")
    s = torch.zeros(8, 3, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="K must be a multiple of 128"):
        torch.ops.strix.mxfp4_wmma_gemm(w, x, s, None)
    x = torch.zeros(4, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(8, 64, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="scale must be uint8"):
        torch.ops.strix.mxfp4_wmma_gemm(w, x, w, None)


@pytest.mark.parametrize("m", [1024, 2048])
def test_prefill_dequant_path_matches_gemm(m):
    # What the plugin runs at M >= DEQUANT_MIN_M: hipBLASLt on the decoded
    # matrix; same bf16 operands as the fused tile, only the summation order
    # differs, so outputs may land one bf16 ulp apart (2^-7 of the largest).
    torch.manual_seed(0)
    n, k = 512, 2688
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    s = torch.randint(118, 134, (n, k // 32), dtype=torch.uint8, device="cuda")
    out = torch.nn.functional.linear(x, torch.ops.strix.mxfp4_dequant(w, s))
    ref = torch.ops.strix.mxfp4_wmma_gemm(w, x, s, None)
    scale = ref.float().abs().amax()
    torch.testing.assert_close(
        out.float() / scale, ref.float() / scale, rtol=0, atol=2**-7
    )
