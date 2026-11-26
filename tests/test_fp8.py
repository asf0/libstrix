# SPDX-License-Identifier: MIT
import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")


def make(n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g)
    w_q = (w * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    bs = torch.rand((n + 127) // 128, k // 128, device="cuda", generator=g) * 0.02 + 0.005
    return w_q, bs


def ref_weight(w_q, bs):
    n, k = w_q.shape
    s = bs.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)
    return w_q.float() * s


# 27B shapes: qkvz 16384x5120, down 5120x17408, small.
SHAPES = [(16384, 5120), (5120, 17408), (256, 1024), (6144, 5120)]


@pytest.mark.parametrize("n,k", SHAPES)
def test_dequant_exact(n, k):
    w_q, bs = make(n, k)
    out = torch.ops.strix.fp8_dequant(w_q, bs)
    ref = ref_weight(w_q, bs).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_dequant_padded_rows():
    w_q, bs = make(512, 1024)
    buf = torch.empty(512, 1024 + 128, dtype=torch.float8_e4m3fn, device="cuda")
    buf[:, :1024] = w_q
    out = torch.ops.strix.fp8_dequant(buf[:, :1024], bs)
    torch.testing.assert_close(out, ref_weight(w_q, bs).to(torch.bfloat16), rtol=0, atol=0)


def test_fp8_decode_all_bytes():
    # Every finite e4m3fn code, once per row, against the float reference.
    codes = torch.arange(256, device="cuda", dtype=torch.uint8)
    codes = codes[(codes & 0x7F) != 0x7F]  # no NaN
    w_q = codes.view(torch.float8_e4m3fn).reshape(1, -1).repeat(128, 1)
    k = 128 * ((w_q.size(1) + 127) // 128)
    w_q = torch.nn.functional.pad(w_q.view(torch.uint8), (0, k - w_q.size(1))).view(
        torch.float8_e4m3fn
    )
    bs = torch.ones(1, k // 128, device="cuda")
    out = torch.ops.strix.fp8_dequant(w_q.contiguous(), bs)
    torch.testing.assert_close(out.float(), w_q.float(), rtol=0, atol=0)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 3, 5])
def test_skinny_matches_reference(n, k, m):
    w_q, bs = make(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    s = bs.repeat_interleave(128, 0)[:n].contiguous()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    out = torch.ops.strix.fp8_skinny_gemm(w_q, x, s, bias, 40)
    ref = x.float() @ ref_weight(w_q, bs).t() + bias.float()
    scale = ref.abs().amax()
    torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=8e-3)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [6, 16, 17, 33, 64, 129, 300])
def test_wmma_matches_reference(n, k, m):
    w_q, bs = make(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = torch.ops.strix.fp8_wmma_gemm(w_q, x, bs)
    ref = x.float() @ ref_weight(w_q, bs).t()
    scale = ref.abs().amax()
    torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=8e-3)


@pytest.mark.parametrize("cfg", range(10))
def test_wmma_every_cfg(cfg):
    w_q, bs = make(5120, 5120 + 1024)
    x = torch.randn(70, 5120 + 1024, device="cuda", dtype=torch.bfloat16)
    out = torch.ops.strix.fp8_wmma_gemm_cfg(w_q, x, bs, cfg)
    ref = x.float() @ ref_weight(w_q, bs).t()
    scale = ref.abs().amax()
    torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=8e-3)


def test_wmma_padded_rows():
    w_q, bs = make(512, 1024)
    buf = torch.empty(512, 1024 + 128, dtype=torch.float8_e4m3fn, device="cuda")
    buf[:, :1024] = w_q
    x = torch.randn(40, 1024, device="cuda", dtype=torch.bfloat16)
    out = torch.ops.strix.fp8_wmma_gemm(buf[:, :1024], x, bs)
    ref = x.float() @ ref_weight(w_q, bs).t()
    scale = ref.abs().amax()
    torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=8e-3)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [6, 16, 17, 33, 64, 129, 300])
def test_wmma_raw_matches_reference(n, k, m):
    w_q, bs = make(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    out = torch.ops.strix.fp8_wmma_gemm_raw(w_q, x, bs)
    assert out.dtype == torch.bfloat16
    ref = x.float() @ ref_weight(w_q, bs).t()
    scale = ref.abs().amax()
    # exact weights and fp32 products: only the bf16 output rounding is left
    torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=4e-3)


@pytest.mark.parametrize("cfg", range(10))
def test_wmma_raw_every_cfg(cfg):
    w_q, bs = make(5120, 5120 + 1024)
    x = torch.randn(70, 5120 + 1024, device="cuda", dtype=torch.float16)
    out = torch.ops.strix.fp8_wmma_gemm_raw_cfg(w_q, x, bs, cfg)
    ref = x.float() @ ref_weight(w_q, bs).t()
    scale = ref.abs().amax()
    torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=4e-3)


def test_wmma_raw_ragged_n_and_padded_rows():
    # N not a multiple of 128 (the last scale block is partial) and padded rows.
    n, k = 300, 1024
    w_q, bs = make(n, k)
    buf = torch.empty(n, k + 128, dtype=torch.float8_e4m3fn, device="cuda")
    buf[:, :k] = w_q
    x = torch.randn(40, k, device="cuda", dtype=torch.float16)
    out = torch.ops.strix.fp8_wmma_gemm_raw(buf[:, :k], x, bs)
    ref = x.float() @ ref_weight(w_q, bs).t()
    scale = ref.abs().amax()
    torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=4e-3)


# Per-output-channel scales (compressed-tensors "channel" checkpoints: Sage-A1.5,
# MiniCPM-V-4.7, gemma-4). gemma's K = 2112 / 704 are multiples of 64, not 128.
CH_SHAPES = [(4096, 2048), (2048, 4096), (2816, 2112), (4224, 2816), (300, 704)]


def make_ch(n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g)
    w_q = (w * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = torch.rand(n, device="cuda", generator=g) * 0.02 + 0.005
    return w_q, s


@pytest.mark.parametrize("n,k", CH_SHAPES)
def test_channel_dequant_exact(n, k):
    w_q, s = make_ch(n, k)
    out = torch.ops.strix.fp8_dequant(w_q, s)
    torch.testing.assert_close(out, (w_q.float() * s[:, None]).to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("n,k", CH_SHAPES)
@pytest.mark.parametrize("m", [1, 2, 5])
def test_channel_skinny(n, k, m):
    w_q, s = make_ch(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = torch.ops.strix.fp8_skinny_gemm(w_q, x, s, None, 40)
    ref = x.float() @ (w_q.float() * s[:, None]).t()
    scale = ref.abs().amax()
    torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=8e-3)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("n,k", CH_SHAPES)
@pytest.mark.parametrize("m", [6, 17, 64, 129, 300])
def test_channel_wmma(n, k, m, raw):
    w_q, s = make_ch(n, k)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16 if raw else torch.bfloat16)
    op = torch.ops.strix.fp8_wmma_gemm_raw if raw else torch.ops.strix.fp8_wmma_gemm
    out = op(w_q, x, s)
    ref = x.float() @ (w_q.float() * s[:, None]).t()
    scale = ref.abs().amax()
    torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=8e-3)


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("cfg", range(10))
def test_channel_every_cfg(cfg, raw):
    for k in (2112, 2816):  # BK 64 and 128 tiles
        w_q, s = make_ch(1408, k)
        x = torch.randn(70, k, device="cuda", dtype=torch.float16 if raw else torch.bfloat16)
        op = torch.ops.strix.fp8_wmma_gemm_raw_cfg if raw else torch.ops.strix.fp8_wmma_gemm_cfg
        out = op(w_q, x, s, cfg)
        ref = x.float() @ (w_q.float() * s[:, None]).t()
        scale = ref.abs().amax()
        torch.testing.assert_close(out.float() / scale, ref / scale, rtol=0, atol=8e-3)


def _channel_kernel():
    """strix_vllm's StrixFp8ChannelMMKernel, registered once."""
    pytest.importorskip("vllm")
    import vllm.model_executor.kernels.linear as linear_kernels
    from vllm.platforms import PlatformEnum

    from strix_vllm import fp8 as strix_fp8

    kernels = linear_kernels._POSSIBLE_FP8_KERNELS[PlatformEnum.ROCM]
    found = [c for c in kernels if c.__name__ == "StrixFp8ChannelMMKernel"]
    if not found:
        strix_fp8.register()
        found = [c for c in kernels if c.__name__ == "StrixFp8ChannelMMKernel"]
    return found[0]


# vLLM hands per-channel weights over as a transposed (K, N) view. A square
# weight has the same sizes either way: it was once used transposed (an online
# FP8 draft's 2048 x 2048 o_proj then proposed garbage). The layer's own output
# is checked, against a copy of the weight: vLLM's replace_parameter copies in
# place when the shapes match, so the handed-over storage itself is rewritten.
@pytest.mark.parametrize("n,k", [(256, 256), (512, 256), (256, 512)])
@pytest.mark.parametrize("m", [1, 8])
def test_channel_layer_square_and_not(n, k, m):
    from types import SimpleNamespace

    cls = _channel_kernel()
    kernel = cls.__new__(cls)
    kernel.config = SimpleNamespace(weight_shape=(n, k))
    kernel.layer_param_names = ("weight", "weight_scale", "input_scale", "input_scale_ub")
    w_q, _ = make(n, k)
    scale = torch.rand(n, 1, device="cuda") + 0.5
    ref_w = w_q.float() * scale
    layer = torch.nn.Module()
    # (N, K) storage handed over as its (K, N) view, as process_fp8_weight_channel_strategy does.
    layer.weight = torch.nn.Parameter(w_q.clone().t(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
    kernel.process_weights_after_loading(layer)
    x = torch.randn(m, k, device="cuda").bfloat16()
    out = kernel.apply_weights(layer, x)
    ref = x.float() @ ref_w.t()
    torch.testing.assert_close(out.float(), ref, atol=0.05 * ref.abs().max().item(), rtol=0.02)
