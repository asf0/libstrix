# SPDX-License-Identifier: MIT
"""Qwen3.8-Flash-Next hc_combine_norm: the row kernel moved from the
strix-vllm fork against vLLM's per-stream kernel, and the hook's routing."""

import pytest
import strix
import torch

pytestmark = pytest.mark.skipif(not strix.is_supported_device(), reason="needs gfx1151")
hc = pytest.importorskip("vllm.models.qwen4_exp.amd.ops.hc")


def _inputs(n, hc_dim, hc_count, shared):
    torch.manual_seed(n)
    dim = hc_dim * hc_count
    residual = torch.randn(n, dim, device="cuda").bfloat16()
    block = torch.randn(n, hc_dim + 64, device="cuda").bfloat16()[:, :hc_dim]
    inj = torch.randn(n, hc_count, device="cuda").bfloat16()
    w = torch.randn(hc_dim if shared else dim, device="cuda").bfloat16() * 0.1
    return residual, block, inj, w


@pytest.mark.parametrize("n", [512, 1031])
@pytest.mark.parametrize("shared", [True, False])
def test_rows_kernel_matches_vllm(n, shared):
    from strix_vllm.qwen4_exp_triton import hc_combine_norm_rows

    args = _inputs(n, 2048, 4, shared)
    out_ref, y_ref = hc._hc_combine_norm(*args, 1e-6, 4)
    out, y = hc_combine_norm_rows(*args, 1e-6, 4)
    torch.testing.assert_close(out, out_ref, atol=0, rtol=0)
    torch.testing.assert_close(y.float(), y_ref.float(), atol=2e-2, rtol=2e-2)


def test_hook_routes_by_rows():
    """Small batches take vLLM's kernel, 512+ rows the row kernel; same results."""
    from strix_vllm import qwen4_exp

    qwen4_exp._register_hc_combine_norm()
    for n in (8, 512):
        args = _inputs(n, 2048, 4, True)
        out, y = torch.ops.vllm.strix_hc_combine_norm(*args, 1e-6, 4)
        out_ref, y_ref = hc._hc_combine_norm(*args, 1e-6, 4)
        torch.testing.assert_close(out, out_ref, atol=0, rtol=0)
        torch.testing.assert_close(y.float(), y_ref.float(), atol=2e-2, rtol=2e-2)
