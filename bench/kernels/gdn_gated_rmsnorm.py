# SPDX-License-Identifier: MIT
"""libstrix gated RMSNorm vs vLLM's Triton rmsnorm_fn at GDN output shapes."""

import strix
import torch

PEAK_GBS = 225.0  # measured gfx1151 bandwidth


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def main():
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn

    cols = ("tokens", "heads", "triton us", "strix us", "speedup", "GB/s", "%peak")
    print(" ".join(f"{c:>10}" for c in cols))
    for heads in (48, 32):
        for tokens in (1, 4, 16, 64, 512, 2048, 8192):
            x = torch.randn(tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
            z = torch.randn_like(x)
            w = torch.ones(128, device="cuda", dtype=torch.bfloat16)
            t_tri = bench(
                lambda x=x, z=z, w=w: rmsnorm_fn(
                    x, w, None, z=z, eps=1e-6, norm_before_gate=True
                )
            )
            t_str = bench(
                lambda x=x, z=z, w=w: strix.gdn_gated_rmsnorm_h128_bf16(x, z, w, 1e-6)
            )
            gbs = 3 * x.numel() * 2 / (t_str * 1e-6) / 1e9
            row = (tokens, heads, f"{t_tri:.1f}", f"{t_str:.1f}")
            row += (f"{t_tri / t_str:.2f}x", f"{gbs:.1f}", f"{gbs / PEAK_GBS:.0%}")
            print(" ".join(f"{c:>10}" for c in row))


if __name__ == "__main__":
    main()
