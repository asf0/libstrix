# SPDX-License-Identifier: MIT
"""strix GDN causal conv + post-conv prep vs vLLM's Triton versions.

Run plainly for the installed vLLM, or under bench/upstream.sh for upstream.
"""

import argparse

import strix
import torch


def time_us(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3


def main() -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
        fused_post_conv_prep,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.3)
    print(
        f"{'model':>5} {'T':>6} {'op':>9} {'vllm us':>8} {'strix us':>8} {'speedup':>7} {'GB/s':>5}"
    )
    for model, hk, hv in (("27b", 16, 48), ("35b", 16, 32)):
        dim = (2 * hk + hv) * 128
        for t in (2048, 8192):
            torch.manual_seed(0)
            proj = torch.randn(t, dim + hv * 128, device="cuda").bfloat16()
            x = proj[:, :dim]
            w = torch.randn(dim, 4, device="cuda").bfloat16()
            states = torch.zeros(4, 3, dim, device="cuda").bfloat16().transpose(1, 2)
            cu = torch.tensor([0, t], dtype=torch.int32, device="cuda")
            idx = torch.tensor([1], dtype=torch.int32, device="cuda")
            init = torch.tensor([False], device="cuda")

            def v_conv(x=x, w=w, states=states, cu=cu, idx=idx, init=init):
                causal_conv1d_fn(
                    x.t(),
                    w,
                    None,
                    conv_states=states,
                    query_start_loc=cu,
                    cache_indices=idx,
                    has_initial_state=init,
                    activation="silu",
                )

            def s_conv(x=x, w=w, states=states, cu=cu, idx=idx, init=init):
                strix.gdn_causal_conv1d(x, w, None, states, cu, idx, init)

            conv = strix.gdn_causal_conv1d(x, w, None, states, cu, idx, init)
            ab = torch.randn(t, 2 * hv, device="cuda").bfloat16()
            a, b = ab[:, :hv], ab[:, hv:]
            a_log = torch.rand(hv, device="cuda")
            dt_bias = torch.randn(hv, device="cuda")

            def v_post(conv=conv, a=a, b=b, a_log=a_log, dt_bias=dt_bias, hk=hk):
                fused_post_conv_prep(conv, a, b, a_log, dt_bias, hk, 128, 128)

            def s_post(conv=conv, a=a, b=b, a_log=a_log, dt_bias=dt_bias, hk=hk):
                strix.gdn_post_conv(conv, a, b, a_log, dt_bias, hk)

            nbytes = 2 * t * dim * 2
            for name, vf, sf in (
                ("conv", v_conv, s_conv),
                ("post_conv", v_post, s_post),
            ):
                tv, ts = time_us(vf, args.iters), time_us(sf, args.iters)
                print(
                    f"{model:>5} {t:>6} {name:>9} {tv:8.0f} {ts:8.0f} "
                    f"{tv / ts:6.2f}x {nbytes / ts / 1e3:5.0f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
