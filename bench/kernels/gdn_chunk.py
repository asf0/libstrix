# SPDX-License-Identifier: MIT
"""strix gdn_chunk_fwd vs vLLM's FLA chunk_gated_delta_rule (one GDN layer).

Shapes: Qwen3.8-27B (Hk=16, Hv=48) and Qwen3.6-35B-A3B (Hk=16, Hv=32).
Timed eagerly (both run outside CUDA graphs in vLLM's prefill path).
"""

import argparse

import strix
import torch
import torch.nn.functional as F


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
    from vllm.third_party.flash_linear_attention.ops.chunk import (
        chunk_gated_delta_rule,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.3)
    print(f"{'model':>5} {'lens':>18} {'fla us':>9} {'strix us':>9} {'speedup':>7}")
    for model, hk, hv in (("27b", 16, 48), ("35b", 16, 32)):
        for lens in ([2048], [8192], [512] * 8, [100, 3000, 37, 5000]):
            t = sum(lens)
            torch.manual_seed(0)
            q = F.normalize(torch.randn(t, hk, 128, device="cuda"), dim=-1).bfloat16()
            k = F.normalize(torch.randn(t, hk, 128, device="cuda"), dim=-1).bfloat16()
            v = torch.randn(t, hv, 128, device="cuda").bfloat16()
            g = F.logsigmoid(torch.randn(t, hv, device="cuda") + 3)
            beta = torch.rand(t, hv, device="cuda").sigmoid()
            state = torch.zeros(len(lens), hv, 128, 128, device="cuda")
            cu = torch.tensor(
                [0, *torch.tensor(lens).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int32,
            )

            def fla(q=q, k=k, v=v, g=g, beta=beta, state=state, cu=cu):
                chunk_gated_delta_rule(
                    q=q[None],
                    k=k[None],
                    v=v[None],
                    g=g[None],
                    beta=beta[None].bfloat16(),
                    scale=128**-0.5,
                    initial_state=state,
                    output_final_state=True,
                    cu_seqlens=cu,
                    use_qk_l2norm_in_kernel=False,
                )

            def ours(q=q, k=k, v=v, g=g, beta=beta, state=state, cu=cu):
                strix.gdn_chunk_fwd(q, k, v, g, beta, state, cu)

            t_fla, t_ours = time_us(fla, args.iters), time_us(ours, args.iters)
            name = f"{len(lens)}x{lens[0]}" if len(set(lens)) == 1 else "ragged"
            print(
                f"{model:>5} {name:>18} {t_fla:9.0f} {t_ours:9.0f} "
                f"{t_fla / t_ours:6.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
