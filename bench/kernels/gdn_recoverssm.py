# SPDX-License-Identifier: MIT
"""strix gdn_recoverssm_verify_bf16 vs vLLM's Triton gdn_recoverssm_verify,
under CUDA graphs, at the served shapes (35B: 16 key / 32 value heads; 27B:
16 / 48), one window of `--tokens` per request.

    bench/container/run.sh python bench/kernels/gdn_recoverssm.py
"""

import argparse

import torch
from vllm.model_executor.layers.mamba.ops.gdn_recoverssm import gdn_recoverssm_verify

D = 128


def bench(fn, iters=50, layers=8):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn:
            f()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for f in fn:
            f()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters / len(fn) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=8)
    a = p.parse_args()
    import strix  # noqa: F401

    for name, h, hv in (("35b", 16, 32), ("27b", 16, 48)):
        for batch in (1, 2, 4, 8, 16):
            layers = 8
            t = batch * a.tokens
            mk = lambda *s: torch.randn(*s, device="cuda").bfloat16()  # noqa: E731
            q, k, v = mk(1, t, h, D), mk(1, t, h, D), mk(1, t, hv, D)
            ab = mk(t, 2 * hv)
            av, bv = ab[:, :hv], ab[:, hv:]
            a_log, dt = torch.rand(hv, device="cuda"), torch.randn(hv, device="cuda")
            blocks = batch + 1
            # A separate state and records per layer, so a layer's reads are not
            # cache hits of the previous one's.
            states = [torch.randn(blocks, hv, D, D, device="cuda") * 0.1 for _ in range(layers)]
            recs = [
                (
                    torch.zeros(blocks, hv, a.tokens, D, device="cuda"),
                    torch.zeros(blocks, h, a.tokens, D, device="cuda"),
                    torch.zeros(blocks, hv, a.tokens, device="cuda"),
                )
                for _ in range(layers)
            ]
            cu = torch.arange(0, t + 1, a.tokens, device="cuda", dtype=torch.int32)
            idx = torch.arange(1, batch + 1, device="cuda", dtype=torch.int64)
            out = torch.empty_like(v)
            tri = [
                (lambda i=i: gdn_recoverssm_verify(
                    q, k, v, av, bv, a_log, dt, states[i], *recs[i], cu, idx, a.tokens))
                for i in range(layers)
            ]
            hip = [
                (lambda i=i: torch.ops.strix.gdn_recoverssm_verify_bf16(
                    q[0], k[0], v[0], av, bv, a_log, dt, states[i], *recs[i], cu, idx,
                    out[0], D**-0.5, 0))
                for i in range(layers)
            ]
            ut, uh = bench(tri), bench(hip)
            state_mb = batch * hv * D * D * 4 / 1e6
            print(f"{name} b{batch}: triton {ut:7.1f} us  strix {uh:7.1f} us  {ut / uh:4.2f}x"
                  f"  (state {state_mb:.1f} MB: {state_mb / uh * 1e3:5.0f} GB/s)", flush=True)


if __name__ == "__main__":
    main()
