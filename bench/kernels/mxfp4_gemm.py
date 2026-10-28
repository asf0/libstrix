# SPDX-License-Identifier: MIT
"""strix::mxfp4_wmma_gemm tile sweep on the gemma-4-31B-it-MXFP4 shapes.

gemma-4-31B at 5376 hidden / 21504 intermediate: gate_proj [21504, 2688],
up_proj [21504, 2688], down_proj [5376, 10752], q/k/v/o [N, 5376]. The
checkpoint is 19 GB of fp4, so a decode step (M = 1) is a pure weight stream:
89 ms at the 213 GB/s the w8a16 decode benchmarks measure, and the tile only
decides whether that stream is reached (every row is read exactly once at
M = 1, so a second m-block is the one thing that cannot happen).

    python bench/kernels/mxfp4_gemm.py                    # M = 1..513
    python bench/kernels/mxfp4_gemm.py --rows 1,32,512

A bf16 matmul on the dequantized weights is timed alongside: at prefill sizes
the useful target is that wall time (the op does the same bf16 WMMA math, with
4x less weight traffic), and at decode it is the 213 GB/s weight stream.
"""

import argparse
from functools import partial

import strix  # noqa: F401  (registers torch.ops.strix)
import torch
from quark.torch.kernel.mx import dq_mxfp4

# (N, K) of the 31B linears: gate/up, down, qkv, o_proj.
SHAPES = [(21504, 2688), (5376, 10752), (10240, 5376), (5376, 5376)]
CONFIGS = {
    0: "(16,128,1,4) BM16  BN128 18 KB",
    1: "(16,64,1,4)  BM16  BN64  10 KB",
    2: "(32,128,2,4) BM32  BN128 23 KB",
    3: "(64,128,2,4) BM64  BN128 28 KB",
    4: "(128,128,4,2) BM128 BN128 37 KB",
    5: "(128,256,4,4) BM128 BN256 55 KB",
    6: "(64,256,2,8) BM64  BN256 46 KB",
    7: "(128,128,2,4) BM128 BN128 37 KB",
    8: "(16,256,1,4) BM16  BN256 39 KB",
    9: "(32,64,1,4)  BM32  BN64  14 KB",
    10: "(16,64,1,2)  BM16  BN64   9 KB",
    11: "(16,128,1,2) BM16  BN128 18 KB",
    12: "(64,64,1,4)  BM64  BN64  18 KB",
    13: "(16,32,1,2)  BM16  BN32   7 KB",
    14: "(16,32,1,2)  BK128 13 KB",
    15: "(16,64,1,4)  BK128 22 KB",
    16: "(16,64,1,2)  BK128 22 KB",
    17: "(32,64,1,4)  BK128 26 KB",
    18: "(16,128,1,4) BK128 39 KB",
    19: "(64,64,1,4)  BK128 35 KB",
}


def graph_us(launch, reps: int) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        launch()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launch()
    g.replay()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1e3


def build(n: int, k: int, device="cuda"):
    """Packed e2m1 bytes and realistic e8m0 scales (2^-6 .. 2^0)."""
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=device)
    s = torch.randint(121, 128, (n, k // 32), dtype=torch.uint8, device=device)
    return w, s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="1,5,32,64,512,2048")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--check", action="store_true", help="verify every config")
    args = p.parse_args()

    for m in (int(x) for x in args.rows.split(",")):
        print(f"\n=== M = {m}")
        print(
            f"{'shape':>13s} {'bf16 ms':>8s} {'GB/s':>6s} "
            + " ".join(f"cfg{c}".rjust(9) for c in CONFIGS)
        )
        for n, k in SHAPES:
            w, s = build(n, k)
            act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            w_bf16 = dq_mxfp4(w, s, torch.bfloat16).t()
            ms_bf16 = graph_us(partial(torch.mm, act, w_bf16), args.reps) / 1e3

            # The production tile for this M, for the bandwidth column.
            us_prod = graph_us(
                partial(torch.ops.strix.mxfp4_wmma_gemm, w, act, s, None),
                args.reps,
            )
            gbs = (n * k * 9 / 16 + m * k * 2) / (us_prod * 1e-6) / 1e9

            cols, ref = [], None
            for cfg in CONFIGS:
                # Both weight and activation bytes, which is what the stream
                # moves; the scales are 1/32 of the weight.
                out = torch.ops.strix.mxfp4_wmma_gemm_cfg(w, act, s, None, cfg)
                if ref is None:
                    ref = out.clone()
                elif args.check and not torch.allclose(out, ref):
                    assert torch.allclose(out.float(), ref.float(), atol=2e-2), (
                        f"cfg{cfg} disagrees"
                    )
                us = graph_us(
                    partial(torch.ops.strix.mxfp4_wmma_gemm_cfg, w, act, s, None, cfg),
                    args.reps,
                )
                cols.append(f"{us / 1e3:7.2f}ms")
            shape = f"{n}x{k}"
            print(
                f"{shape:>13s} {ms_bf16:8.2f} {gbs:6.0f} "
                + " ".join(c.rjust(9) for c in cols)
            )
            del w, s, act, w_bf16, ref
            torch.cuda.empty_cache()

    print("\nconfigs: " + "\n         ".join(CONFIGS.values()))


if __name__ == "__main__":
    main()
