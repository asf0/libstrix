# SPDX-License-Identifier: MIT
"""mxfp4_gemv (M <= 8) vs the production MXFP4 WMMA GEMM, decode shapes.

A decode step is a pure weight stream: N*K/2 bytes of fp4 per linear. This
times both kernels on the gemma-4-31B-it-MXFP4 shapes and reports effective
GB/s, so the prototype is judged on whether it raises the stream.

    python bench/kernels/mxfp4_gemv.py
    python bench/kernels/mxfp4_gemv.py --rows 1,4,8 --shapes 21504x2688,5376x10752
"""

import argparse
from functools import partial

import strix  # noqa: F401  (registers torch.ops.strix)
import torch

SHAPES = [(21504, 2688), (5376, 10752), (10240, 5376), (5376, 5376)]


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="1,2,4,8")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument(
        "--shapes", help="NxK list, default gemma-4-31B's four linear shapes"
    )
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    shapes = (
        [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
        if args.shapes
        else SHAPES
    )
    torch.cuda.set_per_process_memory_fraction(0.6)
    torch.manual_seed(0)
    print(
        f"{'N':>7} {'K':>6} {'M':>2} {'wmma us':>8} {'gemv us':>8} "
        f"{'ratio':>6} {'wmma GB/s':>9} {'gemv GB/s':>9}"
    )
    for n, k in shapes:
        w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        s = torch.randint(118, 134, (n, k // 32), dtype=torch.uint8, device="cuda")
        for m in (int(x) for x in args.rows.split(",")):
            x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            us_w = graph_us(
                partial(torch.ops.strix.mxfp4_wmma_gemm, w, x, s, None), args.reps
            )
            us_g = graph_us(partial(torch.ops.strix.mxfp4_gemv, x, w, s), args.reps)
            if args.check:
                a = torch.ops.strix.mxfp4_wmma_gemm(w, x, s, None)
                b = torch.ops.strix.mxfp4_gemv(x, w, s)
                torch.testing.assert_close(
                    b.float() / a.float().abs().amax(),
                    a.float() / a.float().abs().amax(),
                    rtol=0,
                    atol=1e-2,
                )
            nbytes = n * k // 2
            print(
                f"{n:7d} {k:6d} {m:2d} {us_w:8.1f} {us_g:8.1f} "
                f"{us_w / us_g:5.2f}x {nbytes / us_w / 1e3:9.0f} "
                f"{nbytes / us_g / 1e3:9.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
