# SPDX-License-Identifier: MIT
"""Decode GEMM at batch 1..64 across the tile shapes that fit that batch.

The plugin sends M > 5 to the WMMA op, which lands in its BM 32 branch there.
Measured on the served 27B INT8 that leaves the step compute-bound at x16/x32 -
100 GB/s of weight stream against the 213 GB/s peak - so the skinny
op grew CHUNKED large-N instantiations. This picks the crossover.

    python bench/kernels/w8a16_decode.py                 # production row pitch
    python bench/kernels/w8a16_decode.py --pads 0,128    # unpadded vs production

GB/s is the weight stream the step must read once per call: N * K bytes. Only
shapes above the 32 MB infinity cache are meaningful - 5120x6144 fits it and
replays out of L2 at 300 GB/s.

The row pitch matters more than the tile: the plugin pads every int8 weight row
whose K is a multiple of 1024 by one 128 B cache line (strix_vllm/linear.py,
_pack_w8_rows), so the served model runs with a row stride of K + 128. An
unpadded [N, K] buffer is a 1 KiB multiple, every CTA reads byte 0 of its rows
at the same time, and the stream loses 10-20% to DRAM channel aliasing - which
is what cfg 12-15 (the rotated K start, STAGGER) get back. --pads defaults to
128 (what is served); pass 0 to reproduce the aliasing number.
"""

import argparse
from functools import partial

import strix  # noqa: F401  (registers torch.ops.strix)
import torch

# 27B INT8 linears: qkvz, o_proj, gate_up, down_proj.
SHAPES = [(16384, 5120), (34816, 5120), (5120, 17408)]
# Config ids of w8a8_wmma_gemm_cfg with the BM they tile. Every batch works
# with every tile (rows past M are pad tiles with a zero activation and a zero
# scale), so the comparison is like for like - a tile larger than the batch
# just wastes compute, and a tile smaller than the batch reads the weights once
# per m-block. 8/4/11/0 are the production tiles at M <= 16 / 32 / 64 / above;
# 14/13/12 are the same decode tiles with the rotated K start (STAGGER) and 15
# is the production tile at M 33..64 (identical to 12 at decode, where the grid
# has one m-block and the grouped raster is the identity).
CFGS = {8: 16, 4: 32, 11: 64, 0: 128, 14: 16, 13: 32, 12: 64, 15: 64}
STAGGERED = {14, 13, 12, 15}
BATCHES = [1, 5, 16, 32, 40, 64]
GROUP = 128
PAD = 128  # strix_vllm.linear._STRIDE_PAD_BYTES


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


def weights(n: int, k: int, pad: int) -> torch.Tensor:
    """[N, K] uint8 with a row stride of K + pad (the plugin's layout)."""
    w = torch.randint(0, 256, (n, k), dtype=torch.int32, device="cuda")
    if pad == 0:
        return w.to(torch.uint8)
    buf = torch.empty((n, k + pad), dtype=torch.uint8, device="cuda")
    buf[:, :k] = w.to(torch.uint8)
    return buf[:, :k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--pads", default=str(PAD), help="weight row pads, e.g. 0,128")
    args = p.parse_args()
    pads = [int(x) for x in args.pads.split(",")]

    for pad in pads:
        print(f"\n########## weight row pitch K + {pad} B")
        for n, k in SHAPES:
            w = weights(n, k, pad)
            assert w.stride(0) == k + pad, (w.stride(0), k + pad)
            s16 = (torch.rand(n, k // GROUP, device="cuda") * 1e-3).bfloat16()
            s32 = s16.float().t().contiguous()
            print(f"\n=== {n}x{k}  (weights {n * k / 1e9:.2f} GB)")
            head = " ".join(
                f"{'cfg' + str(c) + ('*' if c in STAGGERED else ''):>9s}" for c in CFGS
            )
            print(f"{'batch':>5s} {'skinny':>8s} {head}")
            for m in BATCHES:
                x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                skinny = partial(
                    torch.ops.strix.w8a16_skinny_gemm, w, x, s16, None, 40, GROUP
                )
                wmma = partial(torch.ops.strix.w8a8_wmma_gemm, w, x, s32)
                # Both read the same weights; the WMMA path also quantizes the
                # activations to int8, so they agree to ~1% rather than exactly.
                # The skinny op only takes 5 activation rows, so compare there.
                out_w = wmma()
                if m <= 5:
                    out_s = skinny()
                    scale = out_s.abs().amax()
                    assert (out_s - out_w).abs().max() < 3e-2 * scale, "disagree"
                ts = graph_us(skinny, args.reps) if m <= 5 else float("nan")
                gb = n * k / 1e9
                cols = []
                for cfg in CFGS:
                    t = graph_us(
                        partial(torch.ops.strix.w8a8_wmma_gemm_cfg, w, x, s32, cfg),
                        args.reps,
                    )
                    cols.append(f"{gb / (t * 1e-6):8.0f}")
                skin = f"{gb / (ts * 1e-6):8.0f}" if m <= 5 else "-".rjust(8)
                print(f"{m:5d} {skin} " + " ".join(cols))
                del x, out_w
                torch.cuda.empty_cache()
            del w, s16, s32
            torch.cuda.empty_cache()

    print("\n* = rotated K start (STAGGER); the rest are the production tiles.")


if __name__ == "__main__":
    main()
