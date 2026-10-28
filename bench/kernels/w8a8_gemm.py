# SPDX-License-Identifier: MIT
"""strix::w8a8_wmma_gemm tile sweep on the served 27B INT8 shapes.

The GEMM is 84% of an 8k prefill and 73% of a 32k one, at ~40 TOPS,
~34% of the int8 WMMA peak, and the activation quantize is only 2.6% of its
time - so the tile is the whole story. Each config is the same kernel at a
fixed (BMF, BNF, WAVES): fragments are BMF * BNF * 16 VGPRs and the staging is
2 * 128 * (BM + BN) bytes, so BM + BN <= 256 for a 64 KB LDS.

    python bench/kernels/w8a8_gemm.py                 # M = 8192, all shapes
    python bench/kernels/w8a8_gemm.py --rows 2048,8192

A bf16 hipBLASLt matmul of the same shape is timed alongside: the useful
target is to beat the wall time of the bf16 GEMM this op replaced, not the
hardware peak.
"""

import argparse
from functools import partial

import strix  # noqa: F401  (registers torch.ops.strix)
import torch

# (N, K) of the 27B INT8 linears: fused qkvz, o_proj, gate_up, down_proj.
SHAPES = [(14336, 5120), (5120, 6144), (34816, 5120), (5120, 17408)]
CONFIGS = {
    0: "(8,1,8) BM128 BN128 64 KB  - production",
    1: "(4,2,4) BM64  BN128 48 KB",
    2: "(4,1,8) BM64  BN128 48 KB",
    3: "(2,2,4) BM32  BN128 40 KB",
    4: "(2,1,8) BM32  BN128 48 KB  - production, M<=32",
    5: "(8,1,8) + 8xn grouped raster",
    6: "(8,1,8) + 4xn grouped raster",
    7: "(4,2,4) + 8xn grouped raster",
    16: "(8,1,8) + 8xn raster + K stagger",
    17: "(4,2,8) 2x4 wave grid + 8xn raster",
    18: "(2,4,8) 4x2 wave grid + 8xn raster",
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
    """Raw uint8b128 rows (one byte per element) and [K/128, N] fp32 scales."""
    w = torch.randint(0, 256, (n, k), dtype=torch.int32, device=device).to(torch.uint8)
    scale = torch.rand(k // 128, n, dtype=torch.float32, device=device) * 1e-3
    return w, scale


def check(ref: torch.Tensor, out: torch.Tensor, cfg: int) -> None:
    """Configs must agree to the last bits of the bf16 output.

    The K-rotated configs (STAGGER) sum the same per-group products in a
    different order, so a result may round to a neighbouring bf16: one ulp at
    the largest magnitude of the matrix is 2^-7 * amax. Two ulps is still a
    last-bits difference; anything past that is a real disagreement.
    """
    if torch.equal(out, ref):
        return
    ulp = torch.finfo(torch.bfloat16).eps * ref.float().abs().amax().item()
    d = (out.float() - ref.float()).abs().max().item()
    assert d <= 2 * ulp, f"cfg{cfg} disagrees: max|d| {d:.3e} > 2 ulp {2 * ulp:.3e}"


def dequant(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    g = k // 128
    ws = scale.t().reshape(n, g, 1)  # [N, G, 1]
    values = w.view(torch.int8).float().reshape(n, g, 128)  # raw byte - 128
    return (values * ws).reshape(n, k)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="8192", help="M values, comma separated")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--check", action="store_true", help="verify every config")
    args = p.parse_args()

    for m in (int(x) for x in args.rows.split(",")):
        print(f"\n=== M = {m}")
        print(
            f"{'shape':>14s} {'bf16 TF':>8s} "
            + " ".join(f"cfg{c}".rjust(9) for c in CONFIGS)
        )
        for n, k in SHAPES:
            w, s = build(n, k)
            act = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
            w_bf16 = dequant(w, s).bfloat16().t()
            t_us = graph_us(partial(torch.mm, act, w_bf16), args.reps)
            tops_bf16 = 2 * m * n * k / (t_us * 1e-6) / 1e12

            cols, ref = [], None
            for cfg in CONFIGS:
                out = torch.ops.strix.w8a8_wmma_gemm_cfg(w, act, s, cfg)
                if ref is None:
                    ref = out.clone()
                elif args.check:
                    check(ref, out, cfg)
                us = graph_us(
                    partial(torch.ops.strix.w8a8_wmma_gemm_cfg, w, act, s, cfg),
                    args.reps,
                )
                cols.append(f"{2 * m * n * k / (us * 1e-6) / 1e12:7.1f}")
            shape = f"{n}x{k}"
            print(
                f"{shape:>14s} {tops_bf16:8.1f} " + " ".join(c.rjust(9) for c in cols)
            )
            del w, s, act, w_bf16
            torch.cuda.empty_cache()

    print("\nconfigs: " + "\n         ".join(CONFIGS.values()))


if __name__ == "__main__":
    main()
