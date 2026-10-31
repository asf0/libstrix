# SPDX-License-Identifier: MIT
"""FP8 linear paths on the Qwen3.8-27B-FP8 shapes, by batch size M.

Per M it times the paths python/strix_vllm/fp8.py chooses between: the skinny
decode kernel in chunks of 5 rows, strix::fp8_wmma_gemm (tile by M and every
fixed tile config), and strix::fp8_dequant + hipBLASLt. Weight bytes are one
per element, so the memory-bound floor is 1 / 213 GB/s per pass.

    python bench/kernels/fp8_gemm.py                 # M = 6..2048
    python bench/kernels/fp8_gemm.py --rows 8,32,512 --cfgs 3,5,6
    python bench/kernels/fp8_gemm.py --shapes 4224x2816,2816x2112 --channel
"""

import argparse

import strix  # noqa: F401  (registers torch.ops.strix)
import torch

# (N, K): qkvz, fused gate/up, o_proj, down. Hidden 5120, intermediate 17408.
SHAPES = [(16384, 5120), (34816, 5120), (5120, 6144), (5120, 17408)]


def time_ms(fn, reps: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="6,8,16,20,24,32,48,64,96,128,256,512,1024,2048")
    p.add_argument("--cfgs", default="")
    p.add_argument("--shapes", default="", help="NxK list (default: the 27B shapes)")
    p.add_argument("--channel", action="store_true", help="per-channel fp32 [N] scales")
    p.add_argument("--pad", type=int, default=-1,
                   help="row pad bytes (default: the plugin's, 128 when K %% 1024 == 0)")
    args = p.parse_args()
    shapes = (
        [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
        if args.shapes else SHAPES
    )
    rows = [int(r) for r in args.rows.split(",")]
    cfgs = [int(c) for c in args.cfgs.split(",")] if args.cfgs else []
    for n, k in shapes:
        w = (torch.randn(n, k, device="cuda") * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
        pad = args.pad if args.pad >= 0 else (128 if k % 1024 == 0 else 0)
        buf = torch.empty(n, k + pad, dtype=torch.float8_e4m3fn, device="cuda")
        buf[:, :k] = w
        w = buf[:, :k]
        if args.channel:
            bs = rs = torch.rand(n, device="cuda") * 0.02 + 0.005
        else:
            bs = torch.rand((n + 127) // 128, k // 128, device="cuda") * 0.02 + 0.005
            rs = bs.repeat_interleave(128, 0)[:n].contiguous()
        print(f"\nN={n} K={k}  (weight {n * k / 1e6:.0f} MB, floor {n * k / 213e9 * 1e3:.2f} ms/pass)")
        head = f"{'M':>5} {'skinny*':>9} {'wmma':>8} {'raw':>8} {'dq+blas':>9}"
        print(head + "".join(f" {'c' + str(c):>7}" for c in cfgs))
        for m in rows:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

            def skinny():
                for i in range(0, m, 5):
                    torch.ops.strix.fp8_skinny_gemm(w, x[i : i + 5], rs, None, 40)

            t_sk = time_ms(skinny) if m <= 64 else float("nan")
            t_w = time_ms(lambda: torch.ops.strix.fp8_wmma_gemm(w, x, bs))
            t_d = time_ms(
                lambda: torch.nn.functional.linear(x, torch.ops.strix.fp8_dequant(w, bs))
            )
            x16 = x.to(torch.float16)
            t_r = time_ms(lambda: torch.ops.strix.fp8_wmma_gemm_raw(w, x16, bs))
            row = f"{m:5d} {t_sk:9.3f} {t_w:8.3f} {t_r:8.3f} {t_d:9.3f}"
            for c in cfgs:
                row += f" {time_ms(lambda: torch.ops.strix.fp8_wmma_gemm_cfg(w, x, bs, c)):7.3f}"
            print(row)


if __name__ == "__main__":
    main()
