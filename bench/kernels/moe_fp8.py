# SPDX-License-Identifier: MIT
"""strix::moe_gemm_fp8 tile sweep at decode batch sizes, Qwen3.6-35B-A3B-FP8.

E=256 experts, top-8, hidden 2048, expert intermediate 512: w13 is
[E, 1024, 2048] and w2 [E, 2048, 512], both float8_e4m3fn with 128x128 block
scales. A decode step touches at most min(E, tokens * top_k) experts, each
read once, so the floor is those bytes at the ~213 GB/s this APU streams.

    python bench/kernels/moe_fp8.py                       # tokens 1..16
    python bench/kernels/moe_fp8.py --tokens 1,64,1024 --scheme channel
"""

import argparse

import strix  # noqa: F401  (registers torch.ops.strix)
import torch

E, TOP_K, HIDDEN, INTER = 256, 8, 2048, 512
BLOCKS = [(16, 64), (32, 64), (64, 64), (64, 128), (128, 64), (128, 128), (64, 256)]


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


def align(topk_ids: torch.Tensor, block_m: int, num_experts: int):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(topk_ids, block_m, num_experts)


def main() -> None:
    global E, TOP_K, HIDDEN, INTER
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1,2,4,8,16")
    p.add_argument("--routings", type=int, default=8)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--scheme", choices=("block", "channel"), default="block")
    p.add_argument("--sets", type=int, default=6,
                   help="independent weight sets timed round-robin, so the weights "
                   "are cold (a decode step reads each layer's experts once)")
    p.add_argument("--raw", action="store_true",
                   help="the fp16-activation kernel (moe_gemm_fp8_raw)")
    p.add_argument("--pad", type=int, default=0,
                   help="extra bytes per weight row (breaks power-of-two strides)")
    p.add_argument("--experts", type=int, default=E)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--inter", type=int, default=INTER)
    p.add_argument("--act", default="silu", choices=("silu", "gelu_tanh"))
    p.add_argument("--blocks", default=None,
                   help="tiles to time, e.g. 16x64,32x64 (default: all)")
    args = p.parse_args()
    E, TOP_K, HIDDEN, INTER = args.experts, args.top_k, args.hidden, args.inter
    blocks = (
        [tuple(int(v) for v in b.split("x")) for b in args.blocks.split(",")]
        if args.blocks else BLOCKS
    )
    torch.manual_seed(0)
    shapes = {"w13": (2 * INTER, HIDDEN), "w2": (HIDDEN, INTER)}
    def make_w(n, k):
        full = torch.empty(E, n, k + args.pad, dtype=torch.float8_e4m3fn, device="cuda")
        full[:, :, :k] = (torch.randn(E, n, k, device="cuda") * 40).clamp(-448, 448).to(
            torch.float8_e4m3fn
        )
        return full[:, :, :k]

    w = {g: [make_w(n, k) for _ in range(args.sets)] for g, (n, k) in shapes.items()}
    sc = {}
    for g, (n, k) in shapes.items():
        sc[g] = [
            torch.rand(E, n // 128, k // 128, device="cuda") * 0.02 + 0.001
            if args.scheme == "block" and n % 128 == 0 and k % 128 == 0
            else torch.rand(E, n, device="cuda") * 0.02 + 0.001
            for _ in range(args.sets)
        ]
    print(f"scheme {args.scheme}; times are us per GEMM, GB/s is expert bytes / time")
    for tokens in [int(t) for t in args.tokens.split(",")]:
        routes = [
            torch.stack([torch.randperm(E, device="cuda")[:TOP_K] for _ in range(tokens)])
            for _ in range(args.routings)
        ]
        adt = torch.float16 if args.raw else torch.bfloat16
        op = torch.ops.strix.moe_gemm_fp8_raw if args.raw else torch.ops.strix.moe_gemm_fp8
        a1 = torch.randn(tokens, HIDDEN, device="cuda", dtype=adt)
        a2 = torch.randn(tokens * TOP_K, INTER, device="cuda", dtype=adt)
        weights = torch.rand(tokens * TOP_K, device="cuda")
        active = min(E, tokens * TOP_K)
        print(f"\n== tokens {tokens} (about {active} experts active)")
        print(f"{'tile':>9} {'w13 us':>8} {'GB/s':>6} {'w2 us':>8} {'GB/s':>6} {'sum us':>8}")
        for bm, bn in blocks:
            if (2 * INTER) % bn or HIDDEN % bn:
                continue
            row = {}
            for g, (n, k) in shapes.items():
                times, gbs = [], []
                for ids in routes:
                    sorted_ids, expert_ids, num_post = align(ids, bm, E)
                    if g == "w13":
                        c = torch.empty(tokens * TOP_K, n // 2, device="cuda", dtype=adt)

                        def launch():
                            for i in range(args.sets):
                                op(
                                    c, a1, w[g][i], sc[g][i], sorted_ids, expert_ids,
                                    num_post, None, TOP_K, bm, bn, True, args.act)
                    else:
                        c = torch.empty(tokens * TOP_K, n, device="cuda", dtype=torch.bfloat16)

                        def launch():
                            for i in range(args.sets):
                                op(
                                    c, a2, w[g][i], sc[g][i], sorted_ids, expert_ids,
                                    num_post, weights, 1, bm, bn, False, args.act)
                    t = graph_us(launch, args.reps) / args.sets
                    n_active = ids.unique().numel()
                    times.append(t)
                    gbs.append(n_active * n * k / t / 1e3)
                row[g] = (sum(times) / len(times), sum(gbs) / len(gbs))
            tot = row["w13"][0] + row["w2"][0]
            print(f"{bm:>4}x{bn:<4} {row['w13'][0]:8.1f} {row['w13'][1]:6.0f} "
                  f"{row['w2'][0]:8.1f} {row['w2'][1]:6.0f} {tot:8.1f}")


if __name__ == "__main__":
    main()
