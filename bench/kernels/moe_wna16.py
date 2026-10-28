# SPDX-License-Identifier: MIT
"""strix moe_gemm_w4a16 vs vLLM's Triton WNA16 expert kernel, same packed
weights and routing. Defaults are gemma-4-26B-A4B AWQ shapes.

Weights are generated as uint4 values, packed into vLLM's Triton WNA16 layout
(uint8 [E, N, K/2], byte j = k 2j low / 2j+1 high) and scaled per `--group`.
Triton pays a separate activation kernel for w13; strix fuses it.
"""

import argparse
from functools import partial

import strix  # noqa: F401  (registers torch.ops.strix)
import torch
import triton.language as tl

E, TOP_K, HIDDEN, INTER, GROUP = 128, 8, 2816, 704, 32
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


def pack(q: torch.Tensor) -> torch.Tensor:
    return (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()


def run_triton(
    invoke, routing, a, w, c, scales, weights, top_k, cfg, group, view_topk, act=None
):
    t = weights.size(0)
    for s, e, n in routing:
        invoke(
            a,
            w,
            c.view(t, view_topk, -1),
            scales,
            None,
            weights if top_k == 1 else None,
            s,
            e,
            n,
            top_k == 1,
            top_k,
            cfg,
            compute_type=tl.bfloat16,
            use_int8_w8a16=False,
            use_int4_w4a16=True,
            block_shape=[0, group],
        )
        if act is not None:
            act()


def run_strix(
    routing, a, w, scales, c, weights, top_k, bm, bn, silu_mul, activation, group
):
    flat = weights.flatten() if top_k == 1 else None
    for s, e, n in routing:
        torch.ops.strix.moe_gemm_w4a16(
            c, a, w, scales, s, e, n, flat, top_k, bm, bn, silu_mul, activation, group
        )


def main() -> None:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_wna16_triton_kernel,
        try_get_optimal_moe_config,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1,4,16,64,256,1024,4096,8192")
    p.add_argument("--routings", type=int, default=8)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--blocks", help="only these strix tiles, e.g. 128x128,64x128")
    p.add_argument("--experts", type=int, default=E)
    p.add_argument("--topk", type=int, default=TOP_K)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--inter", type=int, default=INTER)
    p.add_argument("--group", type=int, default=GROUP)
    p.add_argument("--activation", choices=("silu", "gelu_tanh"), default="gelu_tanh")
    p.add_argument("--check", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.5)
    torch.manual_seed(0)
    e, top_k, hidden, inter, group = (
        args.experts,
        args.topk,
        args.hidden,
        args.inter,
        args.group,
    )
    act_op = {
        "silu": torch.ops._C.silu_and_mul,
        "gelu_tanh": torch.ops._C.gelu_tanh_and_mul,
    }[args.activation]

    shapes = {"w13": (e, 2 * inter, hidden), "w2": (e, hidden, inter)}
    weights_by_gemm = {}
    for name, (ee, n, k) in shapes.items():
        q = torch.randint(0, 16, (ee, n, k), device="cuda", dtype=torch.uint8)
        scales = (
            torch.rand(ee, n, k // group, device="cuda", dtype=torch.bfloat16) * 0.02
            + 0.001
        )
        weights_by_gemm[name] = (pack(q), scales)

    print(
        f"E={e} top_k={top_k} hidden={hidden} inter={inter} "
        f"group={group} act={args.activation}"
    )
    print(
        f"{'gemm':>4} {'T':>5} {'triton':>8} {'strix':>8} {'cfg':>7} "
        f"{'vs tri':>6} {'TFLOPS':>6} {'GB/s':>5}   (us)"
    )
    blocks = (
        [tuple(int(v) for v in b.split("x")) for b in args.blocks.split(",")]
        if args.blocks
        else BLOCKS
    )
    for t in (int(x) for x in args.tokens.split(",")):
        routes = [
            torch.rand(t, e, device="cuda").topk(top_k, dim=1).indices.int()
            for _ in range(args.routings)
        ]
        weights = torch.rand(t, top_k, device="cuda")
        cfg = try_get_optimal_moe_config(
            weights_by_gemm["w13"][0].shape,
            weights_by_gemm["w2"][0].shape,
            top_k,
            # The model's quant method passes this exact string; with None the
            # lookup treats the packed K/2 as N and picks a different default.
            "int4_w4a16",
            t,
            block_shape=[0, group],
        )
        tri_routing = [moe_align_block_size(r, cfg["BLOCK_SIZE_M"], e) for r in routes]
        touched = sum(r.unique().numel() for r in routes) / len(routes)
        for gemm, (w, scales) in weights_by_gemm.items():
            n_out, k = w.shape[1], w.shape[2] * 2
            rows, a_topk = (t, top_k) if gemm == "w13" else (t * top_k, 1)
            fused = gemm == "w13"
            a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            c_tri = torch.empty(t * top_k, n_out, device="cuda", dtype=torch.bfloat16)
            c_strix = torch.empty(
                t * top_k,
                n_out // 2 if fused else n_out,
                device="cuda",
                dtype=torch.bfloat16,
            )
            act_out = (
                torch.empty(t * top_k, n_out // 2, device="cuda", dtype=torch.bfloat16)
                if fused
                else None
            )

            def act_fn(op=act_op, out=act_out, src=c_tri):
                op(out, src)

            def time_triton(
                w,
                a=a,
                a_topk=a_topk,
                routing=tri_routing,
                weights=weights,
                cfg=cfg,
                act=act_fn if fused else None,
                c=c_tri,
                scales=scales,
            ):
                fn = partial(
                    run_triton,
                    invoke_fused_moe_wna16_triton_kernel,
                    routing,
                    a,
                    w,
                    c,
                    scales,
                    weights,
                    a_topk,
                    cfg,
                    group,
                    args.topk,
                    act,
                )
                return graph_us(fn, args.reps) / args.routings

            t_tri = time_triton(w)
            best = None
            for bm, bn in blocks:
                if n_out % bn:
                    continue
                routing = [moe_align_block_size(r, bm, e) for r in routes]
                fn = partial(
                    run_strix,
                    routing,
                    a,
                    w,
                    scales,
                    c_strix,
                    weights,
                    a_topk,
                    bm,
                    bn,
                    fused,
                    args.activation,
                    group,
                )
                us = graph_us(fn, args.reps) / args.routings
                if args.all:
                    print(f"    tile {gemm} T={t} {bm}x{bn} {us:.1f}", flush=True)
                if best is None or us < best[0]:
                    best = (us, f"{bm}x{bn}", bm, bn)
            us, name, bm, bn = best
            if args.check:
                run_triton(
                    invoke_fused_moe_wna16_triton_kernel,
                    tri_routing[:1],
                    a,
                    w,
                    c_tri,
                    scales,
                    weights,
                    a_topk,
                    cfg,
                    group,
                    args.topk,
                    act_fn if fused else None,
                )
                run_strix(
                    [moe_align_block_size(routes[0], bm, e)],
                    a,
                    w,
                    scales,
                    c_strix,
                    weights,
                    a_topk,
                    bm,
                    bn,
                    fused,
                    args.activation,
                    group,
                )
                ref = act_out if fused else c_tri
                torch.testing.assert_close(
                    c_strix.float(), ref.float(), rtol=3e-2, atol=2e-2
                )
                print(f"{gemm} check ok ({bm}x{bn})")
            flops = 2 * t * top_k * n_out * k
            bytes_ = (touched * n_out * k / 2 + t * top_k * (n_out + k)) * 2
            print(
                f"{gemm:>4} {t:5d} {t_tri:8.1f} {us:8.1f} "
                f"{name:>7} {t_tri / us:5.2f}x {flops / us / 1e6:6.1f} "
                f"{bytes_ / us / 1e3:5.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
