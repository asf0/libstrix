# SPDX-License-Identifier: MIT
"""strix moe_gemm_bf16 vs vLLM's Triton fused_moe_kernel on one geometry.

Defaults are Qwen3.6-35B-A3B shapes; --experts/--topk/--hidden/--inter retarget
it (e.g. gemma-4-26B-A4B: --experts 128 --topk 8 --hidden 2816 --inter 704).

Each timed graph cycles through --routings random routings so the selected
experts are not cache-resident between launches (the model touches a new set
every layer). Triton runs vLLM's tuned config for this GPU on the packed
weights vLLM uses today and on weights with a padded row pitch; strix runs on
the padded weights (packed expert rows are 2^n bytes apart, which camps on
the same DRAM channels on gfx1151).

--activation times the epilogue the model actually runs: Triton's GEMM
followed by vLLM's standalone silu_and_mul / gelu_tanh_and_mul, against
strix's fused gate-and-mul. Without it both sides time the bare GEMM.
"""

import argparse
from functools import partial

import strix  # noqa: F401  (registers torch.ops.strix)
import torch
import triton.language as tl

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


def padded(shape, pad: int, src: torch.Tensor | None = None) -> torch.Tensor:
    """A [..., k] view whose rows are pitch bytes apart (src's data if given)."""
    *lead, k = shape
    if src is None:
        src = torch.randn(*lead, k, device="cuda", dtype=torch.bfloat16)
    store = torch.empty(*lead, k + pad, device="cuda", dtype=torch.bfloat16)
    store[..., :k].copy_(src)
    return store[..., :k]


def run_triton(invoke, routing, a, w, c, weights, top_k, cfg, view_topk, act=None):
    t = weights.size(0)
    for s, e, n in routing:
        invoke(
            a,
            w,
            c.view(t, view_topk, -1),
            None,
            None,
            weights if top_k == 1 else None,
            s,
            e,
            n,
            top_k == 1,
            top_k,
            cfg,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
        )
        if act is not None:
            act()


def run_strix(
    routing, a, w, c, weights, top_k, bm, bn, silu_mul=False, activation="silu"
):
    flat_w = weights.flatten() if top_k == 1 else None
    for s, e, n in routing:
        torch.ops.strix.moe_gemm_bf16(
            c, a, w, s, e, n, flat_w, top_k, bm, bn, silu_mul, activation
        )


def main() -> None:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
        try_get_optimal_moe_config,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="1,4,16,64,256,1024,4096,8192")
    p.add_argument("--routings", type=int, default=8)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--pad", type=int, default=64)
    p.add_argument("--blocks", help="only these strix tiles, e.g. 128x128,64x128")
    p.add_argument("--experts", type=int, default=E)
    p.add_argument("--topk", type=int, default=TOP_K)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--inter", type=int, default=INTER)
    p.add_argument(
        "--activation",
        choices=("none", "silu", "gelu_tanh"),
        default="none",
        help="w13 epilogue; Triton pays a separate activation kernel",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="compare strix against Triton on identical weights/routing",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="print every strix tile instead of only the best",
    )
    args = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.3)
    torch.manual_seed(0)
    e, top_k, hidden, inter = args.experts, args.topk, args.hidden, args.inter
    act = None if args.activation == "none" else args.activation
    act_op = {
        "silu": torch.ops._C.silu_and_mul,
        "gelu_tanh": torch.ops._C.gelu_tanh_and_mul,
    }.get(act)

    shapes = {"w13": (e, 2 * inter, hidden), "w2": (e, hidden, inter)}
    weights_by_gemm = {}
    for name, shape in shapes.items():
        w = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        weights_by_gemm[name] = (w, padded(shape, args.pad, src=w))

    print(
        f"E={e} top_k={top_k} hidden={hidden} inter={inter} "
        f"act={args.activation} pad={args.pad}"
    )
    print(
        f"{'gemm':>4} {'T':>5} {'triton':>8} {'tri+pad':>8} {'strix':>8} "
        f"{'cfg':>7} {'vs tri':>6} {'TFLOPS':>6} {'GB/s':>5}   (us)"
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
            None,
            t,
        )
        # vLLM's TritonExperts skips moe_align_block_size when the routing is
        # sparse enough (one block per (token, expert) pair; see
        # _prepare_expert_assignment). Model that, or the small-batch baseline
        # is the slower aligned path.
        if t * top_k * 4 <= e:
            tri_routing = [
                (
                    None,
                    r.view(-1),
                    torch.full(
                        (1,),
                        r.numel() * cfg["BLOCK_SIZE_M"],
                        dtype=torch.int32,
                        device="cuda",
                    ),
                )
                for r in routes
            ]
        else:
            tri_routing = [
                moe_align_block_size(r, cfg["BLOCK_SIZE_M"], e) for r in routes
            ]
        touched = sum(r.unique().numel() for r in routes) / len(routes)
        for gemm, (w_packed, w_pad) in weights_by_gemm.items():
            n_out, k = w_packed.shape[1:]
            rows, a_topk = (t, top_k) if gemm == "w13" else (t * top_k, 1)
            fused = act is not None and gemm == "w13"
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
                top_k=a_topk,
                routing=tri_routing,
                weights=weights,
                cfg=cfg,
                act=act_fn if fused else None,
                c=c_tri,
            ):
                fn = partial(
                    run_triton,
                    invoke_fused_moe_triton_kernel,
                    routing,
                    a,
                    w,
                    c,
                    weights,
                    top_k,
                    cfg,
                    args.topk,
                    act,
                )
                return graph_us(fn, args.reps) / args.routings

            t_tri, t_tri_pad = time_triton(w_packed), time_triton(w_pad)
            best = None
            for bm, bn in blocks:
                if n_out % bn:
                    continue
                routing = [moe_align_block_size(r, bm, e) for r in routes]
                fn = partial(
                    run_strix,
                    routing,
                    a,
                    w_pad,
                    c_strix,
                    weights,
                    a_topk,
                    bm,
                    bn,
                    fused,
                    act or "silu",
                )
                us = graph_us(fn, args.reps) / args.routings
                if args.all:
                    print(f"    tile {gemm} T={t} {bm}x{bn} {us:.1f}", flush=True)
                if best is None or us < best[0]:
                    best = (us, f"{bm}x{bn}", bm, bn)
            us, name, bm, bn = best
            if args.check:
                run_triton(
                    invoke_fused_moe_triton_kernel,
                    tri_routing[:1],
                    a,
                    w_packed,
                    c_tri,
                    weights,
                    a_topk,
                    cfg,
                    args.topk,
                    act_fn if fused else None,
                )
                run_strix(
                    [moe_align_block_size(routes[0], bm, e)],
                    a,
                    w_pad,
                    c_strix,
                    weights,
                    a_topk,
                    bm,
                    bn,
                    fused,
                    act or "silu",
                )
                ref = act_out if fused else c_tri
                # bf16 GEMM vs Triton can differ at a rounding tie; the unit
                # tests hold the kernels to 2e-2, this is a sanity check on a
                # whole routing (one element out of 23M exceeded 2e-2).
                torch.testing.assert_close(
                    c_strix.float(), ref.float(), rtol=3e-2, atol=2e-2
                )
                print(f"{gemm} check ok ({bm}x{bn})")
            flops = 2 * t * top_k * n_out * k
            bytes_ = (touched * n_out * k + t * top_k * (n_out + k)) * 2
            print(
                f"{gemm:>4} {t:5d} {t_tri:8.1f} {t_tri_pad:8.1f} {us:8.1f} "
                f"{name:>7} {t_tri / us:5.2f}x {flops / us / 1e6:6.1f} "
                f"{bytes_ / us / 1e3:5.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
