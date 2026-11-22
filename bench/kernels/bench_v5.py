"""NVFP4 grouped GEMM with bf16 activations (v5): prefill-sized tiles."""
import sys
import torch
sys.path.insert(0, "/work/bench/kernels")
from nvfp4_bmm_v2 import bmm_nvfp4_v5  # noqa: E402
from bench_nvfp4_bmm import decode, setup, time_us  # noqa: E402

for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
    for tile, rows, num_tiles in ((16, 4, 32), (64, 48, 128), (128, 128, 128)):
        host, args, plan, experts = setup(n, k, tile, rows, num_tiles=num_tiles)
        act, _, weight, scale, amax, denominator, tiles = host
        w = decode(weight, scale, amax, denominator)
        ref = torch.cat([act[t * tile : t * tile + rows].float() @ w[e].T for t, e in enumerate(experts)])
        a16 = args[0].to(torch.bfloat16)
        flops = 2 * len(experts) * rows * n * k
        for bm, bn, nw in ((16, 64, 4), (32, 64, 4), (64, 64, 4), (64, 128, 4), (64, 128, 8), (128, 128, 8)):
            if bm > max(tile, 16):
                continue
            fn = lambda: bmm_nvfp4_v5(a16, *args[2:6], args[6], plan, bm=bm, bn=bn, nw=nw)
            out = fn().float().cpu()
            got = torch.cat([out[t * tile : t * tile + rows] for t in range(len(experts))])
            err = ((got - ref).abs().max() / ref.abs().max()).item()
            us = time_us(fn, iters=10)
            print(f"{name} tile{tile} rows{rows}x{len(experts)} v5 {bm}x{bn} w{nw} {us:9.1f} us {flops / us / 1e6:6.1f} TFLOPS err {err:.1e}", flush=True)
