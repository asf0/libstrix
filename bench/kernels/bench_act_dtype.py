"""Grouped expert GEMM with bf16 weights: MXFP8 vs bf16 activations."""
import sys
import torch
sys.path.insert(0, "/work/bench/kernels")
from nvfp4_bmm_v2 import _SCRATCH, _bf16_bmm_kernel, bmm_bf16_act_only  # noqa: E402
from bench_nvfp4_bmm import setup, time_us  # noqa: E402
import triton  # noqa: E402

for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
    host, args, plan, experts = setup(n, k, 128, 128, num_tiles=128)
    act, act_scale, *_ , tiles = args
    wb = torch.randn(128, n, k, device="cuda", dtype=torch.bfloat16)
    act_bf16 = act.to(torch.bfloat16)
    flops = 2 * 128 * 128 * n * k
    for bm, bn, nw in ((64, 128, 4), (64, 64, 4), (128, 128, 8)):
        out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device="cuda")
        sub = 128 // bm
        fn8 = lambda: _bf16_bmm_kernel[(n // bn, plan.max_ctas * sub)](act, act_scale, wb, out, tiles, plan.max_ctas, N=n, K=k, TILE=128, BM=bm, BN=bn, BK=32, num_warps=nw, num_stages=2)
        fn16 = lambda: bmm_bf16_act_only(act_bf16, wb, tiles, plan, bm=bm, bn=bn, nw=nw)
        for label, fn in (("mxfp8 act", fn8), ("bf16 act", fn16)):
            us = time_us(fn, iters=10)
            print(f"{name} {bm}x{bn} w{nw} {label:10s} {us:9.1f} us {flops / us / 1e6:6.1f} TFLOPS", flush=True)
