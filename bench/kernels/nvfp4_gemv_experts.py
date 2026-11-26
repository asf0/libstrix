# SPDX-License-Identifier: MIT
"""libstrix NVFP4 decode GEMV (`gemm_nvfp4` on decode tiles) as a drafted
Inkling 2 Mini step routes it: 1-3 tokens x 8 experts each, so 8, 16 or 24
distinct experts with one row each (or 8 experts with 2-3 rows when tokens
agree), weights streamed from DRAM (a fresh expert set per call), under CUDA
graphs. GB/s counts the weight bytes (e2m1 + e4m3 scales) the call must read.

    bench/container/run.sh python bench/kernels/nvfp4_gemv_experts.py
"""

import torch

from strix_vllm.inkling2_moe import gemm_nvfp4
from vllm.models.inkling2.amd.ops.triton_bmm import ExpertPlan

dev = torch.device("cuda")
E = 128
for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
    weight = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    scale = (torch.rand(E, n, k // 16, device=dev) + 0.5).to(torch.float8_e4m3fn)
    amax = torch.rand(E, device=dev) + 0.5
    den = torch.tensor([2688.0], device=dev)
    for experts, rows in ((8, 1), (16, 1), (24, 1), (8, 3)):
        plan = ExpertPlan(tile=16, max_ctas=experts)
        sets = []
        for i in range(16):
            t = torch.zeros(2 * experts + 3, dtype=torch.int32)
            for j in range(experts):
                t[j] = (i * experts + j) % E
                t[experts + j] = j * 16 + rows
            t[2 * experts] = experts
            sets.append(t.to(dev))
        act = torch.randn(plan.rows, k, device=dev).to(torch.bfloat16)
        g = torch.cuda.CUDAGraph()
        for t in sets:
            gemm_nvfp4(act, weight, scale, amax, den, t, plan)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            for t in sets:
                gemm_nvfp4(act, weight, scale, amax, den, t, plan)
        g.replay()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            g.replay()
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) / 160 * 1000
        gb = experts * n * (k // 2 + k // 16) / 1e9
        print(f"{name} {experts} experts x {rows} rows: {us:7.1f} us {gb / us * 1e6:6.1f} GB/s", flush=True)
