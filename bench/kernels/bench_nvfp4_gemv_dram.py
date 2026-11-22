"""libstrix NVFP4 decode GEMV (split layout), weights streamed from DRAM:
each call routes 1-2 rows to 8 fresh experts of 128."""
import torch

from strix_vllm.inkling2_moe import gemm_nvfp4
from vllm.models.inkling2.amd.ops.triton_bmm import ExpertPlan

dev = torch.device("cuda")
for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
    E = 128
    weight = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    scale = (torch.rand(E, n, k // 16, device=dev) + 0.5).to(torch.float8_e4m3fn)
    amax = torch.rand(E, device=dev) + 0.5
    den = torch.tensor([2688.0], device=dev)
    plan = ExpertPlan(tile=16, max_ctas=8)
    for per_tile in (1, 2):
        sets = []
        for i in range(16):
            t = torch.zeros(19, dtype=torch.int32)
            for j in range(8):
                t[j] = (i * 8 + j) % E
                t[8 + j] = j * 16 + per_tile
            t[16] = 8
            sets.append(t.to(dev))
        act = torch.randn(plan.rows, k, device=dev).to(torch.bfloat16)
        fn = lambda t: gemm_nvfp4(act, weight, scale, amax, den, t, plan)  # noqa: E731
        for t in sets:
            fn(t)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(4):
            for t in sets:
                fn(t)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1000 / 64
        print(f"{name} rows/tile {per_tile} {us:7.1f} us {8 * n * (k // 2 + k // 16) / us / 1e3:6.1f} GB/s")
