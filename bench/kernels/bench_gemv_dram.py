"""NVFP4 decode GEMV with weights streamed from DRAM: each call routes to 8
fresh experts of 128 (340 MB of w13), beyond the 32 MB Infinity Cache."""
import sys
import torch
sys.path.insert(0, "/work/bench/kernels")
from nvfp4_bmm_v2 import bmm_nvfp4_gemv_v6  # noqa: E402
from vllm.models.inkling2.amd.ops.triton_bmm import ExpertPlan  # noqa: E402

try:  # the port's NVFP4 BMM; removed from the strix-vllm fork
    from vllm.models.inkling2.amd.ops.triton_bmm import bmm_nvfp4  # noqa: E402
except ImportError:
    bmm_nvfp4 = None

dev = torch.device("cuda")
for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
    E = 128
    weight = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    scale = (torch.rand(E, n, k // 16, device=dev) + 0.5).to(torch.float8_e4m3fn)
    amax = torch.rand(E, device=dev) + 0.5
    den = torch.tensor([2688.0], device=dev)
    plan = ExpertPlan(tile=16, max_ctas=8)
    tile_sets = []
    for i in range(16):
        t = torch.zeros(17, dtype=torch.int32)
        for j in range(8):
            t[j] = (i * 8 + j) % E
            t[8 + j] = j * 16 + 1
        t[16] = 8
        tile_sets.append(t.to(dev))
    act = torch.randn(plan.rows, k, device=dev).to(torch.bfloat16)
    if bmm_nvfp4 is not None:
        ref = bmm_nvfp4(act, None, weight, scale, amax, den, tile_sets[0], plan)
        variants = [("current", lambda t: bmm_nvfp4(act, None, weight, scale, amax, den, t, plan))]
    else:  # reference: the first v6 variant
        ref = bmm_nvfp4_gemv_v6(act, weight, scale, amax, den, tile_sets[0], plan)
        variants = []
    for bn in (16, 32, 64):
        for bk in (256, 512, 1024):
            for nw in (2, 4, 8):
                variants.append((f"v6 bn{bn} bk{bk} w{nw}", (lambda t, bn=bn, bk=bk, nw=nw: bmm_nvfp4_gemv_v6(act, weight, scale, amax, den, t, plan, bn=bn, bk=bk, nw=nw))))
    bytes_per_call = 8 * n * (k // 2 + k // 16)
    for label, fn in variants:
        try:
            out = fn(tile_sets[0])
        except Exception as ex:  # noqa: BLE001
            print(f"{name} {label}: {type(ex).__name__}")
            continue
        rows = [j * 16 for j in range(8)]
        err = ((out[rows].float() - ref[rows].float()).abs().max() / ref[rows].float().abs().max()).item()
        for t in tile_sets:
            fn(t)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        iters = 0
        for _ in range(4):
            for t in tile_sets:
                fn(t)
                iters += 1
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1000 / iters
        print(f"{name} {label:22s} {us:7.1f} us {bytes_per_call / us / 1e3:6.1f} GB/s err {err:.1e}", flush=True)
