"""Correctness and timing of the NVFP4 expert GEMMs on Inkling 2 Mini shapes.

    IMAGE=strix-vllm:inkling MODELS=/mnt/8tb-usb/models bench/container/run.sh \
        python bench/kernels/bench_nvfp4_bmm.py
"""

import sys

import torch

sys.path.insert(0, "/work/bench/kernels")
from nvfp4_bmm_v2 import bmm_nvfp4_dequant, bmm_nvfp4_v4  # noqa: E402

from vllm.models.inkling2.amd.ops.triton_bmm import ExpertPlan, bmm_nvfp4  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
dev = torch.device("cuda")


def decode(weight, scale, amax, denominator):
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    values = E2M1[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    return values * scale.float().repeat_interleave(16, -1) * (amax / denominator)[:, None, None]


def setup(n, k, tile, rows_per_tile, num_tiles=8, num_experts=128):
    g = torch.Generator().manual_seed(0)
    weight = torch.randint(0, 256, (num_experts, n, k // 2), dtype=torch.uint8, generator=g)
    scale = (torch.rand(num_experts, n, k // 16, generator=g) * 2 + 0.1).to(torch.float8_e4m3fn)
    amax = torch.rand(num_experts, generator=g) + 0.5
    denominator = torch.tensor([6.0 * 448.0])
    experts = torch.randperm(num_experts, generator=g)[:min(num_tiles, num_experts)].tolist()
    plan = ExpertPlan(tile=tile, max_ctas=num_tiles)
    tiles = torch.zeros(2 * num_tiles + 1, dtype=torch.int32)
    for t, e in enumerate(experts):
        tiles[t] = e
        tiles[num_tiles + t] = t * tile + rows_per_tile
    tiles[2 * num_tiles] = num_tiles
    act = (torch.randn(plan.rows, k, generator=g) * 2).to(torch.float8_e4m3fn)
    act_scale = torch.full((plan.rows * k // 32,), 127, dtype=torch.uint8)
    host = (act, act_scale, weight, scale, amax, denominator, tiles)
    return host, tuple(t.to(dev) for t in host), plan, experts


def time_us(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


if __name__ != "__main__":
    pass
else:
  for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
      for tile, rows, num_tiles in ((128, 128, 128), (256, 256, 128)):
          host, args, plan, experts = setup(n, k, tile, rows, num_tiles=num_tiles)
          act, _, weight, scale, amax, denominator, _ = host
          w = decode(weight, scale, amax, denominator)
          ref = torch.cat([act[t * tile : t * tile + rows].float() @ w[e].T for t, e in enumerate(experts)])
          for label, fn in (
              ("v1", lambda: bmm_nvfp4(*args, plan)),
              ("v4 64x64 w4", lambda: bmm_nvfp4_v4(*args, plan, bm=64, bn=64, num_warps=4)),
              *[
                  (f"deq {bm}x{bn} w{nw}", (lambda bm=bm, bn=bn, nw=nw: bmm_nvfp4_dequant(*args, plan, bm=bm, bn=bn, num_warps=nw)))
                  for bm, bn, nw in ((64, 128, 4), (64, 64, 4), (128, 128, 8))
              ],
          ):
              out = fn().float().cpu()
              got = torch.cat([out[t * tile : t * tile + rows] for t in range(len(experts))])
              err = ((got - ref).abs().max() / ref.abs().max()).item()
              flops = 2 * len(experts) * rows * n * k
              us = time_us(fn, iters=10)
              print(f"{name} tile{tile} rows{rows}x{len(experts)} {label:18s} {us:9.1f} us  {flops / us / 1e6:6.1f} TFLOPS  rel_err {err:.2e}", flush=True)
