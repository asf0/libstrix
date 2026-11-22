"""libstrix Inkling2 NVFP4 grouped GEMM (bf16 rows) at prefill shapes:
128 experts x ~320 rows (a 5k-token step), tile 256 as the plan picks."""
import itertools

import torch
import triton

from strix_vllm import inkling2_moe as M
from vllm.models.inkling2.amd.ops.triton_bmm import ExpertPlan

dev = torch.device("cuda")
E, ROWS, TILE = 128, 320, 256
for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
    weight = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    scale = (torch.rand(E, n, k // 16, device=dev) + 0.5).to(torch.float8_e4m3fn)
    amax = torch.rand(E, device=dev) + 0.5
    den = torch.tensor([2688.0], device=dev)
    tiles_per_expert = triton.cdiv(ROWS, TILE)
    max_ctas = E * tiles_per_expert
    plan = ExpertPlan(tile=TILE, max_ctas=max_ctas)
    tiles = torch.zeros(2 * max_ctas + 3, dtype=torch.int32)
    t = 0
    for e in range(E):
        left = ROWS
        for _ in range(tiles_per_expert):
            tiles[t] = e
            tiles[max_ctas + t] = t * TILE + min(left, TILE)
            left -= TILE
            t += 1
    tiles[2 * max_ctas] = max_ctas
    tiles = tiles.to(dev)
    act = torch.randn(plan.rows, k, device=dev).to(torch.bfloat16)
    flops = 2 * E * ROWS * n * k

    def run(bm, bn, nw, ns, out):
        sub = TILE // bm
        M._nvfp4_bmm_bf16_kernel[(n // bn, max_ctas * sub)](
            act, weight, scale, amax, den, out, tiles, max_ctas, N=n, K=k,
            TILE=TILE, BM=bm, BN=bn, num_warps=nw, num_stages=ns)

    ref = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=dev)
    run(128, 128, 8, 2, ref)
    for bm, bn, nw, ns in itertools.product((64, 128, 256), (64, 128, 256), (4, 8), (1, 2, 3)):
        out = torch.empty_like(ref)
        try:
            run(bm, bn, nw, ns, out)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(3):
                run(bm, bn, nw, ns, out)
            e.record()
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001
            continue
        ms = s.elapsed_time(e) / 3
        rows = [r for e_ in range(0, E, 17) for r in (e_ * tiles_per_expert * TILE,)]
        err = (out[rows].float() - ref[rows].float()).abs().max().item()
        print(f"{name} {bm}x{bn} w{nw} s{ns} {ms:7.2f} ms {flops / ms / 1e9:6.1f} TFLOPS err {err:.1e}", flush=True)
