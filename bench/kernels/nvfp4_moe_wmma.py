# SPDX-License-Identifier: MIT
"""strix nvfp4_moe_wmma at Inkling 2 Mini prefill shapes (8k-token chunk:
128 experts x 512 routed rows, w13 2304x2304 and w2 2304x1152), under CUDA
graphs, checked against a dequantized fp32 GEMM on two experts.

    bench/container/run.sh python bench/kernels/nvfp4_moe_wmma.py [--save out.pt]
"""

import argparse

import torch

from strix_vllm.inkling2_moe import pretile_nvfp4

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def dequant(w_u8, s_e4m3, amax_e, den):
    """[N, K/2] e2m1 pairs, [N, K/16] e4m3 -> fp32 [N, K]."""
    lo, hi = (w_u8 & 15).long(), (w_u8 >> 4).long()
    codes = torch.stack([lo, hi], -1).reshape(w_u8.shape[0], -1)
    vals = E2M1.to(w_u8.device)[codes]
    s = s_e4m3.float().repeat_interleave(16, 1)
    return vals * s * (amax_e / den)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=512, help="routed rows per expert")
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--save")
    a = p.parse_args()
    import strix  # noqa: F401

    dev = "cuda"
    E = 128
    saved = {}
    for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
        g = torch.Generator(device=dev).manual_seed(1)
        w = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev, generator=g)
        s = (torch.rand(E, n, k // 16, device=dev, generator=g) * 0.5 + 0.25).to(torch.float8_e4m3fn)
        amax = torch.rand(E, device=dev, generator=g) + 0.5
        den = torch.tensor([2688.0], device=dev)
        wt, st = pretile_nvfp4(w, s)
        per = -(-a.rows // a.tile)
        max_ctas = E * per
        tiles = torch.zeros(2 * max_ctas + 3, dtype=torch.int32)
        t = 0
        for e in range(E):
            left = a.rows
            for _ in range(per):
                tiles[t] = e
                tiles[max_ctas + t] = t * a.tile + min(left, a.tile)
                left -= a.tile
                t += 1
        tiles[2 * max_ctas] = max_ctas
        tiles = tiles.to(dev)
        act = (torch.randn(max_ctas * a.tile, k, device=dev, generator=g) * 0.5).bfloat16()
        out = torch.empty(max_ctas * a.tile, n, dtype=torch.bfloat16, device=dev)

        def run():
            torch.ops.strix.nvfp4_moe_wmma(out, act, wt, st, amax, den, tiles, max_ctas, a.tile,
                                           None, 0.0, 0.0, None)

        run()
        torch.cuda.synchronize()
        # Reference on experts 0 and 77, their first rows.
        errs = []
        for e in (0, 77):
            r0 = e * per * a.tile
            ref = act[r0 : r0 + 64].float() @ dequant(w[e], s[e], amax[e], den[0]).t()
            got = out[r0 : r0 + 64].float()
            errs.append(((got - ref).norm() / ref.norm()).item())
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(5):
                run()
        gr.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(4):
            gr.replay()
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / 20
        flops = 2 * E * a.rows * n * k
        print(f"{name} {n}x{k}: {ms:7.3f} ms {flops / ms / 1e9:6.1f} TFLOPS  rel err {max(errs):.2e}", flush=True)
        saved[name] = out.clone().cpu()
    if a.save:
        torch.save(saved, a.save)


if __name__ == "__main__":
    main()
