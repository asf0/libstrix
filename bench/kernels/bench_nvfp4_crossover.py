"""libstrix NVFP4 expert GEMM at decode / verify / drafter sizes: the per-row
GEMV against the grouped dot kernel, random top-8 routing over 128 experts,
weights streamed from DRAM (a fresh layer per call)."""
import torch
import triton

from strix_vllm import inkling2_moe as M
from vllm.models.inkling2.amd.ops.triton_bmm import expert_plan

dev = torch.device("cuda")
E, TOPK, LAYERS = 128, 8, 6
for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
    weights = [torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev) for _ in range(LAYERS)]
    scale = (torch.rand(E, n, k // 16, device=dev) + 0.5).to(torch.float8_e4m3fn)
    amax = torch.rand(E, device=dev) + 0.5
    den = torch.tensor([2688.0], device=dev)
    for skew, tokens in [(sk, t) for sk in (0.0, 1.2) for t in (1, 2, 4, 8, 16, 25)]:
        plan = expert_plan(tokens, TOPK, E)
        g = torch.Generator().manual_seed(tokens)
        counts = torch.zeros(E, dtype=torch.int64)
        # skew > 0: Zipf-like expert popularity (one sequence's tokens share experts)
        popularity = (torch.arange(E) + 1.0) ** -skew
        popularity = popularity[torch.randperm(E, generator=g)]
        for _ in range(tokens):
            counts[torch.multinomial(popularity, TOPK, generator=g)] += 1
        tiles = torch.zeros(2 * plan.max_ctas + 3, dtype=torch.int32)
        t = 0
        for e in range(E):
            left = int(counts[e])
            while left > 0:
                tiles[t] = e
                tiles[plan.max_ctas + t] = t * plan.tile + min(left, plan.tile)
                left -= plan.tile
                t += 1
        tiles[2 * plan.max_ctas] = t
        tiles = tiles.to(dev)
        act = torch.randn(plan.rows, k, device=dev).to(torch.bfloat16)
        res = {}
        out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=dev)

        def tile_gemv(w):
            M._nvfp4_gemv_tile_kernel[(n // 16, plan.max_ctas)](
                act, w, scale.view(torch.uint8), amax, den, out, tiles, plan.max_ctas,
                N=n, K=k, TILE=plan.tile, BN=16, BK=128, num_warps=1)
            return out

        def hip_gemv(w):
            torch.ops.strix.nvfp4_moe_gemv(out, act, w, scale, amax, den, tiles, plan.max_ctas, plan.tile)
            return out

        for label, fn in (("tile", tile_gemv), ("hip", hip_gemv)):
            for w in weights:
                fn(w)
            torch.cuda.synchronize()
            s, e_ = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(3):
                for w in weights:
                    fn(w)
            e_.record()
            torch.cuda.synchronize()
            res[label] = s.elapsed_time(e_) * 1000 / (3 * LAYERS)
        used = int((counts > 0).sum())
        ref = tile_gemv(weights[0]).clone()
        got = hip_gemv(weights[0])
        valid = torch.cat([torch.arange(i * plan.tile, int(tiles[plan.max_ctas + i])) for i in range(int(tiles[2 * plan.max_ctas]))]).to(dev)
        err = ((got[valid].float() - ref[valid].float()).abs().max() / ref[valid].float().abs().max()).item()
        print(f"{name} skew {skew} tokens {tokens:4d} experts {used:3d} max/expert {int(counts.max()):2d} "
              f"triton tile {res['tile']:7.1f} hip {res['hip']:7.1f} us err {err:.0e} "
              f"(experts once at 165 GB/s: {used * n * (k // 2 + k // 16) / 165e3:6.1f} us)", flush=True)
