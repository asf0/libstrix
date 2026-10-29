# SPDX-License-Identifier: MIT
"""Qwen3.8-Flash-Next QSA sparse attention (vLLM's Triton kernel) at prefill
shapes, sweeping the launch parameters vLLM hard-codes (tuned on GB300).

Each query row attends to its own TOPK selected tokens (indexer budget 2048 +
compress_ratio - 1): 512 random 4-token blocks of the visible context, like
the indexer's block selection, plus padding -1 past the visible length.

    bench/upstream.sh python bench/kernels/qsa_sparse.py
"""

import argparse
import itertools

import torch
import triton


def main() -> None:
    from vllm.models.qwen4_exp.amd.ops.qsa import (
        _qsa_sparse_paged_gqa_splitk_kernel,
        qsa_sparse_paged_attention,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--ctx", type=int, default=16384, help="visible tokens of the request")
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--page", type=int, default=256)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--ratio", type=int, default=4)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--block-n", default="16,32,64")
    p.add_argument("--warps", default="2,4,8")
    p.add_argument("--stages", default="1,2")
    p.add_argument("--heads-first", default="0", help="0,1: grid (kv, rows) order")
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    pages = triton.cdiv(a.ctx, a.page)
    k_cache = torch.randn(pages + 8, a.page, a.kv_heads, a.head_dim, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    # Scatter the request's pages over the cache like a real allocator.
    block_table = (torch.randperm(pages + 8, device=dev)[:pages]).int()[None, :].contiguous()
    q = torch.randn(a.rows, a.heads, a.head_dim, device=dev, dtype=torch.bfloat16)
    width = a.budget + a.ratio - 1
    n_blocks = a.budget // a.ratio
    blocks = torch.rand(a.rows, a.ctx // a.ratio, device=dev).topk(n_blocks, dim=1).indices
    blocks = blocks.sort(dim=1).values
    tok = (blocks[:, :, None] * a.ratio + torch.arange(a.ratio, device=dev)).reshape(a.rows, -1)
    tail = torch.full((a.rows, width - tok.shape[1]), -1, device=dev, dtype=tok.dtype)
    indices = torch.cat([tok, tail], dim=1).int().contiguous()
    token_to_req = torch.zeros(a.rows, device=dev, dtype=torch.int32)

    def timed(fn) -> float:
        fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(a.reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / a.reps

    ref = qsa_sparse_paged_attention(q, k_cache, v_cache, indices, block_table, token_to_req)
    base = timed(lambda: qsa_sparse_paged_attention(q, k_cache, v_cache, indices, block_table, token_to_req))
    gb = a.rows * a.kv_heads * a.budget * a.head_dim * 2 * 2 / 1e9
    print(f"rows={a.rows} ctx={a.ctx} vLLM default: {base:.1f} ms ({gb / base * 1e3:.0f} GB/s gathered)")

    group = a.heads // a.kv_heads
    block_m = triton.next_power_of_2(group)
    out = torch.empty_like(q)
    for bn, w, st, hf in itertools.product(
        *(map(int, s.split(",")) for s in (a.block_n, a.warps, a.stages, a.heads_first))
    ):
        num_tiles = triton.cdiv(width, bn)

        def launch(bn=bn, w=w, st=st, hf=hf, num_tiles=num_tiles):
            grid = (a.kv_heads, a.rows, 1) if hf else (a.rows, a.kv_heads, 1)
            _qsa_sparse_paged_gqa_splitk_kernel[grid](
                q, k_cache, v_cache, indices, block_table, token_to_req,
                out, out, out,
                q.stride(0), q.stride(1),
                k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
                indices.stride(0), block_table.stride(0),
                out.stride(0), out.stride(1),
                a.rows, k_cache.shape[0], 1,
                TOPK=width, PAGE_SIZE=a.page, PAGE_TABLE_WIDTH=block_table.shape[1],
                GROUP_SIZE=group, HEAD_DIM=a.head_dim, NUM_QUERY_HEADS=a.heads,
                NUM_SPLITS=1, NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=bn,
                HEADS_FIRST=bool(hf), num_warps=w, num_stages=st,
            )

        try:
            ms = timed(launch)
        except Exception as exc:  # out of LDS / registers
            print(f"bn={bn:3d} warps={w} stages={st} hf={hf}: failed ({type(exc).__name__})")
            continue
        err = (out.float() - ref.float()).abs().max().item()
        print(f"bn={bn:3d} warps={w} stages={st} hf={hf}: {ms:7.1f} ms  {base / ms:4.2f}x  max|d|={err:.3g}")


if __name__ == "__main__":
    main()
