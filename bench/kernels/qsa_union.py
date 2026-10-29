# SPDX-License-Identifier: MIT
"""strix qsa_union_attn_bf16 vs vLLM's Triton QSA sparse attention (the
RDNA-tuned per-row kernel), on real selections dumped from
Qwen3.8-Flash-Next (torch.save of logical_indices, [rows, 2051] int32, -1
padded) or synthetic 4-token block selections.

    bench/upstream.sh python bench/kernels/qsa_union.py --indices idx.pt
    bench/upstream.sh python bench/kernels/qsa_union.py --ctx 131072
"""

import argparse
import os

import strix  # noqa: F401
import torch
import triton


def main() -> None:
    from vllm.models.qwen4_exp.amd.ops import qsa as ops

    p = argparse.ArgumentParser()
    p.add_argument("--indices", help="dumped logical_indices; default synthetic")
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--page", type=int, default=256)
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    if a.indices:
        indices = torch.load(a.indices).to(dev).int().contiguous()
        ctx = int(indices.max().item()) + 1
    else:
        ctx = a.ctx
        blocks = torch.rand(a.rows, ctx // 4, device=dev).topk(512, dim=1).indices
        tok = (blocks[:, :, None] * 4 + torch.arange(4, device=dev)).reshape(a.rows, -1)
        indices = torch.cat([tok, tok.new_full((a.rows, 3), -1)], 1).int().contiguous()
    rows = indices.shape[0]
    pages = triton.cdiv(ctx, a.page)
    k_cache = torch.randn(pages + 4, a.page, 2, 256, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.randperm(pages + 4, device=dev)[:pages].int().contiguous()
    q = torch.randn(rows, 24, 256, device=dev, dtype=torch.bfloat16)
    token_to_req = torch.zeros(rows, device=dev, dtype=torch.int32)

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

    bt2 = block_table[None, :]
    ref = ops.qsa_sparse_paged_attention(q, k_cache, v_cache, indices, bt2, token_to_req)
    t_ref = timed(lambda: ops.qsa_sparse_paged_attention(q, k_cache, v_cache, indices, bt2, token_to_req))
    out = torch.empty_like(q)

    tokens = int(indices.max()) + 1 if os.environ.get("F16") == "1" else 0

    def new():
        torch.ops.strix.qsa_union_attn_bf16(
            out, q, k_cache, v_cache, indices, block_table, tokens
        )

    t_new = timed(new)
    d = (out.float() - ref.float()).abs()
    print(
        f"rows={rows} ctx={ctx}: triton per-row {t_ref:.1f} ms, strix union {t_new:.1f} ms "
        f"-> {t_ref / t_new:.2f}x; max|d|={d.max().item():.3g} mean|d|={d.mean().item():.2e}"
    )


if __name__ == "__main__":
    main()
