# SPDX-License-Identifier: MIT
"""Qwen3.8-Flash-Next QSA indexer scoring (qsa_mqa_paged): the tl.dot tile
kernel against vLLM's one-row-per-program kernel, on one prefill row chunk.

Rows are the last `--rows` query positions of a `--ctx`-token prompt, split
over two requests at `--split` so one tile straddles a request boundary;
`--split 0` is a single request.

    bench/upstream.sh python bench/kernels/qsa_indexer.py --ctx 131072
"""

import argparse

import torch
import triton


def main() -> None:
    from vllm.models.qwen4_exp.amd.ops import qsa as ops

    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--max-len", type=int, default=131072)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page", type=int, default=64)
    p.add_argument("--ratio", type=int, default=4)
    p.add_argument("--split", type=int, default=100)
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    width = triton.cdiv(a.max_len // a.ratio, a.page)
    pages = 2 * width + 4
    k_cache = torch.randn(pages, a.page, 1, a.head_dim, device=dev, dtype=torch.bfloat16)
    page_table = torch.randperm(pages, device=dev)[: 2 * width].int().view(2, width)
    page_table = page_table.contiguous()
    q = torch.randn(a.rows, a.heads, a.head_dim, device=dev, dtype=torch.bfloat16)
    positions = torch.arange(a.ctx - a.rows, a.ctx, device=dev, dtype=torch.int64)
    token_to_req = torch.zeros(a.rows, device=dev, dtype=torch.int32)
    token_to_req[a.split :] = 1
    seq_lens = torch.tensor([a.ctx - a.rows + a.split, a.ctx], device=dev, dtype=torch.int32)
    positions[: a.split] = torch.arange(a.ctx - a.rows, a.ctx - a.rows + a.split, device=dev)
    if a.split == 0:  # one request: the single-request kernel
        page_table, seq_lens = page_table[1:].contiguous(), seq_lens[1:].contiguous()
        token_to_req.zero_()

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

    def new():
        return ops.qsa_mqa_paged(q, k_cache, page_table, token_to_req, positions, seq_lens, a.ratio)

    rdna = ops._on_rdna
    ops._on_rdna = lambda: False
    try:
        ref_logits, ref_vis = new()
        t_old = timed(new)
    finally:
        ops._on_rdna = rdna
    logits, vis = new()
    t_new = timed(new)
    finite = torch.isfinite(ref_logits)
    same_mask = bool((finite == torch.isfinite(logits)).all())
    err = (logits[finite] - ref_logits[finite]).abs().max().item() if finite.any() else 0.0
    print(
        f"rows={a.rows} ctx={a.ctx} columns={ref_logits.shape[1]}: "
        f"old {t_old:.2f} ms, dot {t_new:.2f} ms ({t_old / t_new:.1f}x); "
        f"visible equal={bool((vis == ref_vis).all())} mask equal={same_mask} "
        f"max|d|={err:.3g} (|ref| max {ref_logits[finite].abs().max().item():.3g})"
    )


if __name__ == "__main__":
    main()
