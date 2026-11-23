"""Launch sweep of the Inkling2 (ROCm port) split-KV decode attention on the
real tensors of long-context decode: the first decode call of each layer
kind and batch size times `_rel_attention_kernel` over split counts, key
block sizes and warps (scratch partials), reporting the max difference of
the merged output against the current launch and the time to stream the KV
at 210 GB/s.

    IMAGE=strix-vllm:inkling MODELS=... VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
        STRIX_WORDS=24000 STRIX_BATCHES=1,4 \\
        bench/container/run.sh python bench/model/inkling_decode_attn_sweep.py
"""

import itertools
import os
import random

import torch
from vllm import LLM, SamplingParams
from vllm.models.inkling2.amd.ops import attention as attn_ops
from vllm.platforms.rocm import on_gfx1x
from vllm.triton_utils import triton

WORDS = int(os.environ.get("STRIX_WORDS", "24000"))
done = set()
orig_launch = attn_ops._launch


def timed(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1000


def merge(o, l, S, T):
    H = o.shape[1]
    o = o.view(S, T, H, -1)
    l = l.view(H, S, T).permute(1, 2, 0)  # S,T,H
    m = l.max(0).values
    w = torch.exp2(l - m)
    return (o * w[..., None]).sum(0) / w.sum(0)[..., None]


def sweep(
    q,
    key_cache,
    value_cache,
    bias,
    out,
    lse,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale,
    window,
    draft,
    num_splits=0,
):
    split = out is None or lse is None
    ctx = int(cache_seqlens.max())
    key = (window, cache_seqlens.shape[0], draft)
    if split and not draft and ctx > 1000 and key not in done:
        done.add(key)
        num_tokens, num_heads, head_dim = q.shape
        page, num_kv_heads = key_cache.shape[1:3]
        group = num_heads // num_kv_heads
        block_h = triton.next_power_of_2(group)
        num_reqs = cache_seqlens.shape[0]
        block_m = max(16, triton.next_power_of_2(max_seqlen_q * block_h))
        block_q = max(1, block_m // block_h)

        def run(ns, bn, nw):
            o = q.new_empty((ns * num_tokens, num_heads, head_dim), dtype=torch.float32)
            l = q.new_empty((num_heads, ns * num_tokens), dtype=torch.float32)

            def go():
                attn_ops._rel_attention_kernel[(ns, num_kv_heads, num_reqs)](
                    q,
                    key_cache,
                    value_cache,
                    bias,
                    o,
                    l,
                    block_table,
                    cache_seqlens,
                    cu_seqlens_q,
                    softmax_scale * attn_ops._LOG2E,
                    ns,
                    num_tokens,
                    q.stride(0),
                    q.stride(1),
                    key_cache.stride(0),
                    key_cache.stride(1),
                    key_cache.stride(2),
                    value_cache.stride(0),
                    value_cache.stride(1),
                    value_cache.stride(2),
                    bias.stride(0),
                    bias.stride(1),
                    o.stride(0),
                    o.stride(1),
                    l.stride(0),
                    block_table.stride(0),
                    GROUP=group,
                    D=head_dim,
                    PAGE=page,
                    E_PAD=bias.shape[2],
                    WINDOW=window,
                    DRAFT=draft,
                    SPLIT=True,
                    BLOCK_Q=block_q,
                    BLOCK_H=block_h,
                    BLOCK_N=bn,
                    HEAD_SPLIT=False,
                    num_warps=nw,
                    num_stages=1,
                )

            return go, o, l

        rdna = on_gfx1x()
        current = (
            num_splits,
            attn_ops._RDNA_DECODE_BLOCK_N if rdna else attn_ops._BLOCK_N,
            attn_ops._RDNA_DECODE_WARPS if rdna else 4,
        )
        go, o, l = run(*current)
        t_cur = timed(go)
        ref = merge(o, l, num_splits, num_tokens)
        res = []
        for ns, bn, nw in itertools.product(
            (2, 3, 5, 8, 10, 20, 40), (16, 32, 64), (1, 2, 4)
        ):
            if bn > page:
                continue
            try:
                go, o, l = run(ns, bn, nw)
                t = timed(go)
            except Exception:
                continue
            err = (merge(o, l, ns, num_tokens) - ref).abs().max().item()
            res.append((t, (ns, bn, nw), err))
        res.sort()
        span = ctx if window < 0 else min(ctx, window + 1)
        ideal = (
            num_reqs
            * span
            * num_kv_heads
            * head_dim
            * 2
            * key_cache.element_size()
            / 210e3
        )
        print(
            f"DATTN window {window} reqs {num_reqs} ctx {ctx} page {page} block_m {block_m} current {current}: {t_cur:.1f} us (ideal at 210 GB/s {ideal:.1f} us)",
            flush=True,
        )
        for t, cfg, err in res[:8]:
            print(
                f"DATTN window {window} reqs {num_reqs}  {t:8.1f} us splits/bn/warps {cfg} err {err:.1e}",
                flush=True,
            )
    return orig_launch(
        q,
        key_cache,
        value_cache,
        bias,
        out,
        lse,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        max_seqlen_q,
        softmax_scale,
        window,
        draft,
        num_splits,
    )


attn_ops._launch = sweep
llm = LLM(
    "/models/Inkling-2-Mini",
    max_model_len=40960,
    max_num_seqs=4,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)
words = ["the", "a", "of", "lighthouse", "keeper", "river", "stone", "engine", "market", "silver", "window", "garden", "north", "letter"]
random.seed(1)
for b in [int(x) for x in os.environ.get("STRIX_BATCHES", "1,4").split(",")]:
    msgs = [
        [
            {
                "role": "user",
                "content": f"({i}) Summarize this: "
                + " ".join(random.choice(words) for _ in range(WORDS)),
            }
        ]
        for i in range(b)
    ]
    llm.chat(msgs, SamplingParams(temperature=0, max_tokens=40, ignore_eos=True))
