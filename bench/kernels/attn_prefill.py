# SPDX-License-Identifier: MIT
"""strix attn_prefill vs vLLM's Triton unified_attention (prefill).

27B: 24 q / 4 kv heads, KV block 1056; 35B: 16 / 2, block 784 (vLLM enlarges
the attention block to the GDN state page on these hybrids).
"""

import argparse
import sys
from pathlib import Path

import strix
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
from test_attn_prefill import build


def time_us(fn, iters: int) -> float:
    for _ in range(2):
        fn()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3


def main() -> None:
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.3)
    print(
        f"{'model':>5} {'case':>22} {'triton us':>10} {'strix us':>9} "
        f"{'speedup':>7} {'TFLOPS':>6}"
    )
    for model, hq, hkv, bs, hd in (
        ("27b", 24, 4, 1056, 256),
        ("35b", 16, 2, 784, 256),
        ("gemma", 32, 4, 16, 512),  # Gemma-4 31B full-attention layers
        ("lfm", 32, 8, 16, 64),  # LFM2.5-24B-A2B
    ):
        for name, q_lens, ctx in (
            ("8k fresh", [8192], [0]),
            ("2k chunk @ 16k ctx", [2048], [16384]),
            ("4x 2k fresh", [2048] * 4, [0] * 4),
        ):
            q, kc, vc, cu, seq_lens, table = build(q_lens, ctx, hq, hkv, bs, hd=hd)
            cu32, sl32 = cu.int(), seq_lens.int()
            out = torch.empty_like(q)
            scale = hd**-0.5

            def triton(
                q=q,
                kc=kc,
                vc=vc,
                cu32=cu32,
                sl32=sl32,
                table=table,
                out=out,
                q_lens=q_lens,
                ctx=ctx,
                scale=scale,
            ):
                unified_attention(
                    q=q,
                    k=kc,
                    v=vc,
                    out=out,
                    cu_seqlens_q=cu32,
                    max_seqlen_q=max(q_lens),
                    seqused_k=sl32,
                    max_seqlen_k=max(a + b for a, b in zip(q_lens, ctx)),
                    softmax_scale=scale,
                    causal=True,
                    window_size=(-1, -1),
                    block_table=table,
                    softcap=0,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                )

            def ours(
                q=q,
                kc=kc,
                vc=vc,
                cu=cu,
                seq_lens=seq_lens,
                table=table,
                out=out,
                scale=scale,
            ):
                strix.attn_prefill(out, q, kc, vc, cu, seq_lens, table, scale)

            t_tri, t_ours = time_us(triton, args.iters), time_us(ours, args.iters)
            flops = sum(
                4 * hq * hd * ql * (c + (ql + 1) / 2) for ql, c in zip(q_lens, ctx)
            )
            print(
                f"{model:>5} {name:>22} {t_tri:10.0f} {t_ours:9.0f} "
                f"{t_tri / t_ours:6.2f}x {flops / t_ours / 1e6:6.1f}",
                flush=True,
            )

    # Flint: head 128, 32 q / 4 kv heads, learned sinks; full layers and the
    # 513-key sliding window (window_left 512) of 36 of its 49 layers. FLOPs
    # count only the keys each query sees.
    for name, q_lens, ctx in (("8k fresh", [8192], [0]), ("8k chunk @ 32k", [8192], [24576])):
        for win in (-1, 512):
            q, kc, vc, cu, seq_lens, table = build(q_lens, ctx, 32, 4, 16, hd=128)
            sinks = torch.randn(32, device="cuda").float()
            cu32, sl32 = cu.int(), seq_lens.int()
            out = torch.empty_like(q)
            scale = 128**-0.5

            def triton(q=q, kc=kc, vc=vc, cu32=cu32, sl32=sl32, table=table, out=out,
                       q_lens=q_lens, ctx=ctx, scale=scale, win=win, sinks=sinks):
                unified_attention(
                    q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu32,
                    max_seqlen_q=max(q_lens), seqused_k=sl32,
                    max_seqlen_k=max(a + b for a, b in zip(q_lens, ctx)),
                    softmax_scale=scale, causal=True, window_size=(win, 0) if win >= 0 else (-1, -1),
                    block_table=table, softcap=0, q_descale=None, k_descale=None,
                    v_descale=None, sinks=sinks,
                )

            def ours(q=q, kc=kc, vc=vc, cu=cu, seq_lens=seq_lens, table=table, out=out,
                     scale=scale, win=win, sinks=sinks):
                strix.attn_prefill(out, q, kc, vc, cu, seq_lens, table, scale, 1, win,
                                   sinks=sinks)

            t_tri, t_ours = time_us(triton, args.iters), time_us(ours, args.iters)
            seen = sum(
                (min(c + i + 1, win + 1) if win >= 0 else c + i + 1)
                for ql, c in zip(q_lens, ctx) for i in range(ql)
            )
            flops = 4 * 32 * 128 * seen
            label = f"flint {'win513' if win >= 0 else 'full'}"
            print(
                f"{label:>5} {name:>22} {t_tri:10.0f} {t_ours:9.0f} "
                f"{t_tri / t_ours:6.2f}x {flops / t_ours / 1e6:6.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
