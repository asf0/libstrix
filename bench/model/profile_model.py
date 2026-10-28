# SPDX-License-Identifier: MIT
"""Phase-separated vLLM run for per-kernel profiling under rocprofv3.

Each phase is preceded by an idle gap so a kernel trace splits into one burst
per phase; the phase labels are written to --labels for analyze_trace.py.
Decode is measured as the difference of a 1-token and a (1 + --decode-steps)
token run of the same batch, with prefix caching off, which cancels prefill.
Every printed number is also recorded in the results DB (bench/results.py).

    rocprofv3 --kernel-trace --output-format csv -d out -- \\
        python bench/model/profile_model.py --model ... --labels out/labels.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/results.py
import results  # noqa: E402
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Upper bound on sampled token ids; lowered to the tokenizer's size for
# models with smaller vocabularies (Qwen ids stay as before).
VOCAB_HI = 150000


def prompts(n: int, length: int, rng: random.Random) -> list[TokensPrompt]:
    # Repeated short phrases give an MTP drafter something to accept.
    vocab = [rng.randrange(1000, VOCAB_HI) for _ in range(64)]
    return [
        TokensPrompt(prompt_token_ids=[rng.choice(vocab) for _ in range(length)])
        for _ in range(n)
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--labels", help="phase labels json for analyze_trace.py")
    p.add_argument("--db", help="results DB (default profiles/results.db)")
    p.add_argument("--prefill", default="8192", help="'' skips the prefill phases")
    p.add_argument("--decode-batches", default="1,4,16")
    p.add_argument(
        "--decode-ctx",
        default="2048",
        help="one context length, or a comma list to sweep depths in one run",
    )
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument(
        "--decode-long-only",
        action="store_true",
        help="skip the 1-token short pass: the DECODE line already comes from "
        "per-request timestamps, and the short pass repeats the whole prefill",
    )
    p.add_argument(
        "--combined",
        help="comma list of depths: one request per depth gives both its "
        "prefill (TTFT) and its decode step, instead of separate prefill and "
        "short/long decode passes that each repeat the prefill",
    )
    p.add_argument("--mtp", type=int, default=0, help="MTP draft tokens, 0 = off")
    p.add_argument("--gap", type=float, default=4.0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="vLLM KV cache dtype, e.g. int8_per_token_head (the served "
        "teacher-corpus config) or bf16",
    )
    p.add_argument(
        "--mamba-ssm-cache-dtype",
        default="auto",
        help="GDN recurrent-state dtype; the decode kernel is bound by the "
        "fp32 state traffic, so bfloat16 halves it",
    )
    p.add_argument(
        "--torch-profile",
        help="trace with vLLM's torch profiler into this dir instead of an "
        "external rocprofv3 (convert with torch_trace_to_csv.py)",
    )
    p.add_argument(
        "--profile-iters",
        default="0,5",
        help="DELAY,ACTIVE engine steps for --torch-profile: skip DELAY steps "
        "after the profile starts, then record ACTIVE",
    )
    args = p.parse_args()
    run_id = results.start_run(__file__, args.model, db=args.db)

    def rec(phase, metric, value, label, depth, batch):
        results.record(
            run_id,
            phase,
            metric,
            value,
            label=label,
            depth=depth,
            batch=batch,
            kv_dtype=args.kv_cache_dtype,
            db=args.db,
        )

    def write_labels() -> None:
        if args.labels:
            with open(args.labels, "w") as f:
                json.dump({"labels": labels, "decode_steps": args.decode_steps}, f)

    llm = LLM(
        model=args.model,
        language_model_only=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=max(int(b) for b in args.decode_batches.split(",")),
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        attention_backend="TRITON_ATTN",
        kv_cache_dtype=args.kv_cache_dtype,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        # Per-request token timestamps (RequestOutput.metrics) for DECODE lines.
        disable_log_stats=False,
        enforce_eager=args.enforce_eager,
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": args.mtp} if args.mtp else None
        ),
        profiler_config=(
            {
                "profiler": "torch",
                "torch_profiler_dir": args.torch_profile,
                "torch_profiler_with_stack": False,
                "delay_iterations": int(args.profile_iters.split(",")[0]),
                "active_iterations": int(args.profile_iters.split(",")[1]),
            }
            if args.torch_profile
            else None
        ),
    )
    global VOCAB_HI
    VOCAB_HI = min(VOCAB_HI, len(llm.get_tokenizer()))
    rng = random.Random(0)
    llm.generate(prompts(1, 512, rng), SamplingParams(max_tokens=8), use_tqdm=False)

    labels: list[str] = []
    if args.torch_profile:
        llm.start_profile()

    def phase(
        label: str, batch: list[TokensPrompt], max_tokens: int, depth: int, bs: int = 1
    ) -> None:
        time.sleep(args.gap)
        sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate(batch, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        labels.append(label)
        print(f"PHASE {label}: {dt:.3f} s", flush=True)
        kind = next((k for k in ("prefill", "decode") if label.startswith(k)), "e2e")
        rec(kind, "phase_s", dt, label, depth, bs)
        # Decode step time of the request that started decoding last: its
        # steps all run after every prefill chunk of the batch (earlier ones
        # also count steps shared with other requests' chunked prefill).
        # long - short cancels prefill too, but slow prefills vary by seconds.
        done = [
            o for o in outs if o.metrics is not None and len(o.outputs[0].token_ids) > 1
        ]
        if done:
            o = max(done, key=lambda o: o.metrics.first_token_ts)
            step = (o.metrics.last_token_ts - o.metrics.first_token_ts) / (
                len(o.outputs[0].token_ids) - 1
            )
            print(f"DECODE {label}: {step * 1e3:.2f} ms/step", flush=True)
            print(f"TTFT {label}: {o.metrics.first_token_latency:.3f} s", flush=True)
            rec("decode", "decode_ms", step * 1e3, label, depth, bs)
            rec("prefill", "ttft_s", o.metrics.first_token_latency, label, depth, bs)

    if args.combined:
        for depth in (int(x) for x in args.combined.split(",")):
            for rep in range(2 if depth < 65536 else 1):
                phase(
                    f"depth{depth}#{rep}",
                    prompts(1, depth, rng),
                    1 + args.decode_steps,
                    depth,
                )
        time.sleep(args.gap)
        write_labels()
        return

    for length in (int(x) for x in args.prefill.split(",") if x):
        # Two reps smooth the shallow depths; a 64k+ prefill is expensive
        # enough that one measured pass is the right trade.
        for rep in range(2 if length < 65536 else 1):
            phase(f"prefill{length}#{rep}", prompts(1, length, rng), 1, length)
    ctxs = [int(x) for x in str(args.decode_ctx).split(",")]
    for bs in (int(x) for x in args.decode_batches.split(",")):
        for ctx in ctxs:
            batch = prompts(bs, ctx, rng)
            # Keep the historical label when there is one depth, so the
            # existing collectors and trace labels stay valid.
            tag = f"decode{bs}:c{ctx}" if len(ctxs) > 1 else f"decode{bs}"
            if not args.decode_long_only:
                phase(f"{tag}:short", batch, 1, ctx, bs)
            phase(f"{tag}:long", batch, 1 + args.decode_steps, ctx, bs)
    time.sleep(args.gap)
    if args.torch_profile:
        llm.stop_profile()

    write_labels()


if __name__ == "__main__":
    main()
