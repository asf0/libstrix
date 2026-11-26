# SPDX-License-Identifier: MIT
"""Phase-separated vLLM run for per-kernel profiling under rocprofv3.

Each phase is preceded by an idle gap so a kernel trace splits into one burst
per phase; the phase labels are written to --labels for analyze_trace.py.
Every printed number is also recorded in the results DB (bench/results.py).

DECODE lines come from the request timestamps (RequestOutput.metrics), with
prefix caching off: the time from the first to the last token of the request
that started decoding last, over its --decode-steps steps. In a batched point
that is the request submitted last, and the others get enough extra tokens to
outlive it, so the batch stays full while it is timed (PHASE seconds include
their tail). ALIVE says how many requests decoded through the whole window; if
it is not all of them, the line reads INVALID and no decode_ms is recorded. That
is the guard against a batch that did not hold its size.

Without --decode-long-only each decode point also runs the batch for 1 token
(`:short`): analyze_trace.py subtracts its kernel times from the `:long` run's,
which cancels prefill, and divides by --decode-steps. --labels runs therefore
keep equal token counts, and most of their deep batched rows are INVALID.

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
# Decode steps of slack for requests that must outlive the timed one (they can
# start while the rest are still being submitted).
HOLD_MARGIN_STEPS = 16


def hold_tokens(lens: list[int], steps: int, budget: int, per_step: int) -> int:
    """max_tokens that keeps a request of the batch decoding through the timed
    window, whichever request starts last.

    The engine computes at most `budget` tokens a step, decode rows included,
    so prefilling prompts of `lens` takes at most one step per started chunk;
    a request that starts first decodes up to `per_step` tokens (1 + draft
    tokens) in each of those steps and in each step of the window.
    """
    chunk = budget - len(lens) * per_step
    if chunk <= 0:
        raise ValueError(f"max_num_batched_tokens {budget} leaves no prefill chunk")
    prefill_steps = sum(-(-n // chunk) for n in lens)
    return 1 + per_step * (steps + prefill_steps + HOLD_MARGIN_STEPS)


def alive_through_last(metrics: list) -> int:
    """How many requests decoded from the first to the last token of the
    request submitted last (metrics in submission order)."""
    last = metrics[-1]
    return sum(
        m.first_token_ts <= last.first_token_ts
        and m.last_token_ts >= last.last_token_ts
        for m in metrics
    )


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
    p.add_argument(
        "--speculative-config",
        help="JSON speculative config (any method, e.g. a DFlash 2 draft); "
        "overrides --mtp",
    )
    p.add_argument(
        "--prompts-file",
        help="JSON list of chat prompts: decode batches use them (chat template "
        "applied, cycled to the batch size) instead of random tokens, so a "
        "draft accepts what it would on real text; --decode-ctx is ignored",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature; 0 (the default) is greedy, as every "
        "measurement in docs/performance.md",
    )
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0, help="0 = off")
    p.add_argument(
        "--prefill-same-prompt",
        action="store_true",
        help="the second prefill rep at a depth reuses the first's prompt: "
        "prefix caching is off, so it recomputes everything, but its "
        "Qwen3.8-Flash-Next n-gram rows are then in the page cache, so the "
        "rep difference is the disk-read share of TTFT",
    )
    p.add_argument("--gap", type=float, default=4.0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument(
        "--compilation-config",
        help="JSON passed as vLLM's compilation_config, e.g. "
        "'{\"cudagraph_mode\": \"NONE\"}' to isolate what the graphs buy",
    )
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument(
        "--attention-backend",
        default="TRITON_ATTN",
        help="'' lets vLLM pick (Qwen4-exp QSA layers bring their own kernels)",
    )
    p.add_argument(
        "--engram-disk-offload",
        action="store_true",
        help="Qwen3.8-Flash-Next: stream the n-gram table from disk "
        "(engram_config={'disk_offload': True})",
    )
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
    run_id = results.start_run(__file__, args.model)

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
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        attention_backend=args.attention_backend or None,
        kv_cache_dtype=args.kv_cache_dtype,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        # Per-request token timestamps (RequestOutput.metrics) for DECODE lines.
        disable_log_stats=False,
        enforce_eager=args.enforce_eager,
        **(
            {"compilation_config": json.loads(args.compilation_config)}
            if args.compilation_config
            else {}
        ),
        **({"engram_config": {"disk_offload": True}} if args.engram_disk_offload else {}),
        speculative_config=(
            json.loads(args.speculative_config)
            if args.speculative_config
            else {"method": "mtp", "num_speculative_tokens": args.mtp}
            if args.mtp
            else None
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
    chat_prompts: list[TokensPrompt] = []
    if args.prompts_file:
        tok = llm.get_tokenizer()
        for text in json.loads(Path(args.prompts_file).read_text()):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
            )
            if not isinstance(ids, list):  # transformers 5: a BatchEncoding
                ids = ids["input_ids"]
            chat_prompts.append(TokensPrompt(prompt_token_ids=list(ids)))

    def spec_counts() -> tuple[float, float]:
        """(drafts, accepted tokens) so far, from the engine's metrics."""
        drafts = accepted = 0.0
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_drafts":
                drafts += getattr(m, "value", 0.0)
            elif m.name == "vllm:spec_decode_num_accepted_tokens":
                accepted += getattr(m, "value", 0.0)
        return drafts, accepted
    rng = random.Random(0)
    llm.generate(prompts(1, 512, rng), SamplingParams(max_tokens=8), use_tqdm=False)

    def kind_of(label: str) -> str:
        return next((k for k in ("prefill", "decode") if label.startswith(k)), "e2e")

    labels: list[str] = []
    if args.torch_profile:
        llm.start_profile()

    def sampling(max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            ignore_eos=True,
        )

    def phase(
        label: str,
        batch: list[TokensPrompt],
        max_tokens: int,
        depth: int,
        bs: int = 1,
        hold: int | None = None,
    ) -> None:
        """hold: max_tokens of every request but the one submitted last."""
        time.sleep(args.gap)
        sp = sampling(max_tokens)
        if hold is not None:
            sp = [sampling(hold) for _ in batch[1:]] + [sp]
        d0, a0 = spec_counts()
        t0 = time.perf_counter()
        outs = llm.generate(batch, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        labels.append(label)
        print(f"PHASE {label}: {dt:.3f} s", flush=True)
        d1, a1 = spec_counts()
        if d1 > d0:
            accept = 1 + (a1 - a0) / (d1 - d0)
            print(f"ACCEPT {label}: {accept:.2f} tokens/step", flush=True)
            rec(kind_of(label), "accept_len", accept, label, depth, bs)
        kind = kind_of(label)
        rec(kind, "phase_s", dt, label, depth, bs)
        # Decode step time of the request that started decoding last: its
        # steps all run after every prefill chunk of the batch (earlier ones
        # also count steps shared with other requests' chunked prefill).
        # long - short cancels prefill too, but slow prefills vary by seconds.
        done = [
            o for o in outs if o.metrics is not None and len(o.outputs[0].token_ids) > 1
        ]
        if not done:
            return
        batched = len(batch) > 1
        valid = True
        if batched:
            # vLLM admits and schedules requests in submission order, so the
            # one submitted last starts last; the timestamps confirm it, and
            # that the others were decoding (not preempted) until it finished.
            o = outs[-1]
            timed = len(done) == len(outs)
            n_alive = alive_through_last([x.metrics for x in outs]) if timed else 0
            preempted = sum(x.metrics.num_preemptions for x in done)
            n_out = len(o.outputs[0].token_ids)
            print(f"ALIVE {label}: {n_alive}/{len(batch)}", flush=True)
            valid = n_alive == len(batch) and not preempted and n_out == max_tokens
            if not valid:
                print(
                    f"INVALID {label}: {n_alive}/{len(batch)} requests decoding "
                    f"through the timed window, {preempted} preemptions, "
                    f"{n_out}/{max_tokens} tokens timed; decode_ms not recorded",
                    flush=True,
                )
        else:
            o = max(done, key=lambda o: o.metrics.first_token_ts)
        if valid:
            step = (o.metrics.last_token_ts - o.metrics.first_token_ts) / (
                len(o.outputs[0].token_ids) - 1
            )
            print(f"DECODE {label}: {step * 1e3:.2f} ms/step", flush=True)
            rec("decode", "decode_ms", step * 1e3, label, depth, bs)
        if o.metrics is not None:
            print(f"TTFT {label}: {o.metrics.first_token_latency:.3f} s", flush=True)
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

    ctxs = [int(x) for x in str(args.decode_ctx).split(",")]
    batch_sizes = [int(x) for x in args.decode_batches.split(",")]
    cfg = llm.llm_engine.vllm_config
    spec = cfg.speculative_config
    # Tokens one request can emit per step.
    per_step = 1 + (spec.num_speculative_tokens or 0 if spec else 0)

    def hold_for(lens: list[int]) -> int | None:
        if len(lens) == 1 or args.labels:
            return None
        hold = hold_tokens(
            lens, args.decode_steps, args.max_num_batched_tokens, per_step
        )
        limit = cfg.model_config.max_model_len
        if max(lens) + hold > limit:
            raise SystemExit(
                f"batch of {len(lens)} at {max(lens)} prompt tokens: the requests "
                f"decoding while the rest prefill need {hold} output tokens, "
                f"{max(lens) + hold} in all, but --max-model-len is {limit}; raise "
                "it or lower --decode-steps"
            )
        return hold

    def batch_lens(bs: int, ctx: int) -> list[int]:
        if not chat_prompts:
            return [ctx] * bs
        return [
            len(chat_prompts[i % len(chat_prompts)]["prompt_token_ids"])
            for i in range(bs)
        ]

    # Every point must fit before the first one runs.
    for bs in batch_sizes:
        for ctx in ctxs:
            hold_for(batch_lens(bs, ctx))

    for length in (int(x) for x in args.prefill.split(",") if x):
        # Two reps smooth the shallow depths; a 64k+ prefill is expensive
        # enough that one measured pass is the right trade.
        first = prompts(1, length, rng)
        for rep in range(2 if length < 65536 else 1):
            batch = first if rep == 0 or args.prefill_same_prompt else prompts(1, length, rng)
            phase(f"prefill{length}#{rep}", batch, 1, length)
    for bs in batch_sizes:
        for ctx in ctxs:
            batch = (
                [chat_prompts[i % len(chat_prompts)] for i in range(bs)]
                if chat_prompts
                else prompts(bs, ctx, rng)
            )
            # Keep the historical label when there is one depth, so the
            # existing collectors and trace labels stay valid.
            tag = f"decode{bs}:c{ctx}" if len(ctxs) > 1 else f"decode{bs}"
            if not args.decode_long_only:
                phase(f"{tag}:short", batch, 1, ctx, bs)
            phase(
                f"{tag}:long",
                batch,
                1 + args.decode_steps,
                ctx,
                bs,
                hold_for(batch_lens(bs, ctx)),
            )
    time.sleep(args.gap)
    if args.torch_profile:
        llm.stop_profile()

    write_labels()


if __name__ == "__main__":
    main()
