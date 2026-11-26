# SPDX-License-Identifier: MIT
"""Greedy generation on fixed prompts; saves token ids for an A/B compare.

STRIX_DISABLE=all python bench/model/e2e_check.py --model M --out base.json
python bench/model/e2e_check.py --model M --out strix.json --compare base.json

--kv-cache-dtype picks the KV cache the decode kernels read (int8_per_token_head
is what the recipes serve). --ctx N puts N tokens of wikitext in front of every
question, a different slice for each, so the greedy decode runs at that depth.
Requests run one at a time unless --batch. --out keeps the settings next to the
outputs, and --compare exits with an error when the two sides' differ, or when
their prompts do.

A comparison also records the number of identical prompts in the results DB
(bench/results.py).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/results.py
import results  # noqa: E402
from kld_check import build_sequences
from vllm import LLM, SamplingParams

PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a haiku about memory bandwidth.",
    "What are the main differences between TCP and UDP? Answer in a list.",
    (
        "Solve step by step: a train leaves at 3:15 pm and travels 210 km at "
        "84 km/h. When does it arrive?"
    ),
    "Translate to French: The kernel reads each weight exactly once.",
    "Summarize the plot of Hamlet in three sentences.",
]


# Settings both sides of a comparison must share. The plugin, its kernels and
# --mamba-ssm-cache-dtype are what a comparison is about.
SAME = ("model", "kv_cache_dtype", "ctx", "batch", "mtp", "max_tokens")


def load_base(path: str, settings: dict) -> list[dict]:
    """The outputs of an earlier run with the same settings; exits if not."""
    base = json.loads(Path(path).read_text())
    if isinstance(base, list):
        sys.exit(
            f"{path} holds outputs without their settings (kv dtype, ctx, batch, "
            "...), so it cannot be compared; write it again with this script"
        )
    diff = [
        f"{k}: {base['settings'].get(k)!r} there, {settings[k]!r} here"
        for k in SAME
        if base["settings"].get(k) != settings[k]
    ]
    if diff:
        sys.exit(f"{path} was run with other settings:\n  " + "\n  ".join(diff))
    return base["outputs"]


def check_prompts(path: str, base: list[dict], result: list[dict]) -> None:
    """Exits unless both sides hold the same prompts."""
    if len(base) != len(result):
        sys.exit(f"{path} holds {len(base)} prompts, this run {len(result)}")
    other = [
        i for i, (a, b) in enumerate(zip(base, result)) if a["prompt"] != b["prompt"]
    ]
    if other:
        sys.exit(f"{path}: the tokens of prompts {other} differ from this run's")


def gap(top: dict) -> float:
    """Top-1 minus top-2 log-probability of one generated token."""
    lps = sorted((lp.logprob for lp in top.values()), reverse=True)
    return lps[0] - lps[1]


def decode_tile_note(alive: int, draft: bool) -> str:
    """Which batches of the run could be decoded by strix.attn_decode."""
    if draft:
        return "every decode step is a verify batch: attn_decode is not reached"
    try:
        from strix_vllm import disabled
    except ImportError:
        return "no strix_vllm here"
    if disabled("attn_decode"):
        return "STRIX_DISABLE keeps attn_decode off"
    up_to = "up to " if alive > 1 else ""
    return (
        f"plain decode of {up_to}{alive} sequence(s) at once; attn_decode runs the "
        "geometries in the kernel table"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--compare")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--batch", action="store_true", help="all prompts at once")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--mtp", type=int, default=0, help="MTP draft tokens, 0 = off")
    p.add_argument(
        "--mamba-ssm-cache-dtype",
        default="auto",
        help="GDN recurrent-state dtype; token agreement between auto (fp32) "
        "and bfloat16 over a long greedy generation is the accuracy check for "
        "halving the state traffic",
    )
    p.add_argument(
        "--kv-cache-dtype",
        default="auto",
        help="vLLM KV cache dtype, e.g. int8_per_token_head",
    )
    p.add_argument(
        "--ctx",
        type=int,
        default=0,
        help="wikitext tokens in front of each question (0 = the bare question)",
    )
    p.add_argument(
        "--corpus",
        default=os.environ.get("STRIX_WIKITEXT"),
        help="directory holding wikitext-2-raw, for --ctx; default $STRIX_WIKITEXT",
    )
    args = p.parse_args()
    if args.ctx and not args.corpus:
        p.error("--ctx needs --corpus or $STRIX_WIKITEXT")
    settings = {
        "model": Path(args.model).name,
        "kv_cache_dtype": args.kv_cache_dtype,
        "ctx": args.ctx,
        "batch": args.batch,
        "mtp": args.mtp,
        "max_tokens": args.max_tokens,
        "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype,
        "strix_disable": os.environ.get("STRIX_DISABLE", ""),
    }
    base = load_base(args.compare, settings) if args.compare else None
    run_id = results.start_run(__file__, args.model)

    llm = LLM(
        model=args.model,
        language_model_only=True,
        # The question, the template and the output fit in the 4096 tokens.
        max_model_len=args.ctx + 4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=8,
        enable_prefix_caching=False,
        attention_backend="TRITON_ATTN",
        kv_cache_dtype=args.kv_cache_dtype,
        # Same prefill chunk as profile_model for long prompts.
        **({"max_num_batched_tokens": 8192} if args.ctx else {}),
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": args.mtp} if args.mtp else None
        ),
    )
    # Two logprobs per token give the top-1 / top-2 gap at a divergence.
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, logprobs=2)
    questions = PROMPTS
    if args.ctx:
        tok = llm.get_tokenizer()
        slices = build_sequences(tok, Path(args.corpus), args.ctx, 0, len(PROMPTS))
        texts = [tok.decode(ids, skip_special_tokens=True) for ids in slices]
        questions = [f"{t}\n\n{q}" for t, q in zip(texts, PROMPTS)]
    msgs = [[{"role": "user", "content": q}] for q in questions]
    if args.batch:
        outs = llm.chat(msgs, sp, use_tqdm=False)
    else:
        outs = [llm.chat(m, sp, use_tqdm=False)[0] for m in msgs]
    if args.ctx:
        print("prompt tokens:", *(len(o.prompt_token_ids) for o in outs))
    alive = len(outs) if args.batch else 1
    print(f"DECODE TILE: {decode_tile_note(alive, args.mtp > 0)}")
    result = [
        {
            "ids": list(o.outputs[0].token_ids),
            "text": o.outputs[0].text,
            "gaps": [gap(t) for t in o.outputs[0].logprobs],
            "prompt": hashlib.md5(str(list(o.prompt_token_ids)).encode()).hexdigest(),
        }
        for o in outs
    ]
    with open(args.out, "w") as f:
        json.dump({"settings": settings, "outputs": result}, f, indent=1)

    if base is not None:
        check_prompts(args.compare, base, result)
        diverge = []
        for i, (a, b) in enumerate(zip(base, result)):
            n = next(
                (j for j, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y),
                None,
            )
            if n is None and len(a["ids"]) != len(b["ids"]):
                n = min(len(a["ids"]), len(b["ids"]))  # one side stopped there
            status = "identical" if n is None else f"diverges at token {n}"
            if n is not None and n < len(a.get("gaps", ())):
                status += f", base top-1/top-2 gap {a['gaps'][n]:.3f}"
            diverge.append(n)
            print(f"prompt {i}: {status}")
            if n is not None:
                print(f"  base:  {a['text'][:200]!r}\n  strix: {b['text'][:200]!r}")
        results.record(
            run_id,
            "e2e",
            "identical_prompts",
            sum(n is None for n in diverge),
            label="greedy",
            depth=args.ctx or None,
            kv_dtype=args.kv_cache_dtype,
            extra={
                "prompts": len(diverge),
                "diverge_token": diverge,
                "mtp": args.mtp,
                "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype,
                "batch": args.batch,
                "ctx": args.ctx,
            },
        )
    print("SAMPLE:", result[0]["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
