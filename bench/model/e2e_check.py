# SPDX-License-Identifier: MIT
"""Greedy generation on fixed prompts; saves token ids for an A/B compare.

STRIX_DISABLE=all python bench/model/e2e_check.py --model M --out base.json
python bench/model/e2e_check.py --model M --out strix.json --compare base.json

A comparison also records the number of identical prompts in the results DB
(bench/results.py).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/results.py
import results  # noqa: E402
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
    p.add_argument("--db", help="results DB (default profiles/results.db)")
    args = p.parse_args()
    run_id = results.start_run(__file__, args.model, db=args.db)

    llm = LLM(
        model=args.model,
        language_model_only=True,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=8,
        enable_prefix_caching=False,
        attention_backend="TRITON_ATTN",
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": args.mtp} if args.mtp else None
        ),
    )
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    msgs = [[{"role": "user", "content": q}] for q in PROMPTS]
    if args.batch:
        outs = llm.chat(msgs, sp, use_tqdm=False)
    else:
        outs = [llm.chat(m, sp, use_tqdm=False)[0] for m in msgs]
    result = [
        {"ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text} for o in outs
    ]
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    if args.compare:
        with open(args.compare) as f:
            base = json.load(f)
        diverge = []
        for i, (a, b) in enumerate(zip(base, result)):
            n = next(
                (j for j, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y),
                None,
            )
            status = "identical" if n is None else f"diverges at token {n}"
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
            kv_dtype="auto",
            extra={
                "prompts": len(diverge),
                "diverge_token": diverge,
                "mtp": args.mtp,
                "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype,
                "batch": args.batch,
            },
            db=args.db,
        )
    print("SAMPLE:", result[0]["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
