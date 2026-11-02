# SPDX-License-Identifier: MIT
"""Speculative decoding: same greedy output, and what it buys.

For the baseline (no speculation) and each candidate (MTP, DFlash 2, ...):
output tok/s at one request and at --concurrency at once, mean tokens
accepted per step, and the KV cache the configuration leaves (a draft model
takes some). Then the check that speculation does not change greedy output:
the one-at-a-time outputs are compared token by token with the baseline's.
Exact equality is not required: the verify step runs K+1
rows through the GEMMs where plain decode runs one, so bf16 rounding can flip
a near-tie. A divergence is accepted when the baseline's own top-1 / top-2
logprob gap at that position is below --max-margin; a divergence at a
position the baseline was sure about is a bug (wrong draft acceptance, KV or
state corruption).

    python bench/container/spec_equiv.py \
        --baseline qwen3.6-35b-a3b-fp8.yaml:MTP=0 \
        qwen3.6-35b-a3b-fp8.yaml:MTP=2 qwen3.6-35b-a3b-fp8-dflash2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import Server, first_divergence, parse_target, tokens_and_margins

PROMPTS = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and "
    "bakes muffins for her friends every day with four. She sells the remainder at the "
    "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
    "every day at the farmers' market? Solve step by step.",
    "A train leaves at 3:15 pm and travels 210 km at 84 km/h. When does it arrive? "
    "Solve step by step.",
    "A shop sells pencils at 3 for $0.75 and pens at $1.20 each. Maria buys 12 pencils "
    "and 5 pens and pays with a $20 bill. How much change does she get? Solve step by step.",
    "Write a Python function that parses an ISO 8601 duration string like 'P3DT4H12M' "
    "into a datetime.timedelta, with a docstring and pytest unit tests.",
    "Explain how a B-tree differs from a binary search tree and why databases use "
    "B-trees. Cover node size, disk pages, and insertion with splits.",
    "Write a bash script that finds the 10 largest files under a directory, prints their "
    "sizes in human-readable form, and skips .git directories. Explain each part.",
    "Summarize the causes and main consequences of the French Revolution in about 200 words.",
    "List the planets of the solar system in order from the Sun, with one fact about each.",
]


def run(target: str, max_tokens: int, concurrency: int) -> dict:
    """Prompts one at a time (compared for equivalence, timed), then all at
    once (timed); KV capacity from the startup log."""
    recipe, env = parse_target(target)
    with Server(recipe, env) as s:
        drafts0 = s.metric("vllm:spec_decode_num_drafts_total")
        acc0 = s.metric("vllm:spec_decode_num_accepted_tokens_total")
        outs = []
        t0 = time.time()
        for p in PROMPTS:
            r = s.chat(p, max_tokens=max_tokens, logprobs=True, top_logprobs=2)
            if "error" in r:
                raise RuntimeError(f"{target}: {r['error']}")
            outs.append(tokens_and_margins(r))
        secs = time.time() - t0
        drafts = s.metric("vllm:spec_decode_num_drafts_total") - drafts0
        acc = s.metric("vllm:spec_decode_num_accepted_tokens_total") - acc0

        def gen(p: str) -> int:
            return s.chat(p, max_tokens=max_tokens)["usage"]["completion_tokens"]

        prompts = (PROMPTS * (concurrency // len(PROMPTS) + 1))[:concurrency]
        t0 = time.time()
        with ThreadPoolExecutor(concurrency) as ex:
            toks = sum(ex.map(gen, prompts))
        conc_tok_s = round(toks / (time.time() - t0), 1)
        kv = s.startup_numbers()
    return {"target": target, "outs": outs,
            "tok_s": round(sum(len(t) for t, _ in outs) / secs, 1),
            f"tok_s_x{concurrency}": conc_tok_s,
            "accept_len": round(1 + acc / drafts, 2) if drafts else None,
            "kv_tokens": kv["kv_tokens"], "kv_gib": kv["kv_gib"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=8, help="requests in the all-at-once pass")
    ap.add_argument("--max-margin", type=float, default=1.0,
                    help="largest baseline top-1/top-2 logprob gap a divergence may sit at")
    ap.add_argument("--out", help="write the full result as JSON")
    args = ap.parse_args()

    base = run(args.baseline, args.max_tokens, args.concurrency)
    keys = ("tok_s", f"tok_s_x{args.concurrency}", "kv_tokens", "kv_gib")
    print(json.dumps({"baseline": args.baseline, **{k: base[k] for k in keys}}), flush=True)
    results, ok = [], True
    for cand in args.candidates:
        c = run(cand, args.max_tokens, args.concurrency)
        rows = []
        for i, ((bt, bm), (ct, _)) in enumerate(zip(base["outs"], c["outs"])):
            d = first_divergence(bt, ct)
            margin = None if d is None or d >= len(bm) else round(bm[d], 3)
            bad = d is not None and (margin is None or margin > args.max_margin)
            ok &= not bad
            rows.append({"prompt": i, "tokens": len(bt), "diverges_at": d,
                         "baseline_margin": margin, "bad": bad})
        same = sum(r["diverges_at"] is None for r in rows)
        res = {"candidate": cand, **{k: c[k] for k in keys}, "accept_len": c["accept_len"], "identical": f"{same}/{len(rows)}",
               "divergences": [r for r in rows if r["diverges_at"] is not None],
               "pass": not any(r["bad"] for r in rows)}
        results.append(res)
        print(json.dumps(res), flush=True)
    if args.out:
        json.dump({"baseline": args.baseline, "results": results}, open(args.out, "w"), indent=1)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
