# SPDX-License-Identifier: MIT
"""GSM8K against a served recipe, with a paired comparison.

Greedy, thinking off, the same instruction as bench/model/gsm8k_check.py.
--out saves per-question correctness; --compare scores two saved runs on the
same questions with an exact McNemar test, which looks only at the questions
the two sides disagree on and is far more sensitive than comparing totals.

    python bench/container/gsm8k.py qwen3.6-35b-a3b-fp8.yaml:STRIX_DISABLE=all --out stock.json
    python bench/container/gsm8k.py qwen3.6-35b-a3b-fp8.yaml --out strix.json
    python bench/container/gsm8k.py --compare stock.json strix.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor

INSTRUCTION = "Solve the problem. End with a line 'Answer: <number>'.\n\n"


def last_number(text: str) -> str | None:
    m = re.search(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text)
    nums = [m.group(1)] if m else re.findall(r"-?[\d,]*\.?\d+", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Questions only a got right, only b got right, two-sided exact p."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return only_a, only_b, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="recipe.yaml[:ENV=VAL,...]")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.compare:
        a, b = (json.load(open(p))["hits"] for p in args.compare)
        assert len(a) == len(b), "runs cover different questions"
        only_a, only_b, p = mcnemar(a, b)
        print(f"A {sum(a)}/{len(a)}, B {sum(b)}/{len(b)}; only A right {only_a}, "
              f"only B right {only_b}; McNemar p = {p:.3f}")
        return

    from common import Server, parse_target
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    recipe, env = parse_target(args.target)
    with Server(recipe, env) as s:
        def one(d) -> bool:
            r = s.chat(INSTRUCTION + d["question"], max_tokens=args.max_tokens)
            text = (r["choices"][0]["message"]["content"] or "") if "choices" in r else ""
            gold = d["answer"].split("####")[-1].strip().replace(",", "")
            pred = last_number(text)
            return pred is not None and float(pred) == float(gold)

        with ThreadPoolExecutor(args.concurrency) as ex:
            hits = list(ex.map(one, data))
    print(f"GSM8K {args.target}: {sum(hits)}/{args.n} = {sum(hits) / args.n:.3f}")
    if args.out:
        json.dump({"target": args.target, "hits": hits}, open(args.out, "w"))


if __name__ == "__main__":
    main()
