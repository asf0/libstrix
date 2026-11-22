"""Time to first token for long prompts on a recipe (prefill speed).

    python bench/container/ttft.py recipe.yaml[:ENV=VAL,...] [--lengths 2048 8192]
"""

import argparse
import random
import time

from common import Server, parse_target

WORDS = "amber basil cedar delta ember fable garnet harbor indigo juniper".split()


def prompt(tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    # About 1.3 tokens per word for these words plus spaces.
    words = [rng.choice(WORDS) for _ in range(int(tokens / 1.3))]
    return "Summarize this list in one sentence:\n" + " ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384])
    args = ap.parse_args()
    recipe, env = parse_target(args.target)
    with Server(recipe, env) as s:
        s.chat("Hello", max_tokens=4)
        for n in args.lengths:
            for seed in range(2):  # the second send is the warm number
                t0 = time.time()
                r = s.chat(prompt(n, seed * 1000 + n), max_tokens=1)
                dt = time.time() - t0
            if "usage" not in r:
                print(f"prompt ~{n}: {str(r)[:300]}", flush=True)
                with open("ttft-error.log", "w") as fh:
                    fh.write(s.logs())
                return
            used = r["usage"]["prompt_tokens"]
            print(f"prompt {used:6d} tokens: TTFT {dt:6.2f} s, {used / dt:7.0f} tok/s",
                  flush=True)


if __name__ == "__main__":
    main()
