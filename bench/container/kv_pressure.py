# SPDX-License-Identifier: MIT
"""KV cache integrity under pressure: preemption, block reuse, draft pools.

Each prompt is a list of numbered records with random code words; the model
copies a run of them back verbatim, so corrupted KV shows up as a record line
with the right number and the wrong words. Two server starts:

1. the recipe with its normal cache runs each prompt alone: the reference,
   plus each request's prompt + completion tokens and the cache's bytes per
   token (startup log);
2. the recipe again with --kv-cache-memory-bytes sized to a fraction of what
   all requests need together runs them all at once, so running requests are
   preempted and their blocks reused, the path the draft-model KV pools (MTP,
   DFlash 2) change. Sizing in bytes, not blocks, works for any block size.
   Each answer is about as long as its prompt, so admitted requests outgrow
   the blocks they started with even where a block holds thousands of
   tokens (hybrid models). The startup log's bytes per token overstate what
   sliding-window and hybrid (GDN, short-conv) layers keep per token, so when
   a pass does not preempt it is repeated with a smaller cache (--fill, then
   0.5 and 0.3 of it).

It passes when the concurrent pass preempted and no record that came back
right alone comes back with wrong contents under pressure. A record that is
missing is not counted: a near-tie flip between the two passes (different
batch shapes) can change how much a model writes before the list and push
the end past max_tokens. The first token divergence and the alone pass's
top-1 / top-2 gap there are reported for reading.

    python bench/container/kv_pressure.py qwen3.6-35b-a3b-fp8-dflash2.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading

from common import Server, first_divergence, parse_target, tokens_and_margins

WORDS = ["amber", "basil", "cedar", "delta", "ember", "fable", "garnet", "harbor", "indigo", "juniper", "kestrel", "lumen", "marble", "nectar", "onyx", "pepper", "quartz", "raven", "saffron", "tundra", "umber", "velvet", "willow", "xenon", "yarrow", "zephyr"]


def make_prompt(seed: int, records: int, copy: int) -> tuple[str, list[str]]:
    rng = random.Random(seed)
    lines = [f"Record {i:04d}: {rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randrange(1000):03d}"
             for i in range(records)]
    start = rng.randrange(records - copy)
    want = lines[start:start + copy]
    prompt = ("Here is a list of records.\n\n" + "\n".join(lines)
              + f"\n\nCopy records {start:04d} through {start + copy - 1:04d} exactly as they "
              "appear above, one per line, and nothing else.")
    return prompt, want


RECORD = re.compile(r"Record (\d{4}): (\S+)")


def score(text: str, want: list[str]) -> tuple[set[str], set[str]]:
    """Record numbers copied right, and those copied with wrong contents."""
    truth = dict(RECORD.match(w).groups() for w in want)
    right, wrong = set(), set()
    for num, val in RECORD.findall(text):
        if num in truth:
            (right if val == truth[num] else wrong).add(num)
    return right, wrong - right


def ask(s: Server, prompt: str, max_tokens: int) -> dict:
    r = s.chat(prompt, max_tokens=max_tokens, logprobs=True, top_logprobs=2)
    if "error" in r:
        raise RuntimeError(r["error"])
    toks, margins = tokens_and_margins(r)
    return {"text": r["choices"][0]["message"]["content"] or "", "toks": toks, "margins": margins,
            "tokens": r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="recipe.yaml[:ENV=VAL,...]")
    ap.add_argument("--fill", type=float, default=0.8,
                    help="pressured cache as a fraction of the tokens all requests need at once")
    ap.add_argument("--requests", type=int, default=12)
    # The copy is about as long as the prompt, so a request needs as many
    # blocks again while it decodes: hybrid models allocate in blocks of
    # ~2k tokens (Qwen3.6: 2,176), which a short answer never leaves.
    ap.add_argument("--records", type=int, default=150, help="records per prompt (~14 tokens each)")
    ap.add_argument("--copy", type=int, default=140, help="records to copy back")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--out")
    args = ap.parse_args()

    recipe, env = parse_target(args.target)
    env.update({"MAX_MODEL_LEN": str(args.max_model_len), "MAX_NUM_SEQS": str(args.requests)})
    prompts = [make_prompt(seed, args.records, args.copy) for seed in range(args.requests)]
    max_tokens = args.copy * 40

    with Server(recipe, env) as s:
        alone = [ask(s, p, max_tokens) for p, _ in prompts]
        normal = s.startup_numbers()
    kv_bytes = float(normal["kv_gib"]) * 2**30
    per_token = kv_bytes / int(normal["kv_tokens"].replace(",", ""))
    need = sum(a["tokens"] for a in alone)

    prev_fill = args.fill
    for fill in (args.fill, args.fill * 0.5 / 0.8, args.fill * 0.3 / 0.8):
        budget = int(fill * need * per_token)
        pressured = dict(env)
        pressured["EXTRA_ARGS"] = (env.get("EXTRA_ARGS", "")
                                   + f" --kv-cache-memory-bytes {budget}").strip()
        try:
            with Server(recipe, pressured) as s:
                pre0 = s.metric("vllm:num_preemptions_total")
                together: list[dict | None] = [None] * len(prompts)

                def worker(i: int, out: list = together) -> None:
                    out[i] = ask(s, prompts[i][0], max_tokens)

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                preemptions = s.metric("vllm:num_preemptions_total") - pre0
                numbers = s.startup_numbers()
                if args.out:
                    with open(args.out + ".log", "w") as f:
                        f.write(s.logs())
        except RuntimeError as e:  # cache below one max-length request
            if fill == args.fill:
                raise
            print(f"fill {fill:.2f}: {str(e).splitlines()[0]}; keeping the previous pass",
                  file=sys.stderr)
            fill = prev_fill
            break
        prev_fill = fill
        print(f"fill {fill:.2f}: {numbers['kv_tokens']} KV tokens, {preemptions:.0f} preemptions",
              file=sys.stderr, flush=True)
        if preemptions > 0:
            break

    rows, ok = [], preemptions > 0
    for i, ((_, want), a, b) in enumerate(zip(prompts, alone, together)):
        (ra, _), (rb, wb) = score(a["text"], want), score(b["text"], want)
        corrupted = sorted(wb & ra)
        d = first_divergence(a["toks"], b["toks"])
        margin = None if d is None or d >= len(a["margins"]) else round(a["margins"][d], 3)
        ok &= not corrupted
        rows.append({"prompt": i, "alone_right": len(ra), "pressure_right": len(rb),
                     "pressure_wrong": len(wb), "of": len(want), "corrupted": corrupted,
                     "diverges_at": d, "alone_margin": margin, "bad": bool(corrupted),
                     "alone_text": a["text"], "pressure_text": b["text"]})
    res = {"target": args.target, "tokens_needed": need, "fill": round(fill, 2),
           "kv_tokens": numbers["kv_tokens"],
           "kv_tokens_normal": normal["kv_tokens"], "preemptions": preemptions, "rows": rows,
           "pass": ok}
    for r in rows:
        print(json.dumps({k: v for k, v in r.items() if not k.endswith("_text")}))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    if preemptions == 0:
        print("NOT PRESSURED: no preemption happened, even at the smallest cache; lower --fill")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
