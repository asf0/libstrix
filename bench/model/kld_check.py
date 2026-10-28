# SPDX-License-Identifier: MIT
"""KLD, perplexity and top-1 agreement between two numerics settings, at depth.

Two modes, like state_agreement.py:

    # reference (e.g. STRIX_DISABLE=all)
    python bench/model/kld_check.py --model M --depth 32768 --out ref.json
    # compared settings, replaying the same token sequences
    python bench/model/kld_check.py --model M --depth 32768 --out new.json \
        --compare ref.json

The sequences are held-out wikitext-2-raw contexts of `--depth` tokens plus a
`--continuation`-token tail; only the tail is scored. Each position stores the
top-K prompt logprobs (`--topk`), which gives:

  * perplexity of the tail, exp(-mean logprob of the actual next token);
  * KLD between the two top-K distributions, renormalized over the union of
    their ids (ids missing from one side get half that side's smallest
    observed probability). This is a lower bound on the full-vocabulary KLD,
    with the same truncation caveat as llama.cpp's kld gate;
  * top-1 agreement and the paired |dlogprob| of the chosen token.

The numbers are also recorded in the results DB (bench/results.py): a
`kld_ref` row when the reference is written, `kld` rows for a comparison. The
reference JSON stays a file because --compare replays its sequences.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # bench/results.py
import results  # noqa: E402
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

TESTS = ("wiki.test.raw", "wiki.valid.raw", "wiki.train.raw")


def corpus_text(corpus: Path) -> str:
    for name in TESTS:
        hits = list(corpus.rglob(name))
        if hits:
            return hits[0].read_text()
    raise FileNotFoundError(f"no wiki.*.raw under {corpus}")


def build_sequences(
    tok, corpus: Path, depth: int, tail: int, samples: int
) -> list[list[int]]:
    ids = tok.encode(corpus_text(corpus))
    need = depth + tail
    # wikitext-2 is ~250k tokens; a 128k context needs repeats to get more
    # than one sample (documented: the corpus is cycled, not disjoint).
    reps = max(1, (samples * need + len(ids) - 1) // len(ids))
    ids = ids * reps
    n = max(min(samples, len(ids) // need), 1)
    return [ids[i * need : (i + 1) * need] for i in range(n)]


def topk_map(d: dict, k: int) -> dict[int, float]:
    items = sorted(d.items(), key=lambda kv: kv[1].logprob, reverse=True)[:k]
    return {int(i): float(v.logprob) for i, v in items}


def kld(a: dict[int, float], b: dict[int, float]) -> float:
    union = set(a) | set(b)
    pa = {i: math.exp(a[i]) for i in a}
    pb = {i: math.exp(b[i]) for i in b}
    floor_a = min(pa.values()) / 2 if pa else 0.0
    floor_b = min(pb.values()) / 2 if pb else 0.0
    pa = {i: pa.get(i, floor_a) for i in union}
    pb = {i: pb.get(i, floor_b) for i in union}
    sa, sb = sum(pa.values()), sum(pb.values())
    return sum((v / sa) * math.log((v / sa) / (pb[i] / sb)) for i, v in pa.items())


def load_json(path: str) -> dict:
    """JSON turns integer token ids into string keys; restore them."""
    d = json.loads(Path(path).read_text())
    d["rows"] = [[{int(k): v for k, v in pos.items()} for pos in r] for r in d["rows"]]
    return d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--corpus",
        default=os.environ.get("STRIX_WIKITEXT"),
        help="directory holding wikitext-2-raw (wiki.test.raw); "
        "default $STRIX_WIKITEXT",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--compare", help="reference json from an earlier run")
    p.add_argument("--depth", type=int, default=32768, help="scored context")
    p.add_argument("--continuation", type=int, default=512)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--max-model-len", type=int, help="default: depth + tail")
    p.add_argument("--mamba-ssm-cache-dtype", default="auto")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--db", help="results DB (default profiles/results.db)")
    args = p.parse_args()
    if not args.compare and not args.corpus:
        p.error("a reference run needs --corpus or $STRIX_WIKITEXT")
    run_id = results.start_run(__file__, args.model, db=args.db)

    def rec(phase, metric, value, label, extra=None):
        results.record(
            run_id,
            phase,
            metric,
            value,
            label=label,
            depth=args.depth,
            kv_dtype=args.kv_cache_dtype,
            extra=extra,
            db=args.db,
        )

    seq_len = args.depth + args.continuation
    llm = LLM(
        model=args.model,
        language_model_only=True,
        # vLLM rejects prompt + max_tokens > max_model_len; leave room for
        # the one generated token (the same off-by-one as profile_model).
        max_model_len=args.max_model_len or seq_len + 8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        # Same prefill chunk as profile_model, so a vLLM default change does
        # not move the numerics under the comparison.
        max_num_batched_tokens=8192,
        attention_backend="TRITON_ATTN",
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_prefix_caching=False,
    )
    tok = llm.get_tokenizer()

    if args.compare:
        seqs = load_json(args.compare)["seqs"]
    else:
        seqs = build_sequences(
            tok, Path(args.corpus), args.depth, args.continuation, args.samples
        )

    sp = SamplingParams(max_tokens=1, prompt_logprobs=args.topk, temperature=0.0)
    rows = []
    for ids in seqs:
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        per_pos = []
        for i in range(args.depth, len(ids)):
            d = out[0].prompt_logprobs[i]
            per_pos.append(topk_map(d, args.topk) if d else {})
        rows.append(per_pos)

    data = {
        "seqs": seqs,
        "rows": rows,
        "depth": args.depth,
        "continuation": args.continuation,
        "topk": args.topk,
    }
    Path(args.out).write_text(json.dumps(data))

    if not args.compare:
        n = sum(len(r) for r in rows)
        print(f"wrote {n} positions to {args.out}")
        rec("kld_ref", "positions", n, "ref", {"out": args.out, "samples": len(seqs)})
        return

    # The evaluated set is fixed by the REFERENCE alone: every position where
    # the reference's top-K holds the chosen token. Requiring it in both maps
    # made the set depend on the compared side, and the positions that dropped
    # out were exactly the ones where the two distributions disagree most, so
    # PPL and KLD were biased by a pair-dependent selection effect.
    ref = load_json(args.compare)
    base = ref["rows"]
    tot = hit = 0
    dlp: list[float] = []
    kls: list[float] = []
    nll_ref = nll_new = 0.0
    for bi, ni, seq in zip(base, rows, seqs):
        for j, (ra, rn) in enumerate(zip(bi, ni)):
            if not ra or not rn:
                continue
            sid = seq[args.depth + j]
            if sid not in ra:
                continue
            tot += 1
            hit += max(ra, key=ra.get) == max(rn, key=rn.get)
            nll_ref -= ra[sid]
            if sid in rn:
                dlp.append(abs(ra[sid] - rn[sid]))
                nll_new -= rn[sid]
            else:
                # The compared side did not keep it in its top-K: charge the
                # same floor kld() uses, so PPL stays on the fixed set.
                nll_new -= math.log(min(math.exp(v) for v in rn.values()) / 2)
            kls.append(kld(ra, rn))
    kls.sort()
    if tot == 0:
        print("positions 0: no comparable positions (check --depth/topk)")
        return

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")

    label = f"depth{args.depth}"
    ppl_ref, ppl_new = math.exp(nll_ref / tot), math.exp(nll_new / tot)
    print(f"positions {tot}")
    rec("kld", "positions", tot, label, {"samples": len(seqs), "topk": args.topk})
    rec("kld", "top1", hit / tot, label)
    rec("kld", "dlogprob_mean", sum(dlp) / len(dlp), label)
    rec("kld", "dlogprob_p95", pct(sorted(dlp), 0.95), label)
    rec("kld", "ppl_ref", ppl_ref, label)
    rec("kld", "ppl", ppl_new, label)
    rec("kld", "ppl_delta_pct", 100 * (ppl_new / ppl_ref - 1), label)
    rec("kld", "kld", sum(kls) / len(kls), label)
    rec("kld", "kld_p95", pct(kls, 0.95), label)
    print(
        f"top-1 agreement {hit}/{tot} = {hit / tot:.4f} "
        f"({100 * (1 - hit / tot):.2f} points lost)"
    )
    print(
        f"mean |dlogprob| {sum(dlp) / len(dlp):.4f}  p95 {pct(sorted(dlp), 0.95):.4f}"
    )
    print(
        f"perplexity: ref {math.exp(nll_ref / tot):.3f}  "
        f"new {math.exp(nll_new / tot):.3f}  "
        f"delta {100 * (math.exp(nll_new / tot) / math.exp(nll_ref / tot) - 1):+.2f}%"
    )
    print(
        f"KLD (top-{args.topk}, union-renormalized): "
        f"mean {sum(kls) / len(kls):.5f}  p95 {pct(kls, 0.95):.5f}"
    )


if __name__ == "__main__":
    main()
