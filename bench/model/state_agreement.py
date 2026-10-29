# SPDX-License-Identifier: MIT
"""Teacher-forced top-1 agreement between two numerics settings.

Greedy generation flips on the first near-tie, so "diverges at token 94" says
nothing about how much a change actually moved the distribution. This fixes the
token sequence and compares the model's next-token decision at every position,
which is the metric the W8A8 activation-quant scheme was judged on (2.6 points of agreement was too much to ship).

    python bench/model/state_agreement.py --model M --out ref.json \\
        --mamba-ssm-cache-dtype float32
    python bench/model/state_agreement.py --model M --out new.json \\
        --mamba-ssm-cache-dtype bfloat16 --compare ref.json

The sequences are fixed prompts followed by the token ids a reference run
generated, so the comparison spans a real long generation rather than a
prompt. Only the first token's prefill is skipped (its logprob has no context).
"""

import argparse
import json

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    (
        "Solve step by step: a train leaves at 3:15 pm and travels 210 km at "
        "84 km/h. When does it arrive?"
    ),
    "Summarize the plot of Hamlet in three sentences.",
    (
        "Write a Python function that merges two sorted lists, then explain the "
        "complexity of each step."
    ),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--compare", help="reference json from an earlier run")
    p.add_argument("--tokens", type=int, default=384, help="continuation length")
    p.add_argument("--mamba-ssm-cache-dtype", default="auto")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument(
        "--attention-backend",
        default="TRITON_ATTN",
        help="'' lets vLLM pick (Qwen4-exp QSA layers bring their own kernels)",
    )
    p.add_argument(
        "--engram-disk-offload",
        action="store_true",
        help="Qwen3.8-Flash-Next: n-gram table read from disk",
    )
    args = p.parse_args()

    llm = LLM(
        model=args.model,
        language_model_only=True,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=8,
        attention_backend=args.attention_backend or None,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_prefix_caching=False,
        **({"engram_config": {"disk_offload": True}} if args.engram_disk_offload else {}),
    )
    tok = llm.get_tokenizer()

    # Both settings must see the SAME tokens: the reference run generates the
    # continuation and saves it, the compared run replays it. Letting each run
    # generate its own makes the comparison meaningless (they branch apart).
    ref = None
    if args.compare:
        with open(args.compare) as f:
            ref = json.load(f)
        seqs = ref["seqs"]
    else:
        gen = SamplingParams(max_tokens=args.tokens, temperature=0.0, ignore_eos=True)
        seqs = []
        for q in PROMPTS:
            ids = tok.encode(
                tok.apply_chat_template(
                    [{"role": "user", "content": q}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
            out = llm.generate(
                [TokensPrompt(prompt_token_ids=ids)], gen, use_tqdm=False
            )
            seqs.append(ids + list(out[0].outputs[0].token_ids))

    # prompt_logprobs=0 returns the top-1 id and logprob at every position.
    lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
    res = []
    for ids in seqs:
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], lp, use_tqdm=False)
        rows = []
        for d in out[0].prompt_logprobs[1:]:  # position 0 has no context
            if not d:
                continue
            i, v = next(iter(d.items()))
            rows.append([i, round(float(v.logprob), 4)])
        res.append(rows)
    with open(args.out, "w") as f:
        json.dump({"seqs": seqs, "rows": res}, f)

    if not args.compare:
        print(f"wrote {sum(len(r) for r in res)} positions to {args.out}")
        return

    base = ref["rows"]
    tot = hit = 0
    dlp = 0.0
    for bi, ni in zip(base, res):
        for (bid, bl), (nid, nl) in zip(bi, ni):
            tot += 1
            hit += bid == nid
            dlp += abs(bl - nl)
    print(
        f"teacher-forced top-1 agreement {hit}/{tot} = {hit / tot:.4f} "
        f"({100 * (1 - hit / tot):.2f} points lost), "
        f"mean |dlogprob| {dlp / tot:.4f}"
    )


if __name__ == "__main__":
    main()
