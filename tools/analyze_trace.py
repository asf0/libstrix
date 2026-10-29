# SPDX-License-Identifier: MIT
"""Per-phase kernel breakdown of a profile_model.py rocprofv3 trace.

Bursts (separated by idle gaps) are matched to the phase labels, last first.
Prefill phases report per-step kernel time; decode phases report
(long - short) / decode_steps, i.e. the time of one decode step.
"""

import argparse
import csv
import json
import re
from collections import defaultdict

CATEGORIES = [
    # Qwen3.8-Flash-Next (qwen4_exp). Ahead of the generic patterns: the
    # indexer's top_k and the QSA kernels would otherwise land in moe/sampling.
    ("qsa_indexer", r"_qsa_mqa_paged|top_k_per_row|_expand_qsa"),
    ("qsa_attn", r"_qsa_sparse|_qsa_merge"),
    ("qsa_prep", r"_qsa_prepare|_store_qsa|_compress_qsa"),
    ("hc", r"_hc_|grouped_gemma_rmsnorm"),
    ("ple", r"ple_|ngram"),
    ("moe",r"fused_moe|moe_|topk|_moe_align|count_and_sort"),
    # W8A16/W4A16 linears: the mainloop, the activation-quantize pass and the
    # split-K reduce are separate kernels. Which of the three dominates is what
    # decides the next tile change, so they get their own buckets instead of
    # disappearing into `gemm`.
    ("w8a8_gemm", r"w8a8_gemm_group|w4a16_gemm_kernel|w4a8_gemm"),
    ("w8a8_actq", r"act_quant"),
    ("w8a8_reduce", r"w8a8_reduce|reduce_splitk"),
    ("gemm", r"^Cijk|gemm|Gemm|wvSplitK|LLMM|wvSpltK|wvSplit"),
    (
        "attention",
        r"unified_attention|reduce_segments|paged_attention|attn_prefill|attn_split_reduce",
    ),
    ("gdn_chunk", r"chunk_|recompute_w_u|solve_tril|merge_16x16|kkt"),
    ("gdn_recurrent", r"recurrent|fused_sigmoid_gating|recoverssm"),
    (
        "gdn_conv",
        r"causal_conv1d|_fused_post_conv|conv_fwd_kernel|conv_state_kernel|post_conv_kernel",
    ),
    ("norm", r"layer_norm|rms_norm|rmsnorm|fused_norm_gate|l2norm|gated_rmsnorm"),
    ("kv_write", r"reshape_and_cache|reshape_cache"),
    ("sampling", r"sample|argmax|softmax|top_k|logits"),
    ("triton_fused", r"^triton_"),
    ("torch_eager", r"at::native|elementwise"),
]


def category(name: str) -> str:
    for label, pat in CATEGORIES:
        if re.search(pat, name):
            return label
    return "other"


def bursts(path: str, gap_s: float) -> list[list[tuple[int, int, str]]]:
    with open(path, newline="") as f:
        rows = sorted(
            (int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"])
            for r in csv.DictReader(f)
        )
    out, cur, last = [], [], None
    for row in rows:
        if last is not None and row[0] - last > gap_s * 1e9:
            out.append(cur)
            cur = []
        cur.append(row)
        last = row[1] if last is None else max(last, row[1])
    out.append(cur)
    return out


def summarize(burst) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    kers, cnt = defaultdict(float), defaultdict(int)
    for s, e, n in burst:
        short = re.sub(r"\(.*", "", n.replace("(anonymous namespace)::", ""))[:90]
        kers[short] += (e - s) / 1e9
        cnt[short] += 1
    cats = defaultdict(float)
    for k, d in kers.items():
        cats[category(k)] += d
    return cats, kers, cnt


def report(title, cats, kers, cnt, top) -> None:
    busy = sum(cats.values())
    print(f"\n=== {title}: GPU busy {busy * 1e3:.2f} ms")
    for c, d in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:14s} {d * 1e3:9.2f} ms {100 * d / busy:5.1f}%")
    for k, d in sorted(kers.items(), key=lambda x: -x[1])[:top]:
        print(f"    {d * 1e3:9.3f} ms n={cnt[k]:6.1f} {category(k):13s} {k}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("labels")
    p.add_argument("--gap-s", type=float, default=2.5)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--json", help="write per-phase category ms here")
    args = p.parse_args()
    with open(args.labels) as f:
        meta = json.load(f)
    labels, steps = meta["labels"], meta["decode_steps"]
    bs = bursts(args.trace, args.gap_s)
    assert len(bs) >= len(labels), f"{len(bs)} bursts < {len(labels)} phases"
    by_label = dict(zip(labels, bs[-len(labels) :]))
    out = {}

    for label, b in by_label.items():
        if label.startswith("prefill") and label.endswith("#1"):
            cats, kers, cnt = summarize(b)
            report(label, cats, kers, cnt, args.top)
            out[label] = {k: v * 1e3 for k, v in cats.items()}
    for label in labels:
        if not label.endswith(":long"):
            continue
        base = label[: -len(":long")]
        cl, kl, nl = summarize(by_label[label])
        cs, ks, ns = summarize(by_label[base + ":short"])
        cats = {c: (cl[c] - cs.get(c, 0.0)) / steps for c in cl}
        kers = {k: (kl[k] - ks.get(k, 0.0)) / steps for k in kl}
        cnt = {k: (nl[k] - ns.get(k, 0)) / steps for k in nl}
        report(f"{base} per decode step", cats, kers, cnt, args.top)
        out[base] = {k: v * 1e3 for k, v in cats.items()}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
