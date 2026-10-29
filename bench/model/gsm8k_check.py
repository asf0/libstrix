# SPDX-License-Identifier: MIT
"""Small GSM8K accuracy check for plugin A/B runs (greedy, thinking off).

STRIX_DISABLE=all python bench/model/gsm8k_check.py --model M
python bench/model/gsm8k_check.py --model M
"""

import argparse
import json
import re

from datasets import load_dataset
from vllm import LLM, SamplingParams

INSTRUCTION = "Solve the problem. End with a line 'Answer: <number>'.\n\n"


def last_number(text: str) -> str | None:
    m = re.search(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text)
    nums = [m.group(1)] if m else re.findall(r"-?[\d,]*\.?\d+", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--save", help="write per-question correctness (JSON list)")
    p.add_argument("--mtp", type=int, default=0, help="MTP draft tokens, 0 = off")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--mamba-ssm-cache-dtype", default="auto")
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
        help="<= 5 keeps decode on the skinny (M <= 5) GEMM paths",
    )
    p.add_argument("--attention-backend", default="TRITON_ATTN")
    p.add_argument(
        "--engram-disk-offload",
        action="store_true",
        help="Qwen3.8-Flash-Next: n-gram table read from disk",
    )
    args = p.parse_args()

    data = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    llm = LLM(
        model=args.model,
        language_model_only=True,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        attention_backend=args.attention_backend or None,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        **({"engram_config": {"disk_offload": True}} if args.engram_disk_offload else {}),
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": args.mtp} if args.mtp else None
        ),
    )
    msgs = [[{"role": "user", "content": INSTRUCTION + d["question"]}] for d in data]
    outs = llm.chat(
        msgs,
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )
    hits = []
    for d, o in zip(data, outs):
        gold = d["answer"].split("####")[-1].strip().replace(",", "")
        pred = last_number(o.outputs[0].text)
        hits.append(pred is not None and float(pred) == float(gold))
    print(f"GSM8K {sum(hits)}/{args.n} = {sum(hits) / args.n:.3f}")
    if args.save:
        with open(args.save, "w") as f:
            json.dump(hits, f)


if __name__ == "__main__":
    main()
