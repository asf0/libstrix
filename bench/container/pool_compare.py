"""Compare KV capacity, decode speed and draft acceptance of one image over
several recipe targets. Usage:

    IMAGE=... MODELS=/mnt/8tb-usb/models python pool_compare.py LABEL target...

Writes profiles/pool-compare/<LABEL>/<target>.json and .log under libstrix.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LIBSTRIX = Path("/home/ubuntu/local-llm/engines/libstrix")
HARNESS = Path("/home/ubuntu/local-llm/worktrees/libstrix-vllm-main")
os.environ.setdefault("CACHE", str(LIBSTRIX / "docker" / "recipes" / "cache"))
sys.path.insert(0, str(HARNESS / "bench" / "container"))
from common import Server, parse_target  # noqa: E402

PROMPTS = [
    "Write a Python function that parses an ISO 8601 duration string like P3DT4H5M and returns the total seconds. Include docstring and tests.",
    "Explain how a B-tree insertion with node splitting works, step by step, with a small worked example.",
    "Write a short story (about 400 words) about a lighthouse keeper who finds a message in a bottle.",
    "Solve step by step: a train leaves at 9:40 going 84 km/h; a second leaves the same station at 10:15 going 108 km/h on the same track. When and where does the second catch the first?",
    "Implement a thread-safe LRU cache in C++17 with get and put, and explain the locking strategy.",
    "Summarize the causes and consequences of the 1929 stock market crash in a structured outline.",
    "Write a bash script that finds the 10 largest files under a directory, skipping .git, and prints human-readable sizes.",
    "Translate this into French and then explain three grammar points in it: 'If I had known you were coming, I would have baked a cake.'",
]
MAX_TOKENS = 512
EXTRA = {"ignore_eos": True} if os.environ.get("IGNORE_EOS") else {}


def drain_ttm() -> None:
    for _ in range(30):
        gtt = sum(
            int(p.read_text())
            for p in Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used")
        )
        if gtt < 2 << 30:
            break
        time.sleep(2)
    for _ in range(5):
        subprocess.run(
            ["sudo", "cat", "/sys/kernel/debug/ttm/page_pool_shrink"],
            capture_output=True,
        )


def spec_counters(s: Server) -> tuple[float, float]:
    return (
        s.metric("vllm:spec_decode_num_accepted_tokens_total"),
        s.metric("vllm:spec_decode_num_drafts_total"),
    )


def decode(s: Server, concurrency: int) -> dict:
    prompts = (PROMPTS * 2)[: max(concurrency, 3)]
    a0, d0 = spec_counters(s)
    t0 = time.time()
    if concurrency == 1:
        resps = [s.chat(p, max_tokens=MAX_TOKENS, **EXTRA) for p in prompts]
    else:
        with ThreadPoolExecutor(concurrency) as ex:
            resps = list(ex.map(lambda p: s.chat(p, max_tokens=MAX_TOKENS, **EXTRA), prompts))
    wall = time.time() - t0
    a1, d1 = spec_counters(s)
    errors = [r["error"] for r in resps if "error" in r]
    toks = sum(r["usage"]["completion_tokens"] for r in resps if "usage" in r)
    drafts = d1 - d0
    return {
        "concurrency": concurrency,
        "requests": len(prompts),
        "completion_tokens": toks,
        "wall_s": round(wall, 2),
        "tok_s": round(toks / wall, 1) if wall else None,
        "mean_accepted_per_step": round(1 + (a1 - a0) / drafts, 3) if drafts else None,
        "errors": errors[:2],
        "texts": [
            (m.get("content") or m.get("reasoning_content") or m.get("reasoning") or "")[:200]
            for r in resps
            if "choices" in r
            for m in [r["choices"][0]["message"]]
        ],
    }


def main() -> None:
    label, targets = sys.argv[1], sys.argv[2:]
    out = LIBSTRIX / "profiles" / "pool-compare" / label
    out.mkdir(parents=True, exist_ok=True)
    for target in targets:
        name = re.sub(r"[^A-Za-z0-9._=-]+", "_", target)
        if (out / f"{name}.json").exists():
            print(f"skip {target} (done)", flush=True)
            continue
        recipe, env = parse_target(target)
        env.setdefault("GPU_UTIL", "0.80")
        drain_ttm()
        print(f"=== {label} {target}", flush=True)
        result: dict = {"target": target, "image": os.environ.get("IMAGE"), "env": env}
        try:
            with Server(recipe, env) as s:
                result["startup"] = s.startup_numbers()
                result["kernels"] = s.kernels()
                s.chat("Hello", max_tokens=16)  # warm up
                result["c1"] = decode(s, 1)
                max_seqs = int(env.get("MAX_NUM_SEQS", "4" if "flash-next" in recipe else "8"))
                result[f"c{max_seqs}"] = decode(s, max_seqs)
                for rep in range(1, int(os.environ.get("REPEATS", "1"))):
                    result[f"c{max_seqs}_r{rep}"] = decode(s, max_seqs)
                (out / f"{name}.log").write_text(s.logs())
        except Exception as e:  # noqa: BLE001
            result["error"] = str(e)[-3000:]
        (out / f"{name}.json").write_text(json.dumps(result, indent=1))
        summary = {k: v for k, v in result.items() if k in ("startup", "error")}
        for k in result:
            if k.startswith("c") and isinstance(result[k], dict):
                summary[k] = {x: result[k][x] for x in ("tok_s", "mean_accepted_per_step", "errors")}
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
