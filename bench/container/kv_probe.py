"""KV GiB one max-model-len request needs, per recipe target, from vLLM's own
refusal with dummy weights and a 256 MiB KV budget. Usage:

    IMAGE=... MODELS=... python kv_probe.py LABEL target...
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

LIBSTRIX = Path("/home/ubuntu/local-llm/engines/libstrix")
HARNESS = Path("/home/ubuntu/local-llm/worktrees/libstrix-vllm-main")
os.environ.setdefault("CACHE", str(LIBSTRIX / "docker" / "recipes" / "cache"))
sys.path.insert(0, str(HARNESS / "bench" / "container"))
from common import Server, parse_target  # noqa: E402

PROBE_ARGS = "--load-format dummy --enforce-eager --kv-cache-memory-bytes=268435456"


def main() -> None:
    label, targets = sys.argv[1], sys.argv[2:]
    out = LIBSTRIX / "profiles" / "pool-compare" / label
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for target in targets:
        recipe, env = parse_target(target)
        env["EXTRA_ARGS"] = PROBE_ARGS
        s = Server(recipe, env, timeout=900)
        logs = ""
        try:
            with s:
                logs = s.logs()
        except RuntimeError as e:
            logs = str(e)
        m = re.findall(r"\(([\d.]+) GiB KV cache is needed", logs)
        results[target] = float(m[-1]) if m else None
        if not m:
            (out / f"probe-{re.sub(r'[^A-Za-z0-9._=-]+', '_', target)}.log").write_text(logs)
        print(target, results[target], flush=True)
    (out / "kv_probe.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
