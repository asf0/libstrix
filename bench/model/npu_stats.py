# SPDX-License-Identifier: MIT
# Summarizes the JSONL written by STRIX_NPU_STATS (python/strix_vllm/
# qwen4_exp_npu.py): per prefill step, one line with per-chain NPU busy,
# kick-to-done, the GPU window the chain had and the time the GPU actually
# spent waiting at the chain's wait (the true stall), plus per-bank GPU times.
#
#   python bench/model/npu_stats.py profiles/npu-stats/npu-idx.jsonl [--skip 1]
#
# Attribution: take_stats is read at host-enqueue time, so a call returns the
# stats of chains completed since the previous call - typically the chain
# before the current one. Chains complete in kick order and entries are in
# kick order, so the stats are walked through a FIFO. window_ms / wait_ms
# come from this chain's own events and need no shifting.
#
# Reading it: busy < window says the chain hides; wait ~ 0 confirms it;
# sum(wait) is the step's NPU-bound time.

import argparse
import json
from collections import defaultdict, deque


def attributed(step):
    """Chains with busy/k2d moved to the chain they belong to (FIFO; see the
    module docstring). Unattributed chains keep busy_ms None."""
    out, q = [], deque()
    for c in step["chains"]:
        ch = dict(c)
        out.append(ch)
        q.append(ch)
        n = c["chains"]
        while n > 0 and q:
            done = q.popleft()
            if n == 1:
                done["busy_ms"], done["k2d_ms"] = c["busy_ms"], c["k2d_ms"]
            n -= 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--skip", type=int, default=0, help="drop the first N steps")
    args = p.parse_args()

    steps = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    steps = [s for s in steps if not s.get("profile")][args.skip:]

    by_depth = defaultdict(list)
    for s in steps:
        by_depth[s["n"]].append(s)

    for depth in sorted(by_depth):
        ss = by_depth[depth]
        print(f"=== {depth}-token steps: {len(ss)}")
        for s in ss:
            chains = attributed(s)
            ok = [c for c in chains if c.get("busy_ms") is not None and c.get("window_ms") is not None]
            wait = [c.get("wait_ms", 0.0) for c in ok]
            win = [c["window_ms"] for c in ok]
            busy = [c["busy_ms"] for c in ok]
            banks = [b["ms"] for b in s["banks"]]
            marks = [b for b in s["banks"] if b["layer"] < 0]
            pre = [b["ms"] for b in s["banks"] if b["layer"] == -3]
            print(f"  wall {s['wall_ms']:7.0f} | bank GPU {sum(banks):7.0f} "
                  f"(mean {sum(banks)/max(1,len(banks)):5.1f}) | chains {len(ok)}"
                  f" busy {sum(busy)/max(1,len(ok)):5.1f} win {sum(win)/max(1,len(ok)):5.1f} "
                  f"| GPU wait at chain: {sum(wait):6.1f} ms/step (mean {sum(wait)/max(1,len(ok)):4.2f})")
        # aggregate over the depth's steps
        chains = [c for s in ss for c in attributed(s)]
        ok = [c for c in chains if c.get("busy_ms") is not None and c.get("window_ms") is not None]
        for rows in sorted({c["rows"] for c in ok}):
            cs = [c for c in ok if c["rows"] == rows]
            print(f"  rows {rows}: n={len(cs)} busy {sum(c['busy_ms'] for c in cs)/len(cs):.1f} "
                  f"k2d {sum(c['k2d_ms'] for c in cs)/len(cs):.1f} win {sum(c['window_ms'] for c in cs)/len(cs):.1f} "
                  f"wait {sum(c.get('wait_ms', 0) for c in cs)/len(cs):.2f}")
        w = [s["wake"] for s in ss if s.get("wake")]
        if w:
            print(f"  wake: busy {sum(x['busy_ms'] for x in w)/len(w):.1f} ms "
                  f"k2d {sum(x['k2d_ms'] for x in w)/len(w):.1f} ms (chains {sum(x['chains'] for x in w)/len(w):.1f})")


if __name__ == "__main__":
    main()
