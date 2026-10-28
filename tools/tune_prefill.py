# SPDX-License-Identifier: MIT
"""Re-derive the attn_prefill per-shape tune table (PrefillTune in
csrc/attn/attn_prefill_bf16.hip).

For every (mask_skip, hoist) combination the script forces that choice on
all shapes, rebuilds, and times each shape at an 8k chunk on 32k context with
bf16 and int8 KV. It then prints the STRIX_TUNE rows that pick the fastest
combination per (head_dim, GQA, KV dtype) and, with --write, puts them into
the source. The original source is restored on any failure, and the library
is rebuilt with whatever ends up in the file.

Run after any change to the prefill mainloop (register allocation moves,
and a flag that helped one shape has spilled another), or when a new model
adds a shape. One GPU job at a time: each measurement already goes
through bench/run_guarded.sh, so do not wrap the script in it.

    python tools/tune_prefill.py            # report only
    python tools/tune_prefill.py --write    # also update the table
"""

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "csrc/attn/attn_prefill_bf16.hip"
# (label, Hq, Hkv, head_dim, KV block size, window_left) - one per
# instantiated shape, windowed where the model only runs it windowed.
SHAPES = [
    ("256g6", 24, 4, 256, 784, -1),
    ("256g8", 16, 2, 256, 1056, -1),
    ("512g8", 32, 4, 512, 16, -1),
    ("64g4", 32, 8, 64, 16, -1),
    ("256g2", 32, 16, 256, 16, 1023),
]
TABLE_RE = re.compile(r"(?ms)^STRIX_TUNE\(.*?^#undef STRIX_TUNE\n")
PRIMARY = "  static constexpr bool mask_skip = true;\n  static constexpr bool hoist = false;\n"

MEASURE = r"""
import sys
sys.path.insert(0, {tests!r})
import strix, torch
from test_attn_prefill import build, quantize

def bench(fn, iters=3):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters

for name, hq, hkv, hd, bs, win in {shapes!r}:
    q, kc, vc, cu, sl, t = build([8192], [32768], hq, hkv, bs, hd=hd)
    out = torch.empty_like(q)
    k8, ks = quantize(kc, bs, hkv)
    v8, vs = quantize(vc, bs, hkv)
    a = bench(lambda: strix.attn_prefill(out, q, kc, vc, cu, sl, t, hd**-0.5, 1, win))
    b = bench(lambda: strix.attn_prefill(out, q, k8, v8, cu, sl, t, hd**-0.5, 1, win, ks, vs))
    print(f"RESULT {{name}} {{a:.2f}} {{b:.2f}}", flush=True)
    del q, kc, vc, k8, v8
    torch.cuda.empty_cache()
"""


def build() -> None:
    subprocess.run(["./build.sh"], cwd=ROOT, check=True, capture_output=True)


def measure() -> dict[str, tuple[float, float]]:
    code = MEASURE.format(tests=str(ROOT / "tests"), shapes=SHAPES)
    r = subprocess.run(
        ["bench/run_guarded.sh", sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    res = {}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            _, name, a, b = line.split()
            res[name] = (float(a), float(b))
    return res


def forced(src: str, mask: bool, hoist: bool) -> str:
    """Every shape takes (mask, hoist): the primary template is set and the
    specialization table is emptied."""
    flags = (
        f"  static constexpr bool mask_skip = {str(mask).lower()};\n"
        f"  static constexpr bool hoist = {str(hoist).lower()};\n"
    )
    out = src.replace(PRIMARY, flags, 1)
    return TABLE_RE.sub("#undef STRIX_TUNE\n", out, count=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--write", action="store_true", help="update the table in the source"
    )
    args = p.parse_args()
    orig = SRC.read_text()
    if PRIMARY not in orig or not TABLE_RE.search(orig):
        sys.exit("tune table not found in attn_prefill_bf16.hip")
    combos = list(itertools.product((False, True), repeat=2))
    times: dict[tuple[bool, bool], dict[str, tuple[float, float]]] = {}
    final = orig
    try:
        for mask, hoist in combos:
            SRC.write_text(forced(orig, mask, hoist))
            build()
            times[(mask, hoist)] = measure()
            print(f"mask={mask:d} hoist={hoist:d}: {times[(mask, hoist)]}", flush=True)
        rows = []
        for name, _, _, hd, _, _ in SHAPES:
            g = int(name.split("g")[1])
            for kv, q8 in ((0, "false"), (1, "true")):
                cell = {c: times[c][name][kv] for c in combos}
                best = min(cell, key=cell.get)
                ms = " / ".join(f"{cell[c]:.1f}" for c in combos)
                rows.append(
                    f"STRIX_TUNE({hd}, {g}, {q8}, {str(best[0]).lower()}, "
                    f"{str(best[1]).lower()})  // {ms}"
                )
        table = "\n".join(rows) + "\n#undef STRIX_TUNE\n"
        print("\n// (none / hoist / mask / both) ms at an 8k chunk @ 32k")
        print(table)
        if args.write:
            final = TABLE_RE.sub(table, orig, count=1)
    finally:
        SRC.write_text(final)
        build()


if __name__ == "__main__":
    main()
