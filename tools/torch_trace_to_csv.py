# SPDX-License-Identifier: MIT
"""Converts a torch profiler Chrome trace to rocprofv3's kernel-trace CSV
columns (Start_Timestamp, End_Timestamp, Kernel_Name; ns) for analyze_trace.py.
"""

import argparse
import csv
import gzip
import json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("trace", help="*.pt.trace.json[.gz]")
    p.add_argument("out")
    args = p.parse_args()
    opener = gzip.open if args.trace.endswith(".gz") else open
    with opener(args.trace, "rt") as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Start_Timestamp", "End_Timestamp", "Kernel_Name"])
        for e in kernels:
            start = int(float(e["ts"]) * 1e3)
            w.writerow([start, start + int(float(e["dur"]) * 1e3), e["name"]])
    print(f"{len(kernels)} kernels -> {args.out}")


if __name__ == "__main__":
    main()
