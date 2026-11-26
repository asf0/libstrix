"""Inkling 2 decode attention at op level on the served shapes: 1 and 3 rows
(decode, the recipe's verify) at 2k / 32k / 131k keys and a 1024-key window,
bf16 and int8 KV, `strix::inkling_rel_attn_decode` with 1, 3 and 5 query rows
per block and the port's Triton kernel (bf16 only), over split counts. Calls
rotate over copies of the KV, so no call finds its keys in the MALL. Every
output is checked against the dense fp32 reference of the kernel tests; a
wrong strix output exits 1.

    MODELS=... bench/container/run.sh python bench/kernels/inkling_rel_decode.py
"""
import argparse
import functools
import pathlib
import statistics
import sys

import strix  # noqa: F401  (registers the ops)
import torch
from vllm.models.inkling2.amd.ops.attention import (
    plan_decode_attention,
    rel_decode_attention,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
from test_inkling_rel_attn_decode import (  # noqa: E402
    D,
    HKV,
    HQ,
    SCALE,
    build_verify,
    head_error,
    verify_reference,
)

p = argparse.ArgumentParser()
p.add_argument("--rows", default="1,3")
p.add_argument("--depths", default="2048,32768,131072")
p.add_argument("--kv", default="bf16,int8")
p.add_argument("--splits", default="8,12,16,20,24,32,40")
p.add_argument("--q-rows", default="1,3,5")
p.add_argument("--page", type=int, default=64)
p.add_argument("--rounds", type=int, default=7)
a = p.parse_args()


def ints(s):
    return [int(x) for x in s.split(",")]


def us_per_call(fn, calls):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(calls):
        fn(i)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / calls * 1e3


def bench(rows, depth, window, int8) -> bool:
    """Times every variant and split count of one batch; False if a strix
    output is wrong."""
    keys = depth if window < 0 else min(depth, window + rows)
    kv_bytes = keys * HKV * 2 * (D + 4 if int8 else 2 * D)
    layers = max(1, -(-(96 << 20) // kv_bytes))
    args, kv = build_verify([rows] * layers, [depth - rows] * layers, a.page, int8)
    q, k, v, ks, vs, rel, gate, sinks, _, seq_lens, table = args
    cu = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    rel = rel[: rows + 128]

    def layer(i):
        """The arguments of copy `i` of the request (its own KV)."""
        i %= layers
        return (q[:rows], k, v, ks, vs, rel, gate[:rows], sinks[:rows], cu,
                seq_lens[i : i + 1], table[i : i + 1])

    def op(n, i, splits):
        out = torch.empty((rows, HQ * D), dtype=q.dtype, device="cuda")
        torch.ops.strix.inkling_rel_attn_decode(out, *layer(i), SCALE, splits, window, n)
        return out

    def port(i, splits):
        ql, kl, vl, _, _, rl, gl, sl, cul, sql, tl = layer(i)
        out = torch.empty((rows, HQ * D), dtype=q.dtype, device="cuda")
        rel_decode_attention(
            ql, kl, vl, rl, gl, sl, out, block_table=tl, cache_seqlens=sql,
            cu_seqlens_q=cul, max_query_len=rows, num_actual_tokens=rows,
            softmax_scale=SCALE, is_local=window >= 0, num_splits=splits,
        )
        return out

    variants = {f"strix q{n}": functools.partial(op, n) for n in ints(a.q_rows)}
    if not int8:
        variants["port"] = port
    ref = verify_reference(layer(0), kv, window)
    del kv
    torch.cuda.empty_cache()
    planned = plan_decode_attention(window >= 0, 1, rows, HQ, HKV, 1024)
    splits = sorted({*ints(a.splits), planned})

    def counts(name):  # one row per block does not split
        return [planned] if name == "strix q1" else splits

    ok, times = True, {}
    for name, fn in variants.items():
        for s in counts(name):
            err = head_error(fn(0, s), ref)
            times[name, s] = []
            if name != "port" and not err < 1e-2:
                ok = False
                print(f"BAD {name} splits {s} error {err:.2e}", flush=True)
            times[name, "err"] = max(err, times.get((name, "err"), 0.0))
    for _ in range(a.rounds):
        for name, fn in variants.items():
            for s in counts(name):
                run = functools.partial(fn, splits=s)
                times[name, s].append(us_per_call(run, max(8, layers)))
    print(
        f"rows {rows} keys {depth} window {window} {'int8' if int8 else 'bf16'}: "
        f"{kv_bytes / 1e6:.1f} MB of KV, {kv_bytes / 210e3:.0f} us at 210 GB/s",
        flush=True,
    )
    for name in variants:
        med = {s: statistics.median(times[name, s]) for s in counts(name)}
        best = min(med, key=med.get)
        row = " ".join(f"{s}:{t:.0f}" for s, t in med.items())
        print(
            f"  {name:9s} planned {planned}: {med[planned]:7.1f} us, best {best}: "
            f"{med[best]:7.1f} us ({kv_bytes / med[best] / 1e3:.0f} GB/s), "
            f"error {times[name, 'err']:.1e} | {row}",
            flush=True,
        )
    return ok


good = True
for depth in ints(a.depths):
    for window in (-1, 1023) if depth == min(ints(a.depths)) else (-1,):
        for kv_dtype in a.kv.split(","):
            for rows in ints(a.rows):
                good &= bench(rows, depth, window, kv_dtype == "int8")
                torch.cuda.empty_cache()
sys.exit(0 if good else 1)
