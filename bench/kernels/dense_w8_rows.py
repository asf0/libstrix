# SPDX-License-Identifier: MIT
"""dense_w8 at 1-8 rows: the int8 skinny GEMM against the bf16 path it replaces.

A speculative verify step runs every dense linear on 1 + K rows (MTP 2: 3,
DFlash 2 with 4-7 draft tokens: 5-8). Below the dense_w8 row limit those
linears read the int8 copy (strix::w8a16_skinny_gemm); above it the bf16
weight through vLLM's ROCm unquantized GEMM, twice the bytes. This times both
on Qwen3.8-Flash-Next's bf16 linears to place the limit.

    python bench/kernels/dense_w8_rows.py
"""
import argparse

import strix  # noqa: F401  (registers torch.ops.strix)
import torch

from strix_vllm.dense_w8 import _quantize

# Qwen3.8-Flash-Next bf16 linears (N, K): GDN in_proj_qkvz and out_proj, QSA
# qkv and o_proj, shared expert gate_up, LM head.
SHAPES = {
    "gdn_qkvz": (16384, 2560),
    "gdn_out": (2560, 6144),
    "qsa_qkv": (13312, 2560),
    "qsa_o": (2560, 6144),
    "shared_gate_up": (1280, 2560),
    "lm_head": (248320, 2560),
}


def bench(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps * 1e3  # us


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="1,3,5,6,7,8")
    p.add_argument("--reps", type=int, default=50)
    args = p.parse_args()
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    cu = torch.cuda.get_device_properties(0).multi_processor_count
    torch.manual_seed(0)
    print(f"{'layer':16} {'rows':>4} {'int8 us':>9} {'bf16 us':>9} {'speedup':>8} {'err':>7}")
    for name, (n, k) in SHAPES.items():
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        w_q, w_s, group = _quantize(w)
        for m in (int(x) for x in args.rows.split(",")):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            want = (x.float() @ w.float().t())
            got = torch.ops.strix.w8a16_skinny_gemm(w_q, x, w_s, None, cu, group).float()
            err = ((got - want).norm() / want.norm()).item()
            t8 = bench(
                lambda: torch.ops.strix.w8a16_skinny_gemm(w_q, x, w_s, None, cu, group),
                args.reps,
            )
            t16 = bench(lambda: rocm_unquantized_gemm_impl(x, w, None), args.reps)
            print(f"{name:16} {m:4d} {t8:9.1f} {t16:9.1f} {t16 / t8:7.2f}x {err:7.4f}")
        del w, w_q, w_s


if __name__ == "__main__":
    main()
