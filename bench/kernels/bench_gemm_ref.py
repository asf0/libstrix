"""Reference GEMM throughput on gfx1151 for Inkling 2 Mini expert shapes:
hipBLASLt (torch.mm) and a plain Triton bf16 tl.dot kernel.

    IMAGE=strix-vllm:inkling MODELS=/mnt/8tb-usb/models bench/container/run.sh \
        python bench/kernels/bench_gemm_ref.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, B, C, M, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + rows[:, None] * K + k + ks[None, :], (rows < M)[:, None], other=0.0)
        b = tl.load(B + cols[:, None] * K + k + ks[None, :])
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(C + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16), (rows < M)[:, None])


def time_us(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters


for m in (128, 512, 2048):
    for n, k in ((2304, 2304), (2304, 1152)):
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        flops = 2 * m * n * k
        us = time_us(lambda: torch.mm(a, b.t()))
        print(f"M{m:5d} N{n} K{k} hipBLASLt          {us:8.1f} us {flops / us / 1e6:6.1f} TFLOPS")
        for bm, bn, bk, nw in ((64, 64, 32, 4), (64, 128, 32, 4), (128, 128, 32, 8), (64, 64, 64, 4)):
            if m < bm:
                continue
            fn = lambda: _gemm[(n // bn, triton.cdiv(m, bm))](a, b, c, m, N=n, K=k, BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=2)
            try:
                us = time_us(fn)
            except Exception as ex:  # noqa: BLE001
                print(f"  triton {bm}x{bn}x{bk} w{nw}: {type(ex).__name__}")
                continue
            print(f"M{m:5d} N{n} K{k} triton {bm}x{bn}x{bk} w{nw} {us:8.1f} us {flops / us / 1e6:6.1f} TFLOPS")
