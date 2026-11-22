"""NVFP4 grouped GEMM (bf16 rows) at a 5k-token Inkling 2 Mini prefill: the
current kernel (even/odd activation columns, fp8 scale conversion per value)
against the split-nibble layout (byte j of each 32-block = k+j low, k+16+j
high), contiguous activation halves and one bit-decoded scale per row."""
import itertools

import torch
import triton
import triton.language as tl

from strix_vllm import inkling2_moe as M
from vllm.models.inkling2.amd.ops.triton_bmm import ExpertPlan


@triton.jit
def _split_kernel(A, W, WScale, WAmax, WDenominator, Out, Tiles, max_ctas,
                  N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr):
    SUB: tl.constexpr = TILE // BM if TILE >= BM else 1
    pid_m = tl.program_id(1)
    tile = pid_m // SUB
    if tile >= tl.load(Tiles + 2 * max_ctas):
        return
    row_end = tl.load(Tiles + max_ctas + tile)
    row0 = tile * TILE + (pid_m % SUB) * BM
    if row0 >= row_end:
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    outer = tl.load(WAmax + expert) / tl.load(WDenominator)
    rows = row0 + tl.arange(0, BM)
    row_mask = (rows < row_end)[:, None]
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    j = tl.arange(0, 16)
    weight_rows = (expert * N + cols).to(tl.int64)
    a_ptrs = A + rows[:, None].to(tl.int64) * K + j[None, :]
    w_ptrs = W + weight_rows[:, None] * (K // 2) + j[None, :]
    ws_ptrs = WScale + weight_rows * (K // 16)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, 32):
        a_lo = tl.load(a_ptrs + k, row_mask, other=0.0)
        a_hi = tl.load(a_ptrs + k + 16, row_mask, other=0.0)
        packed = tl.load(w_ptrs + k // 2).to(tl.int32)
        s_lo = (M._e4m3_f32(tl.load(ws_ptrs + k // 16).to(tl.int32)) * outer).to(tl.bfloat16)
        s_hi = (M._e4m3_f32(tl.load(ws_ptrs + k // 16 + 1).to(tl.int32)) * outer).to(tl.bfloat16)
        acc = tl.dot(a_lo, tl.trans(M._e2m1_bf16(packed & 15) * s_lo[:, None]), acc)
        acc = tl.dot(a_hi, tl.trans(M._e2m1_bf16(packed >> 4) * s_hi[:, None]), acc)
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], acc.to(tl.bfloat16), row_mask)


def split_layout(weight):
    """[E, N, K/2] standard packing (byte j = k 2j low, 2j+1 high) -> split."""
    lo, hi = weight & 15, weight >> 4
    codes = torch.stack((lo, hi), -1).view(*weight.shape[:-1], -1, 32)  # k order
    return (codes[..., :16] | (codes[..., 16:] << 4)).reshape(weight.shape).contiguous()


dev = torch.device("cuda")
E, ROWS, TILE = 128, 320, 256
for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
    weight = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev)
    wsplit = split_layout(weight)
    scale = (torch.rand(E, n, k // 16, device=dev) + 0.5).to(torch.float8_e4m3fn)
    amax = torch.rand(E, device=dev) + 0.5
    den = torch.tensor([2688.0], device=dev)
    tpe = triton.cdiv(ROWS, TILE)
    max_ctas = E * tpe
    tiles = torch.zeros(2 * max_ctas + 3, dtype=torch.int32)
    t = 0
    for e in range(E):
        left = ROWS
        for _ in range(tpe):
            tiles[t] = e
            tiles[max_ctas + t] = t * TILE + min(left, TILE)
            left -= TILE
            t += 1
    tiles[2 * max_ctas] = max_ctas
    tiles = tiles.to(dev)
    rows_total = TILE * max_ctas
    act = torch.randn(rows_total, k, device=dev).to(torch.bfloat16)
    flops = 2 * E * ROWS * n * k

    def cur(bm, bn, nw, ns, out):
        M._nvfp4_bmm_bf16_kernel[(n // bn, max_ctas * (TILE // bm))](
            act, weight, scale, amax, den, out, tiles, max_ctas, N=n, K=k,
            TILE=TILE, BM=bm, BN=bn, num_warps=nw, num_stages=ns)

    def new(bm, bn, nw, ns, out):
        _split_kernel[(n // bn, max_ctas * (TILE // bm))](
            act, wsplit, scale.view(torch.uint8), amax, den, out, tiles, max_ctas,
            N=n, K=k, TILE=TILE, BM=bm, BN=bn, num_warps=nw, num_stages=ns)

    ref = torch.empty((rows_total, n), dtype=torch.bfloat16, device=dev)
    cur(64, 128, 4, 2, ref)
    valid = torch.cat([torch.arange(t0 * TILE, t0 * TILE + min(ROWS - (t0 % tpe) * TILE, TILE))
                       for t0 in range(max_ctas)]).to(dev)
    for label, fn, cfgs in (
        ("current", cur, [(64, 128, 4, 2)]),
        ("split", new, itertools.product((64, 128), (64, 128, 256), (4, 8), (1, 2, 3))),
    ):
        for bm, bn, nw, ns in cfgs:
            out = torch.zeros_like(ref)
            try:
                fn(bm, bn, nw, ns, out)
                torch.cuda.synchronize()
                s, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(5):
                    fn(bm, bn, nw, ns, out)
                e_.record()
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                continue
            ms = s.elapsed_time(e_) / 5
            err = ((out[valid].float() - ref[valid].float()).abs().max() / ref[valid].float().abs().max()).item()
            print(f"{name} {label:7s} {bm}x{bn} w{nw} s{ns} {ms:7.2f} ms {flops / ms / 1e9:5.1f} TFLOPS err {err:.1e}", flush=True)
