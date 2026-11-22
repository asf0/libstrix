"""Inkling2 prefill attention (`_rel_attention_kernel`) launch sweep on
gfx1151 with Inkling 2 Mini shapes."""
import itertools

import torch
import triton

from vllm.models.inkling2.amd.ops import attention as A

dev = torch.device("cuda")
T, H, KVH, D, PAGE, EXT = 5128, 24, 8, 128, 64, 1024
nblocks = triton.cdiv(T, PAGE) + 1
torch.manual_seed(0)
kv = torch.randn(nblocks, KVH, PAGE, 2 * D, device=dev, dtype=torch.bfloat16)
key_cache, value_cache = kv.transpose(1, 2).split(D, dim=-1)
q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
bias = (torch.randn(T + 128, H, EXT + 256, device=dev) * 0.1).to(torch.bfloat16)
block_table = torch.arange(nblocks, device=dev, dtype=torch.int32)[None]
seqlens = torch.tensor([T], device=dev, dtype=torch.int32)
cu = torch.tensor([0, T], device=dev, dtype=torch.int32)
group = H // KVH
block_h = triton.next_power_of_2(group)


def launch(window, block_m, block_n, warps, stages, out, lse):
    block_q = block_m // block_h
    grid = (triton.cdiv(T, block_q), KVH, 1)
    A._rel_attention_kernel[grid](
        q, key_cache, value_cache, bias, out, lse, block_table, seqlens, cu,
        D**-0.5 * A._LOG2E, 0, T, q.stride(0), q.stride(1), key_cache.stride(0),
        key_cache.stride(1), key_cache.stride(2), value_cache.stride(0),
        value_cache.stride(1), value_cache.stride(2), bias.stride(0),
        bias.stride(1), out.stride(0), out.stride(1), lse.stride(0),
        block_table.stride(0), GROUP=group, D=D, PAGE=PAGE, E_PAD=bias.shape[2],
        WINDOW=window, DRAFT=False, SPLIT=False, BLOCK_Q=block_q, BLOCK_H=block_h,
        BLOCK_N=block_n, num_warps=warps, num_stages=stages)


def time_ms(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


import os
WINDOWS = ((1023, "local"),) if os.environ.get("STRIX_ONLY_LOCAL") else ((-1, "full"), (1023, "local"))
for window, name in WINDOWS:
    ref_out = q.new_empty(q.shape, dtype=torch.float32)
    ref_lse = q.new_empty((H, T), dtype=torch.float32)
    launch(window, 128, 64, 8, 1, ref_out, ref_lse)
    base = time_ms(lambda: launch(window, 128, 64, 8, 1, ref_out, ref_lse))
    print(f"{name}: current 128/64/w8/s1 {base:7.2f} ms", flush=True)
    for block_m, block_n, warps, stages in itertools.product(
        (32, 64, 128), (16, 32, 64), (2, 4, 8), (1, 2)
    ):
        out = torch.empty_like(ref_out)
        lse = torch.empty_like(ref_lse)
        try:
            launch(window, block_m, block_n, warps, stages, out, lse)
            ms = time_ms(lambda: launch(window, block_m, block_n, warps, stages, out, lse))
        except Exception as ex:  # noqa: BLE001
            continue
        err = (out - ref_out).abs().max().item()
        if ms < base * 0.9:
            print(f"{name}: {block_m}/{block_n}/w{warps}/s{stages} {ms:7.2f} ms "
                  f"({base / ms:.2f}x) err {err:.1e}", flush=True)
