"""Launch sweeps of Inkling2 (ROCm port) prefill kernels on the real tensors of
a 5k-token prefill: the first large call of each patched op times its kernel
over a grid of launch configs (scratch outputs, bit-exact check against the
current launch), then runs as before.

    IMAGE=strix-vllm:inkling MODELS=... VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
        bench/container/run.sh python bench/model/inkling_insitu_sweep.py
"""

import itertools
import random

import torch

from vllm import LLM, SamplingParams
from vllm.models.inkling2.amd import sconv as sconv_mod
from vllm.models.inkling2.amd.ops import sconv as ops
from vllm.triton_utils import triton

MIN_TOKENS = 4096
# Configs whose time is reported for every call variant.
PROBE = {"attn_res": ((4096, 4), (4096, 8), (2048, 4)), "gate_sink": ((4096, 8),)}
done: set[str] = set()


def timed(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def report(name, results, current):
    results.sort()
    cur = [r for r in results if r[1] == current]
    print(f"SWEEP {name} current {current}: {cur[0][0]:.3f} ms" if cur else f"SWEEP {name}")
    for t, cfg, exact in results[:5]:
        print(f"SWEEP {name} {t:.3f} ms {cfg} exact {exact}")
    for t, cfg, exact in results:
        if cfg in PROBE.get(name.split()[0], ()):
            print(f"PROBE {name} {t:.3f} ms {cfg} exact {exact}")


orig_short_conv = ops.short_conv


def short_conv_sweep(x, weight, cache, positions, block_table, seq_idx, slot_mapping,
                     query_start, stream_offset, stream_width, addend=None,
                     residual=None, out=None):
    num_tokens, channels = x.shape
    key = f"short_conv res={residual is not None} add={addend is not None}"
    if num_tokens >= MIN_TOKENS and key not in done:
        done.add(key)
        res = None if residual is None else residual.clone()
        num_heads = cache.shape[1]
        width = weight.shape[1]

        def launch(bt, bc, nw, dst):
            grid = (triton.cdiv(num_tokens, bt), triton.cdiv(channels, bc))
            ops._sconv_kernel[grid](
                x, cache, weight, dst, res, addend, positions, seq_idx, slot_mapping,
                block_table, query_start, num_tokens, x.stride(0), cache.stride(0),
                cache.stride(1), cache.stride(2), weight.stride(0), weight.stride(1),
                block_table.stride(0), block_table.shape[1], cache.shape[2],
                (width + 1) ** -0.5, W=width, OFF=stream_offset, WS=stream_width,
                H=num_heads, HAS_RES=res is not None, HAS_ADDEND=addend is not None,
                BT=bt, BLOCK_C=bc, SKIP_IDLE_CACHE=True, num_warps=nw,
                enable_fp_fusion=False)

        dtype = cache.dtype
        ref = torch.empty((num_tokens, channels), dtype=dtype, device=x.device)
        launch(4, 256, 4, ref)
        results = []
        for bt, bc, nw in itertools.product((1, 2, 4, 8, 16), (128, 256, 512, 1024), (1, 2, 4, 8)):
            dst = torch.empty_like(ref)
            try:
                t = timed(lambda: launch(bt, bc, nw, dst))
            except Exception:
                continue
            results.append((t, (bt, bc, nw), torch.equal(dst, ref)))
        report(key, results, (4, 256, 4))
    return orig_short_conv(x, weight, cache, positions, block_table, seq_idx,
                           slot_mapping, query_start, stream_offset, stream_width,
                           addend=addend, residual=residual, out=out)


ops.short_conv = short_conv_sweep
sconv_mod.short_conv = short_conv_sweep

from vllm.models.inkling2.amd import loop as loop_mod  # noqa: E402
from vllm.models.inkling2.amd import model as model_mod  # noqa: E402
from vllm.models.inkling2.amd.ops import attention as attn_ops  # noqa: E402
from vllm.models.inkling2.amd.ops import attn_res as res_ops  # noqa: E402

orig_gate_sink = attn_ops.gate_sink


def gate_sink_sweep(output, gate, lse, sinks, result):
    num_tokens, num_heads = sinks.shape
    if num_tokens >= MIN_TOKENS and "gate_sink" not in done:
        done.add("gate_sink")
        width = gate.shape[1]

        def launch(block, nw, dst):
            attn_ops._gate_sink_kernel[(num_tokens, triton.cdiv(width, block))](
                output, gate, lse, sinks, dst, output.stride(0), gate.stride(0),
                lse.stride(0), lse.stride(1), sinks.stride(0), num_heads,
                width // num_heads, block, num_warps=nw, enable_fp_fusion=False)

        ref = torch.empty_like(result)
        launch(512, 4, ref)
        results = []
        for block, nw in itertools.product((256, 512, 1024, 2048, 4096), (1, 2, 4, 8)):
            dst = torch.empty_like(result)
            try:
                t = timed(lambda: launch(block, nw, dst))
            except Exception:
                continue
            results.append((t, (block, nw), torch.equal(dst, ref)))
        report("gate_sink", results, (512, 4))
    return orig_gate_sink(output, gate, lse, sinks, result)


attn_ops.gate_sink = gate_sink_sweep

orig_attn_res = res_ops.attn_res


def attn_res_sweep(stack, last, addend, queries, norm_weight, norm_eps,
                   normalize_values, eps=1e-6):
    num_tokens, d = last.shape
    key = f"attn_res q={queries.shape[0]} n={stack.shape[0]} norm={norm_weight is not None}"
    if num_tokens >= MIN_TOKENS and key not in done:
        done.add(key)
        num_q = queries.shape[0]
        apply_norm = norm_weight is not None
        nw_t = norm_weight if apply_norm else queries
        out_dtype = last.dtype if apply_norm else torch.float32
        meta = dict(D=d, N_STACK=stack.shape[0], HAS_ADDEND=addend is not None,
                    NORMALIZE=normalize_values, APPLY_NORM=apply_norm,
                    BLOCK_N=triton.next_power_of_2(stack.shape[0] + 1),
                    enable_fp_fusion=False)

        def launch(bd, nw, dst):
            res_ops._attn_res_kernel[(num_q, num_tokens)](
                stack, last, addend, queries, nw_t, dst, stack.stride(0), num_tokens,
                eps, norm_eps, **meta, BLOCK_D=bd, num_warps=nw)

        ref = torch.empty((num_q, num_tokens, d), dtype=out_dtype, device=last.device)
        launch(1024, 4, ref)
        results = []
        for bd, nw in itertools.product((256, 512, 1024, 2048, 4096), (1, 2, 4, 8)):
            dst = torch.empty_like(ref)
            try:
                t = timed(lambda: launch(bd, nw, dst))
            except Exception:
                continue
            results.append((t, (bd, nw), torch.equal(dst, ref)))
        report(key, results, (1024, 4))
    return orig_attn_res(stack, last, addend, queries, norm_weight, norm_eps,
                         normalize_values, eps)


res_ops.attn_res = attn_res_sweep
loop_mod.attn_res = attn_res_sweep
model_mod.attn_res = attn_res_sweep

llm = LLM("/models/Inkling-2-Mini", max_model_len=16384, max_num_seqs=4,
          max_num_batched_tokens=8192, gpu_memory_utilization=0.8,
          enforce_eager=True, enable_prefix_caching=False)
words = "amber basil cedar delta ember fable garnet harbor indigo juniper".split()
rng = random.Random(0)
text = " ".join(rng.choice(words) for _ in range(4000))
llm.chat([{"role": "user", "content": "Summarize: " + text}], SamplingParams(max_tokens=1))
