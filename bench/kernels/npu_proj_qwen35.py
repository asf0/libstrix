# SPDX-License-Identifier: MIT
"""The Qwen3.8-27B (Qwen3_5) GDN projection on the NPU: two K halves, two chains.

One layer's `in_proj_qkvz` + `in_proj_ba` (16,480 x 5,120) run as two 2,560-wide
chains (strix_vllm.qwen3_5_npu), summed in bf16, and are compared with the exact
arithmetic of that contract and with the bf16 GPU product; then timed against
the GPU's own GEMM at the same shapes.

    python bench/kernels/npu_proj_qwen35.py --model /models/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 \
        [--layer 1] [--rows 4096] [--pci 0000:c6:00.0]

Needs /dev/accel/accel0, a memlock limit of a few GB and libstrix_npu.so.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from strix_vllm import npu
from strix_vllm import qwen3_5_npu as q35


def grid15_up(v: np.ndarray) -> np.ndarray:
    f, ex = np.frexp(v.astype(np.float32))
    m = np.ceil((f * np.float32(32768.0)).astype(np.float32))
    e = ex - 15
    carry = m == 32768
    m[carry], e[carry] = 16384, e[carry] + 1
    s = (m * np.exp2(e.astype(np.float64))).astype(np.float32)
    s[v == 0] = 0
    return s


def scale_of(a: torch.Tensor) -> torch.Tensor:
    """One grid15-up scale per row, from the row's largest magnitude."""
    am = a.float().cpu().numpy().__abs__().max(axis=1)
    return torch.from_numpy(grid15_up((am.astype(np.float64) / 127.0).astype(np.float32)).astype(np.float64))


def codes(a: torch.Tensor, s: torch.Tensor, multiply: bool) -> torch.Tensor:
    a = a.float().cpu().numpy()
    sf = s.numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if multiply:  # activations: x * (1 / s), like the GPU producer
            inv = np.where(sf > 0, (np.float32(1) / sf.astype(np.float32)).astype(np.float32), np.float32(0))
            c = np.rint((a * inv[:, None]).astype(np.float32))
        else:  # weights: w / s, like the packer
            c = np.where(sf[:, None] > 0, np.rint((a / sf[:, None]).astype(np.float32)), 0)
    return torch.from_numpy(np.clip(c, -127, 127).astype(np.float64))


def emulate_halves(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """What the NPU path computes: one activation scale per row over the whole
    5,120-wide input, per-half weight scales, int32 dot, one bf16 rounding per
    half, then a bf16 add."""
    out = None
    for h in (0, 1):
        sl = slice(h * npu.K, (h + 1) * npu.K)
        xs = scale_of(x[:, sl])  # per half, as the path packs them
        ws_ = scale_of(w[:, sl])
        part = ((codes(x[:, sl], xs, True) @ codes(w[:, sl], ws_, False).T)
                * xs[:, None] * ws_[None, :]).to(torch.bfloat16)
        out = part if out is None else out + part
    return out


def load_layer(model: Path, layer: int) -> torch.Tensor:
    idx = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    pre = next(p for p in (f"model.language_model.layers.{layer}.", f"model.layers.{layer}.") if p + "linear_attn.in_proj_qkv.weight" in idx)
    parts = []
    for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        with safe_open(model / idx[pre + f"linear_attn.{n}.weight"], "pt") as f:
            parts.append(f.get_tensor(pre + f"linear_attn.{n}.weight"))
    return torch.cat(parts).to(torch.bfloat16)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--rows", type=int, default=4096)
    p.add_argument("--pci", default="0000:c6:00.0")
    p.add_argument("--reps", type=int, default=10)
    a = p.parse_args()
    torch.manual_seed(0)
    model = Path(a.model)
    cfg = json.loads((model / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    while cfg["layer_types"][a.layer] != "linear_attention":
        a.layer += 1
    w = load_layer(model, a.layer)
    n_live = w.shape[0]
    ba_cols = 2 * cfg["linear_num_value_heads"]
    assert w.shape[1] == 2 * npu.K and n_live == 10240 + 6144 + ba_cols, (w.shape, ba_cols)
    print(f"layer {a.layer}: weight {tuple(w.shape)} {w.dtype}", flush=True)

    t0 = time.time()
    runtimes = q35.open_runtimes({a.layer: w}, a.pci)
    q35._State.runtimes = runtimes
    print(f"two runtimes open in {time.time() - t0:.1f} s", flush=True)
    try:
        rows = min(a.rows, q35._SIZES[-1])
        x = (torch.randn(rows, 2 * npu.K, device="cuda") * 0.5).to(torch.bfloat16)
        # `_assemble` takes the projections separately: qkvz, then ba.
        ws = (w[: w.shape[0] - ba_cols], w[w.shape[0] - ba_cols:])
        seqs = q35._kick(x, a.layer, 0)
        full = torch.cat(q35._assemble(ws, x, rows, seqs, 0), dim=1).clone()
        torch.cuda.synchronize()
        for rt in runtimes:
            rt.check()
        n = 64
        ex = emulate_halves(x[:n], w).cuda()
        rel = ((full[:n].float() - ex.float()).norm() / ex.float().norm()).item()
        wg = w.cuda()
        tg = x[:n] @ wg.T
        rel_bf16 = ((full[:n].float() - tg.float()).norm() / tg.float().norm()).item()
        print(f"vs the contract of two halves, bf16 add ({n} rows): rel err {rel:.4%}", flush=True)
        print(f"vs the bf16 GPU product ({n} rows): rel err {rel_bf16:.4%}", flush=True)

        def run():
            s = q35._kick(x, a.layer, 0)
            q35._assemble(ws, x, rows, s, 0)
        for _ in range(2):
            run()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            run()
        torch.cuda.synchronize()
        npu_ms = (time.perf_counter() - t0) / a.reps * 1e3
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(a.reps):
            x @ wg.T
        e.record()
        torch.cuda.synchronize()
        gpu_ms = s.elapsed_time(e) / a.reps
        fl = 2 * rows * n_live * 2 * npu.K
        print(f"{rows} rows: NPU {npu_ms:.1f} ms ({fl / npu_ms / 1e9:.1f} TOPS), GPU {gpu_ms:.1f} ms", flush=True)
    finally:
        for rt in runtimes:
            rt.close()


if __name__ == "__main__":
    main()
