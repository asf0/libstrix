# SPDX-License-Identifier: MIT
"""NPU input projections (strix_vllm.npu) against references, on real weights.

1. pack_a: the Triton A packer, byte for byte against a NumPy port of hipfire's
   scalar producer (npu_trunk_pack_a_bf16_old) on 512 rows (two 256-row waves,
   one partial).
2. One layer's projection on the NPU (GDN in_proj_qkvz + in_proj_ba, or QSA qkv)
   against an exact CPU emulation of the W8A8 direct-BF16 contract and against the
   bf16 GPU product; then the NPU time per chain and what the GPU takes.

    python bench/kernels/npu_proj.py --model /path/to/Qwen3.8-Flash-Next-AWQ-W4A16 \
        [--layer 0] [--rows 8192] [--pci 0000:c6:00.0]

Needs /dev/accel/accel0, a memlock limit of a few GB, and libstrix_npu.so
(npu/, `cargo build --release`).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from strix_vllm import npu

K = npu.K


def ref_pack_a(x: np.ndarray, padded: int) -> np.ndarray:
    """hipfire's scalar producer, in float32 like the device."""
    rows = x.shape[0]
    kc, waves = K // 64, padded // 256
    out = np.zeros(padded * 2880, dtype=np.uint8)
    xs = x.astype(np.float32)
    am = np.abs(xs).max(axis=1)
    raw = (am / np.float32(127.0)).astype(np.float32)
    f, ex = np.frexp(raw)
    e = ex - 15
    m = np.ceil((f * np.float32(32768.0)).astype(np.float32)).astype(np.int64)
    carry = m == 32768
    m[carry], e[carry] = 16384, e[carry] + 1
    m[am == 0], e[am == 0] = 0, 0
    scale = (m.astype(np.float32) * np.exp2(e.astype(np.float32))).astype(np.float32)
    with np.errstate(divide="ignore"):
        inv = np.where(am != 0, (np.float32(1.0) / scale).astype(np.float32), np.float32(0))
    q = np.clip(np.rint((xs * inv[:, None]).astype(np.float32)), -127, 127).astype(np.int8)
    i = np.arange(K)
    ch, kb = i // 64, (i % 64) // 8
    for t in range(padded):
        wave, rt, pair = t // 256, t % 128, (t % 256) // 128
        col = pair * 2 + (rt // 8) % 2
        if t < rows:
            off = (col * waves * kc + wave * kc + ch) * 4608 + (rt // 16) * 512 + kb * 64 + (rt % 8) * 8 + i % 8
            out[off] = q[t].view(np.uint8)
        mm, ee = (int(m[t]), int(e[t])) if t < rows else (0, 0)
        for c in range(kc):
            for h in range(2):
                base = ((pair * 2 + h) * waves * kc + wave * kc + c) * 4608 + 4096
                out[base + 2 * rt: base + 2 * rt + 2] = np.frombuffer(np.uint16(mm).tobytes(), np.uint8)
                out[base + 256 + 2 * rt: base + 258 + 2 * rt] = np.frombuffer(np.int16(ee).tobytes(), np.uint8)
    return out


def grid15_up(v: np.ndarray) -> np.ndarray:
    """Per-row scale on the grid15 (rounded up), as float32 (numpy port of grid15_up)."""
    f, ex = np.frexp(v.astype(np.float32))
    m = np.ceil((f * np.float32(32768.0)).astype(np.float32))
    e = ex - 15
    carry = m == 32768
    m[carry], e[carry] = 16384, e[carry] + 1
    s = (m * np.exp2(e.astype(np.float64))).astype(np.float32)
    s[v == 0] = 0
    return s


def emulate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Exact W8A8 direct-BF16: int8 codes with grid15-up scales, int32 dot, one rounding."""
    def q(a, multiply):
        # Activations: x * (1 / s) like the GPU producer; weights: w / s like the packer.
        a = a.float().cpu().numpy()
        am = np.abs(a).max(axis=1)
        s = grid15_up((am.astype(np.float64) / 127.0).astype(np.float32))
        with np.errstate(divide="ignore", invalid="ignore"):
            if multiply:
                inv = np.where(s > 0, (np.float32(1) / s).astype(np.float32), np.float32(0))
                c = np.rint((a * inv[:, None]).astype(np.float32))
            else:
                c = np.where(s[:, None] > 0, np.rint((a / s[:, None]).astype(np.float32)), 0)
        return torch.from_numpy(np.clip(c, -127, 127).astype(np.float64)), torch.from_numpy(s.astype(np.float64))
    xq, xs = q(x, True)
    wq, ws = q(w, False)
    acc = xq @ wq.T  # exact in float64 (|acc| < 2^53)
    return (acc * xs[:, None] * ws[None, :]).to(torch.bfloat16)


def load_weight(model: Path, layer: int, shape: int) -> torch.Tensor:
    idx = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.language_model.layers.{layer}."
    names = (
        ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"]
        if shape == npu.GDN_IN
        else ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    )
    parts = []
    for n in names:
        key = pre + n + ".weight"
        with safe_open(model / idx[key], "pt") as f:
            parts.append(f.get_tensor(key))
    return torch.cat(parts).to(torch.bfloat16)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", type=int, default=8192)
    p.add_argument("--rows-max", type=int, help="runtime capacity (default --rows): a smaller --rows runs a row-adaptive program")
    p.add_argument("--pci", default="0000:c6:00.0")
    p.add_argument("--reps", type=int, default=10)
    a = p.parse_args()
    torch.manual_seed(0)

    # 1. pack_a bytes.
    x = torch.randn(300, K, device="cuda", dtype=torch.bfloat16) * 3
    x[5] = 0
    arena = torch.zeros(512 * 2880, dtype=torch.uint8, device="cuda")
    npu.pack_a(x, arena, 512)
    got = arena.cpu().numpy()
    want = ref_pack_a(x.cpu().float().numpy(), 512)
    bad = int((got != want).sum())
    print(f"pack_a: {bad} of {want.size} bytes differ", flush=True)

    # 2. One layer on the NPU.
    model = Path(a.model)
    layer_type = json.loads((model / "config.json").read_text())
    lt = layer_type.get("text_config", layer_type)["layer_types"][a.layer]
    shape = npu.GDN_IN if lt == "linear_attention" else npu.QSA_IN
    w = load_weight(model, a.layer, shape)
    t0 = time.time()
    rt = npu.NpuRuntime.open(a.rows_max or a.rows, [(a.layer, shape, w)], a.pci)
    print(f"layer {a.layer} {lt}: weight {tuple(w.shape)}, open {time.time() - t0:.1f} s", flush=True)
    try:
        x = (torch.randn(a.rows, K, device="cuda") * 0.5).to(torch.bfloat16)
        seq = rt.kick(a.layer, shape, x)
        c = rt.result(shape, a.rows, seq)[:, : w.shape[0]].clone()
        rt.retire(seq)
        torch.cuda.synchronize()
        rt.check()
        sub = slice(0, 64)
        emu = emulate(x[sub], w)
        exact = (c[sub].cpu() == emu).float().mean().item()
        wg = w.cuda()
        ref = x @ wg.T
        rel = ((c.float() - ref.float()).norm() / ref.float().norm()).item()
        print(f"NPU vs exact W8A8 emulation (64 rows): {exact * 100:.3f} % bit-equal", flush=True)
        print(f"NPU vs bf16 GPU product: relative error {rel:.4%}", flush=True)

        # Timing: NPU chain (kick to result on the stream) and the GPU GEMM.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.reps):
            seq = rt.kick(a.layer, shape, x)
            rt.result(shape, a.rows, seq)
            rt.retire(seq)
        torch.cuda.synchronize()
        npu_ms = (time.perf_counter() - t0) / a.reps * 1e3
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(a.reps):
            x @ wg.T
        e.record()
        torch.cuda.synchronize()
        gpu_ms = s.elapsed_time(e) / a.reps
        flop = 2 * a.rows * K * w.shape[0]
        print(f"{a.rows} rows: NPU {npu_ms:.1f} ms ({flop / npu_ms / 1e9:.1f} TOPS), GPU {gpu_ms:.1f} ms", flush=True)
    finally:
        rt.close()


if __name__ == "__main__":
    main()
