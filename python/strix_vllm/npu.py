# SPDX-License-Identifier: MIT
"""XDNA2 NPU input projections for Qwen3.8-Flash-Next prefill.

The NPU runs W8A8 GEMMs with a BF16 epilogue (hipfire's G80 program and trunk
transport, through the `strix-npu` shim in npu/): the GDN in_proj_qkvz +
in_proj_ba (16,480 columns) and the QSA qkv_proj (13,312) of a block of prompt
rows, while the GPU runs the rest of the layer for the other rows.

    rt = NpuRuntime.open(rows_max, layers, pci)   # layers: (index, shape, weight)
    seq = rt.kick(layer, shape, x)                # packs x on the current stream
    out = rt.result(shape, rows, seq)             # stream waits; view of C

Numerics follow hipfire's direct-BF16 contract: weights int8 per output row and
activations int8 per token, both with grid15-up scales, int32 accumulation and
one rounding to BF16.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

GDN_IN, QSA_IN = 0, 1
K = 2560
_WAVE = 256
_KC = K // 64
_CHUNK = 4608  # 4096 code bytes + 128 u16 mantissas + 128 i16 exponents

_LIB: ctypes.CDLL | None = None


def _lib_path() -> str:
    env = os.environ.get("STRIX_NPU_LIB")
    if env:
        return env
    for cand in (
        Path("/opt/strix/lib/libstrix_npu.so"),
        Path(__file__).resolve().parents[2] / "build/npu-target/release/libstrix_npu.so",
    ):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("libstrix_npu.so not found (set STRIX_NPU_LIB)")


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        L = ctypes.CDLL(_lib_path())
        vp, sz, i, u32, u64 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64
        sig = {
            "strix_npu_last_error": ([], ctypes.c_char_p),
            "strix_npu_builder_new": ([sz], vp),
            "strix_npu_builder_add_layer": ([vp, sz, i, vp, sz], i),
            "strix_npu_builder_free": ([vp], None),
            "strix_npu_open": ([vp, ctypes.c_char_p, sz, u64], vp),
            "strix_npu_arena": ([vp, i, sz, i, ctypes.POINTER(u64), ctypes.POINTER(vp), ctypes.POINTER(sz)], i),
            "strix_npu_chain_rows": ([vp, i, sz], sz),
            "strix_npu_pitch": ([i], sz),
            "strix_npu_prepare_bank": ([vp, sz, vp], i),
            "strix_npu_kick": ([vp, sz, i, sz, sz, vp, ctypes.POINTER(u32)], i),
            "strix_npu_wait_gpu": ([vp, sz, u32, vp], i),
            "strix_npu_retire": ([vp, sz, u32, vp], i),
            "strix_npu_wait_host": ([vp, sz, u32, ctypes.POINTER(u64)], i),
            "strix_npu_check_error": ([vp], i),
            "strix_npu_close": ([vp], i),
            "strix_npu_take_stats": ([vp, ctypes.POINTER(u64)], i),
        }
        for name, (args, res) in sig.items():
            f = getattr(L, name)
            f.argtypes, f.restype = args, res
        _LIB = L
    return _LIB


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"strix-npu {what}: {lib().strix_npu_last_error().decode()}")


class _DevView:
    """__cuda_array_interface__ over a raw device address (host-registered arena)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3,
        }


def _view(ptr: int, nbytes: int) -> torch.Tensor:
    return torch.as_tensor(_DevView(ptr, nbytes), device="cuda")


@triton.jit
def _pack_a_kernel(
    x_ptr, a_ptr, rows, stride_x, waves,
    K: tl.constexpr, KC: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
):
    """One token row: hipfire's grid15-up per-token scale, RNE int8 codes, G80 tiled A
    layout (npu_trunk_pack_a_bf16_old), and the scale sidecars of every K64 chunk.
    Rows past `rows` (up to the padded rows) are written as zeros."""
    t = tl.program_id(0)
    k = tl.arange(0, BK)
    kin = k < K
    live = t < rows
    x = tl.load(x_ptr + t * stride_x + k, mask=live & kin, other=0.0).to(tl.float32)
    am = tl.max(tl.abs(x), axis=0)
    raw = tl.math.div_rn(am, 127.0)
    # frexp(raw) for a normal float: raw = f * 2^ex, f in [0.5, 1).
    bits = raw.to(tl.int32, bitcast=True)
    ex = ((bits >> 23) & 0xFF) - 126
    f = ((bits & 0x007FFFFF) | (126 << 23)).to(tl.float32, bitcast=True)  # raw >= 0
    e = ex - 15
    m = tl.math.ceil(f * 32768.0).to(tl.int32)
    carry = m == 32768
    m = tl.where(carry, 16384, m)
    e = tl.where(carry, e + 1, e)
    nz = am != 0.0
    m = tl.where(nz, m, 0)
    e = tl.where(nz, e, 0)
    scale = m.to(tl.float32) * tl.math.exp2(e.to(tl.float32))
    inv = tl.where(nz, tl.math.div_rn(1.0, tl.where(nz, scale, 1.0)), 0.0)
    v = x * inv
    # Round half to even, exact for |v| < 2^22.
    big = 12582912.0
    q = ((v + big) - big).to(tl.int32)
    q = tl.minimum(tl.maximum(q, -127), 127)
    wave = t // 256
    rt = t % 128
    pair = (t % 256) // 128
    col = pair * 2 + (rt // 8) % 2
    ch = k // 64
    kb = (k % 64) // 8
    off = (col * waves * KC + wave * KC + ch) * 4608 + (rt // 16) * 512 + kb * 64 + (rt % 8) * 8 + k % 8
    tl.store(a_ptr + off, q.to(tl.int8), mask=kin)
    # Scales: every K64 chunk of both halves of the pair repeats the token's (m, e).
    c = tl.arange(0, BC)
    cin = c < KC
    for h in tl.static_range(2):
        base = ((pair * 2 + h) * waves * KC + wave * KC + c) * 4608 + 4096
        mp = (a_ptr + base + 2 * rt).to(tl.pointer_type(tl.int16))
        ep = (a_ptr + base + 256 + 2 * rt).to(tl.pointer_type(tl.int16))
        tl.store(mp, tl.full([BC], 0, tl.int32).to(tl.int16) + m.to(tl.int16), mask=cin)
        tl.store(ep, tl.full([BC], 0, tl.int32).to(tl.int16) + e.to(tl.int16), mask=cin)


def pack_a(x: torch.Tensor, arena: torch.Tensor, padded_rows: int) -> None:
    """x: [rows, 2560] bf16 (row stride may exceed K); arena: uint8 view of the A arena."""
    rows = x.shape[0]
    assert x.shape[1] == K and x.stride(1) == 1 and padded_rows % _WAVE == 0
    assert arena.numel() >= padded_rows // _WAVE * 4 * _KC * _CHUNK, "A arena too small"
    _pack_a_kernel[(padded_rows,)](
        x, arena, rows, x.stride(0), padded_rows // _WAVE,
        K=K, KC=_KC, BK=4096, BC=64, num_warps=8,
    )


class NpuRuntime:
    """One NPU transport with every layer's projection panels resident."""

    def __init__(self, handle: int, rows_max: int, banks: int):
        self.h = handle
        self.rows_max = rows_max
        self.banks = banks
        self.a = []
        self.c: list[dict[int, tuple[int, int]]] = []
        L = lib()
        for b in range(banks):
            dev, host, n = ctypes.c_uint64(), ctypes.c_void_p(), ctypes.c_size_t()
            _check(L.strix_npu_arena(self.h, 0, b, 0, ctypes.byref(dev), ctypes.byref(host), ctypes.byref(n)), "A arena")
            self.a.append(_view(dev.value, n.value))
            cs = {}
            for shape in (GDN_IN, QSA_IN):
                if L.strix_npu_arena(self.h, 1, b, shape, ctypes.byref(dev), ctypes.byref(host), ctypes.byref(n)) == 0:
                    cs[shape] = (dev.value, n.value)
            self.c.append(cs)
        self.pitch = {s: L.strix_npu_pitch(s) for s in (GDN_IN, QSA_IN)}

    @classmethod
    def open(cls, rows_max: int, layers, pci: str, banks: int = 1) -> NpuRuntime:
        """layers: iterable of (layer_index, shape, weight [n_live, 2560] bf16 tensor)."""
        L = lib()
        b = L.strix_npu_builder_new(rows_max)
        if not b:
            _check(-1, "builder")
        try:
            for idx, shape, w in layers:
                w = w.detach().to("cpu", torch.bfloat16).contiguous()
                assert w.shape[1] == K, w.shape
                _check(L.strix_npu_builder_add_layer(b, idx, shape, w.data_ptr(), w.shape[0]), f"layer {idx}")
        except BaseException:
            L.strix_npu_builder_free(b)
            raise
        # railgun bounds a chain at 300 ms; an NPU waking from idle has taken 177 ms.
        watchdog = int(os.environ.get("STRIX_NPU_WATCHDOG_MS", "1000"))
        h = L.strix_npu_open(b, pci.encode(), banks, watchdog)
        if not h:
            _check(-1, "open")
        return cls(h, rows_max, banks)

    def chain_rows(self, shape: int, rows: int) -> int:
        return lib().strix_npu_chain_rows(self.h, shape, rows)

    def kick(self, layer: int, shape: int, x: torch.Tensor, bank: int = 0) -> int:
        """Packs x ([rows, 2560] bf16 on the current stream) and queues the layer's chain."""
        L = lib()
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        rows = x.shape[0]
        padded = self.chain_rows(shape, rows)
        _check(L.strix_npu_prepare_bank(self.h, bank, stream), "prepare")
        pack_a(x, self.a[bank], padded)
        seq = ctypes.c_uint32()
        _check(L.strix_npu_kick(self.h, layer, shape, bank, rows, stream, ctypes.byref(seq)), "kick")
        return seq.value

    def result(self, shape: int, rows: int, seq: int, bank: int = 0) -> torch.Tensor:
        """Enqueues the wait on the current stream and returns the C arena as a
        [rows, pitch] bf16 view. Valid until `retire`."""
        L = lib()
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(L.strix_npu_wait_gpu(self.h, bank, seq, stream), "wait")
        dev, n = self.c[bank][shape]
        pitch = self.pitch[shape]
        c = _view(dev, n).view(torch.bfloat16).view(-1, pitch)
        return c[:rows]

    def retire(self, seq: int, bank: int = 0) -> None:
        """After the last consumer of the bank's C is enqueued on the current stream."""
        L = lib()
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(L.strix_npu_retire(self.h, bank, seq, stream), "retire")

    def busy_ms(self, seq: int, bank: int = 0) -> float:
        """Blocks until (bank, seq) completes; its NPU time (diagnostics only)."""
        ns = ctypes.c_uint64()
        _check(lib().strix_npu_wait_host(self.h, bank, seq, ctypes.byref(ns)), "wait_host")
        return ns.value / 1e6

    def take_stats(self) -> list[int]:
        """[chains, busy_ns, kick_to_done_ns] of GdnIn then QsaIn since the last call."""
        out = (ctypes.c_uint64 * 6)()
        _check(lib().strix_npu_take_stats(self.h, out), "stats")
        return list(out)

    def check(self) -> None:
        _check(lib().strix_npu_check_error(self.h), "chain")

    def close(self) -> None:
        if self.h:
            _check(lib().strix_npu_close(self.h), "close")
            self.h = 0
