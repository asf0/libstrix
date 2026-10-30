# SPDX-License-Identifier: MIT
"""Kernel table: every entry point, the op it implements and its geometry.

A caller asks `select(op, **geometry)` instead of hard-coding which shapes a
kernel takes. `None` means no kernel here covers the shape, so the caller keeps
its default path. `geometries(op, *keys)` returns the shape points a family
covers, so a plugin module can build its dispatch gate from this table rather
than repeating the list (and drifting from it).

Per entry:

  constraints  what a caller must pass to match. A scalar is one required
               value, a tuple is the set of allowed values, `Range` is a
               bound. Order matters: the first matching row of an op wins.
  spec         informational only (tile shape, staging, why it is what it is);
               never matched against.

A scalar constraint is also a *geometry point*: `geometries("attn_prefill",
"head_dim", "gqa")` collects the (head_dim, gqa) pairs of every row, i.e. the
shapes the prefill kernel accepts.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

_DTYPE_NAMES = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.int8: "int8",
}

# Values a geometry key can hold: an entry with one of these as its constraint
# is a concrete shape point, not a set of acceptable values.
_POINTS = (int, bool, str)


def _norm(v: Any) -> Any:
    if isinstance(v, torch.dtype):
        return _DTYPE_NAMES.get(v, str(v))
    if isinstance(v, str) and v.startswith("torch."):
        return _DTYPE_NAMES.get(getattr(torch, v[6:], None), v)
    return v


def _ok(want: Any, got: Any) -> bool:
    if isinstance(want, Range):
        return want.covers(got)
    if isinstance(want, tuple):
        return got in want
    return got == want


def _need(want: Any) -> str:
    if isinstance(want, Range):
        return str(want)
    if isinstance(want, tuple):
        return f"in {want}"
    return f"== {want}"


@dataclass(frozen=True)
class Range:
    """Numeric bound on a geometry value: `Range(lo=6)` means 6 or more."""

    lo: int | None = None
    hi: int | None = None

    def covers(self, got: Any) -> bool:
        if not isinstance(got, int) or isinstance(got, bool):
            return False
        if self.lo is not None and got < self.lo:
            return False
        return self.hi is None or got <= self.hi

    def __str__(self) -> str:
        if self.lo is not None and self.hi is not None:
            return f"in {self.lo}..{self.hi}"
        if self.lo is not None:
            return f">= {self.lo}"
        return f"<= {self.hi}"


@dataclass(frozen=True)
class Kernel:
    name: str
    op: str
    doc: str
    # key -> required value, tuple of allowed values, or Range
    constraints: dict[str, Any] = field(default_factory=dict)
    # informational: tile shape, staging, measurement notes
    spec: dict[str, Any] = field(default_factory=dict)

    def check(self, geometry: dict[str, Any]) -> str | None:
        for key, want in self.constraints.items():
            if key not in geometry:
                return f"{key} not given"
            got = _norm(geometry[key])
            if not _ok(want, got):
                return f"{key}={got}, needs {_need(want)}"
        return None

    def accepts(self, geometry: dict[str, Any]) -> bool:
        """Whether the given (partial) geometry falls inside this row."""
        return all(
            key in self.constraints and _ok(self.constraints[key], _norm(value))
            for key, value in geometry.items()
        )


_BF16 = {"dtype": "bf16"}

KERNELS: list[Kernel] = [
    # ---------------------------------------------------------------- norm
    Kernel(
        "gdn_gated_rmsnorm_h128_bf16",
        "rms_norm_gated",
        "out = rms(x) * w * act(z) over 128-channel rows, norm before gate",
        {
            "hidden": 128,
            "dtype": "bf16",
            "weight_dtype": ("bf16", "fp32"),
            "group_size": (None, 128),
            "norm_before_gate": True,
            "activation": ("silu", "swish", "sigmoid"),
        },
    ),
    # ----------------------------------------------------------------- moe
    Kernel(
        "moe_gemm_bf16",
        "moe_gemm",
        "expert w13 GEMM with the gate nonlinearity and mul fused into the "
        "epilogue: writes [slots, n / 2] instead of [slots, n]. SiLU is the "
        "Qwen/LFM path; GELU-tanh is here because gemma-4's MoE sets "
        "hidden_activation gelu_pytorch_tanh, so on that model a SiLU-only "
        "epilogue would leave every expert on vLLM's Triton kernel. "
        "n % 128 == 0, k % 64 == 0",
        {
            "silu_mul": True,
            **_BF16,
            "weight_dtype": "bf16",
            "quant": "none",
            "activation": ("silu", "gelu_tanh"),
        },
        {"block_m": (16, 32, 64, 128), "block_n": (64, 128, 256)},
    ),
    Kernel(
        "moe_gemm_bf16",
        "moe_gemm",
        "expert w2 GEMM, unweighted or with topk weights applied on the "
        "output. Takes the same activation set as the w13 row: its own input "
        "is already activated, but a gemma-4 config must select this row too. "
        "n % 128 == 0, k % 64 == 0",
        {
            "silu_mul": False,
            **_BF16,
            "weight_dtype": "bf16",
            "quant": "none",
            "activation": ("silu", "gelu_tanh"),
        },
        {"block_m": (16, 32, 64, 128), "block_n": (64, 128, 256)},
    ),
    Kernel(
        "moe_gemm_w4a16",
        "moe_gemm_w4a16",
        "expert w13 GEMM over vLLM's Triton WNA16 layout: uint8 [E, N, K/2], "
        "byte j holds k=2j (low nibble) and k=2j+1 (high), bf16 scales "
        "[E, N, K/group], weights dequantized ((nibble - 8) * scale) into LDS. "
        "Same fused gate-and-mul epilogue as moe_gemm_bf16. n % 128 == 0, "
        "k % 64 == 0",
        {
            "silu_mul": True,
            **_BF16,
            "weight_dtype": "uint8",
            "weight_bits": 4,
            "quant": "wna16",
            "group_size": (32, 64, 128),
            "activation": ("silu", "gelu_tanh"),
        },
        {"block_m": (16, 32, 64, 128), "block_n": (64, 128, 256)},
    ),
    Kernel(
        "moe_gemm_w8a16",
        "moe_gemm_w8a16",
        "expert w13 GEMM over compressed-tensors int8 experts: int8 "
        "[E, N, K], one scale per output channel ([E, N] or [E, N, 1]), "
        "weights dequantized (int8 * scale) into LDS and bf16 activations "
        "(W8A16). Same fused gate-and-mul epilogue as moe_gemm_bf16. "
        "n % 128 == 0, k % 64 == 0",
        {
            "silu_mul": True,
            **_BF16,
            "weight_dtype": "int8",
            "weight_bits": 8,
            "quant": "int8",
            "group_size": (0,),
            "activation": ("silu", "gelu_tanh"),
        },
        {"block_m": (16, 32, 64, 128), "block_n": (64, 128, 256)},
    ),
    Kernel(
        "moe_gemm_w8a16",
        "moe_gemm_w8a16",
        "expert w2 GEMM over the same int8 expert layout, unweighted or with "
        "the topk weights applied on the output. n % 128 == 0, k % 64 == 0",
        {
            "silu_mul": False,
            **_BF16,
            "weight_dtype": "int8",
            "weight_bits": 8,
            "quant": "int8",
            "group_size": (0,),
            "activation": ("silu", "gelu_tanh"),
        },
        {"block_m": (16, 32, 64, 128), "block_n": (64, 128, 256)},
    ),
    Kernel(
        "moe_gemm_w4a16",
        "moe_gemm_w4a16",
        "expert w2 GEMM over the same WNA16 layout, unweighted or with the "
        "topk weights applied on the output. n % 128 == 0, k % 64 == 0",
        {
            "silu_mul": False,
            **_BF16,
            "weight_dtype": "uint8",
            "weight_bits": 4,
            "quant": "wna16",
            "group_size": (32, 64, 128),
            "activation": ("silu", "gelu_tanh"),
        },
        {"block_m": (16, 32, 64, 128), "block_n": (64, 128, 256)},
    ),
    Kernel(
        "moe_pad",
        "moe_pad",
        "(weight layout, not a kernel) expert rows copied to a padded pitch "
        "in process_weights_after_loading: 2048/512-element rows sit 4 KiB / "
        "1 KiB apart and camp on the same DRAM channels",
        {**_BF16, "quant": "none"},
        {"pad": 64, "when": "after load; vLLM's #46009 copies the view back"},
    ),
    # ----------------------------------------------------------------- gdn
    Kernel(
        "gdn_chunk_fwd_k128_v128_c64_bf16",
        "gdn_chunk",
        "chunked gated delta rule (chunk prefill scan), 64-token chunks",
        {
            "head_k": 128,
            "head_v": 128,
            "chunk": 64,
            **_BF16,
            "l2norm_in_kernel": False,
            "state_dtype": "fp32",
        },
    ),
    Kernel(
        "gdn_causal_conv1d_w4_bf16",
        "gdn_conv",
        "width-4 causal depthwise conv over varlen rows, with the conv-state "
        "ring update and optional SiLU. Rows bf16, stride(0) % 8 == 0, 16 B "
        "aligned",
        {
            **_BF16,
            "width": 4,
            "activation": ("silu", "none"),
            "pad_slot_id": -1,
            "null_block_id": 0,
        },
    ),
    Kernel(
        "gdn_post_conv_bf16",
        "gdn_post_conv",
        "splits conv output into L2-normalized q, k and v, and computes "
        "g = -exp(a_log) softplus(a + dt_bias), beta = sigmoid(b)",
        {"head_k": 128, "head_v": 128, **_BF16, "l2norm": True, "g_exp": False},
    ),
    Kernel(
        "gdn_decode_mtp_bf16",
        "gdn_mtp",
        "fused GDN decode for MTP/speculative batches (vLLM's CUDA-only "
        "kernel, ported): state update, gated norm and output gate in one pass",
        {
            "head_k": 128,
            "head_v": 128,
            **_BF16,
            "output_gate_activation": ("silu",),
        },
        {
            "lds_staging": False,
            "note": "direct global loads; LDS-staged was 11% slower at x16",
        },
    ),
    # ----------------------------------------------------------- attention
    Kernel(
        "attn_prefill_bf16",
        "attn_prefill",
        "causal paged prefill/verify attention; 16-query tiles, HPB heads per "
        "block. int8 KV is vLLM's int8_per_token_head layout: byte caches "
        "plus one fp32 dequant scale per (key, head), widened to bf16 on the "
        "way into LDS and scaled once per score instead of per element. "
        "Split-KV (num_splits > 1) for short queries over long contexts is "
        "built for head 256 GQA 6 only: 1.2x Triton at 16k on 27B, while "
        "GQA 8 is slower than Triton there",
        {
            "head_dim": 256,
            "gqa": 6,
            **_BF16,
            "kv_dtype": ("bf16", "int8"),
            "split_kv": True,
            "window": (True, False),
            "causal": True,
            "sinks": False,
        },
        {"hpb": 6, "bn": 32},
    ),
    Kernel(
        "attn_prefill_bf16",
        "attn_prefill",
        "head 256 GQA 8 (Qwen3.8-27B full layers): 16-key steps, BN 16; "
        "BN 32 needs > 256 VGPRs per 512-thread block and spills",
        {
            "head_dim": 256,
            "gqa": 8,
            **_BF16,
            "kv_dtype": ("bf16", "int8"),
            "split_kv": False,
            "window": (True, False),
            "causal": True,
            "sinks": False,
        },
        {"hpb": 8, "bn": 16},
    ),
    Kernel(
        "attn_prefill_bf16",
        "attn_prefill",
        "head 256 GQA 2 (Gemma-4 sliding layers, 32 q / 16 kv heads)",
        {
            "head_dim": 256,
            "gqa": 2,
            **_BF16,
            "kv_dtype": ("bf16", "int8"),
            "split_kv": False,
            "window": (True, False),
            "causal": True,
            "sinks": False,
        },
        {"hpb": 2, "bn": 16, "xt": "fp16"},
    ),
    Kernel(
        "attn_prefill_bf16",
        "attn_prefill",
        "head 512 GQA 8 (Gemma-4 full layers): four waves per head, all eight "
        "heads of a KV head per block so each K/V tile is staged once; fp16 "
        "score partials fit it in 58 KB at 192 VGPRs (21.7 -> 28.5 TF)",
        {
            "head_dim": 512,
            "gqa": 8,
            **_BF16,
            "kv_dtype": ("bf16", "int8"),
            "split_kv": False,
            "window": (True, False),
            "causal": True,
            "sinks": False,
        },
        {"hpb": 8, "bn": 16, "xt": "fp16"},
    ),
    Kernel(
        "attn_prefill_bf16",
        "attn_prefill",
        "head 64 GQA 4 (LFM2.5): one wave per head, all four heads of a KV "
        "head per block; BN 32 (185 VGPRs) beat 64 by 12% at 8k",
        {
            "head_dim": 64,
            "gqa": 4,
            **_BF16,
            "kv_dtype": ("bf16", "int8"),
            "split_kv": False,
            "window": (True, False),
            "causal": True,
            "sinks": False,
        },
        {"hpb": 4, "bn": 32},
    ),
    Kernel(
        "attn_prefill32_bf16",
        "attn_prefill32",
        "NOT ROUTED: D=512 prefill with 32 query columns per K/V stage. Warps "
        "are 2 column tiles x 4 D-slices for the score phase, P^T and row "
        "stats go through LDS, and each warp owns DV/8 V dims for both tiles "
        "in the PV phase. Was 1.06x production; after production's occupancy "
        "and VALU fixes it is 0.95x (20 vs 21 TF), so Gemma-4 uses attn_prefill",
        {
            "head_dim": 512,
            "gqa": 8,
            **_BF16,
            "kv_dtype": ("bf16",),
            "split_kv": False,
            "window": (False,),
            "causal": True,
            "sinks": False,
        },
        {"hpb": 2, "bn": 16},
    ),
    Kernel(
        "attn_decode_bf16",
        "attn_decode",
        "one query per sequence, split-KV, key-tiled: Gemma-4 sliding layers, "
        "where Triton decode reached about half the KV bandwidth. int8 KV "
        "widens bytes in LDS and folds the per-key scale into score/P",
        {
            "head_dim": 256,
            "gqa": 2,
            "queries": 1,
            **_BF16,
            "kv_dtype": ("bf16", "int8"),
            "window": (True, False),
        },
        {"block": 32},
    ),
    Kernel(
        "attn_decode_bf16",
        "attn_decode",
        "one query per sequence, split-KV, key-tiled: Gemma-4 full layers "
        "(head 512 GQA 8), bf16 or int8 KV",
        {
            "head_dim": 512,
            "gqa": 8,
            "queries": 1,
            **_BF16,
            "kv_dtype": ("bf16", "int8"),
            "window": (True, False),
        },
        {"block": 16},
    ),
    Kernel(
        "attn_decode_bf16",
        "attn_decode",
        "head 256 GQA 8 (Qwen3.6-35B-A3B), bf16 or int8 KV. With the parallel "
        "softmax, 2.0x the vendored Triton decode at x1 from 64k on int8 "
        "(176 vs 85 GB/s at 256k), 1.43x on bf16 at 128k with a large KV "
        "pool; 0.88-0.90x at x4/x16, so attn.py takes it at <= 2 sequences",
        {
            "head_dim": 256,
            "gqa": 8,
            "queries": 1,
            **_BF16,
            "kv_dtype": ("bf16", "int8"),
            "window": (False,),
        },
        {"block": 32},
    ),
    # ----------------------------------------------------- quantized dense
    Kernel(
        "w8a16_skinny_gemm",
        "w8a16",
        "decode GEMM (M <= 5) on the raw uint8b128 weight bytes, no "
        "activation quantize; the weight copy is skipped entirely. K % 16 == 0",
        {
            "weight_bits": 8,
            "group_size": (0, 32, 64, 128),
            "rows": Range(hi=5),
            "dtype": ("bf16", "fp16"),
        },
    ),
    Kernel(
        "w8a8_wmma_gemm",
        "w8a16",
        "prefill GEMM: int8 activations quantized per (row, 128-element "
        "group) inside the op, native iu8 WMMA, fp32 group-scale fold. "
        "K % 128 == 0",
        {
            "weight_bits": 8,
            "group_size": (0, 128),
            "rows": Range(lo=6),
            "dtype": ("bf16", "fp16"),
        },
        {
            "tile": "16/32/64/128 rows by batch x 128 cols, 8 waves: the "
            "smallest tile that still covers M in one m-block, because a "
            "second m-block re-reads the whole weight matrix",
            "lds_kb": "36 / 40 / 48 / 64",
            "raster": "8 x n grouped (a wave shares the weight tile via L2)",
            "note": "operand-traffic bound, so the tile's reuse factor is the "
            "whole story: batch 40 streams 68 GB/s at BM 32 against 172 at "
            "BM 64, and BNF 2 does not help",
        },
    ),
    Kernel(
        "w4a16_wmma_gemm",
        "w4a16",
        "prefill GEMM at 32-element groups and prefill sizes: bf16 dequant of "
        "the int4 weights inside the kernel, bf16 WMMA; 1.16-1.24x the W4A8 "
        "op from M ~512. K % 64 == 0",
        {
            "weight_bits": 4,
            "group_size": 32,
            "rows": Range(lo=129),
            "dtype": "bf16",
            "scale_dtype": "bf16",
        },
    ),
    Kernel(
        "w4a8_wmma_gemm",
        "w4a16",
        "prefill GEMM: int4 weights unpacked into LDS (nibble + 120), int8 "
        "activations, iu8 WMMA; the W4 variant of the W8A8 kernel. "
        "K % 128 == 0",
        {
            "weight_bits": 4,
            "group_size": (0, 32, 128),
            "rows": Range(lo=6),
            "dtype": ("bf16", "fp16"),
        },
    ),
    Kernel(
        "mxfp4_wmma_gemm",
        "mxfp4",
        "MXFP4 W4A16 GEMM: the Quark checkpoint's packed e2m1 nibbles "
        "[N, K/2] with e8m0 bytes [N, K/32] are decoded to bf16 as they are "
        "staged into LDS (the scale folded into the exponent) and the main "
        "loop is a plain bf16 WMMA GEMM, so weights are read in place - no "
        "per-forward dequantized copy and no activation QDQ. K % 128 == 0",
        {
            "weight_bits": 4,
            "group_size": 32,
            "dtype": "bf16",
            "scale_dtype": "e8m0",
        },
        {
            "tile": "16/32/64/128 rows by batch x 128 cols, 4-8 waves: the "
            "smallest tile that still covers M in one m-block, because a "
            "second m-block re-reads the whole weight matrix",
            "lds_kb": "18 / 23 / 28 / 37",
            "note": "decode is a pure weight stream, so the tile is occupancy: "
            "127.5 ms/step on gemma-4-31B at 512 context with 18.1 GiB of "
            "weights, against 271 ms and 59.2 GiB for the emulation's bf16 "
            "copy; a CTA that starts its K loop at a different chunk stops "
            "power-of-two row strides camping on one DRAM channel (22 -> "
            "120 GB/s at K = 8192). M <= 2 dispatches to mxfp4_gemv instead "
            "(177-201 vs 125-135 GB/s at M=1, 103.0 vs 127.5 ms/step on the "
            "model)",
        },
    ),
    Kernel(
        "mxfp4_wmma_gemm_cfg",
        "mxfp4_tile_sweep",
        "bench-only: the MXFP4 GEMM pinned to one tile config instead of the "
        "tile-by-batch choice, for bench/kernels/mxfp4_gemm.py",
        {"weight_bits": 4, "group_size": 32, "dtype": "bf16"},
        {
            "cfgs": "0 (16,128,1,4) | 1 (16,64,1,4) | 2 (32,128,2,4) | "
            "3 (64,128,2,4) | 4 (128,128,4,2) | 5 (128,256,4,4) | "
            "6 (64,256,2,8) | 7 (128,128,2,4) | 8 (16,256,1,4) | "
            "9 (32,64,1,4) | 10 (16,64,1,2) | 11 (16,128,1,2) | "
            "12 (64,64,1,4) | 13 (16,32,1,2) | 14..19 the same with BK 128"
        },
    ),
    Kernel(
        "mxfp4_gemv",
        "mxfp4_gemv",
        "MXFP4 W4A16 GEMV for M <= 8: one warp per output row, weights "
        "decoded in registers straight from the packed stream, A staged in "
        "LDS and shared by eight rows. Prototype for the decode step, where "
        "the WMMA tile stages 16 A rows for one real row",
        {"weight_bits": 4, "group_size": 32, "dtype": "bf16", "m_max": 8},
        {"rows_per_block": 8, "k_chunk": 1024},
    ),
    Kernel(
        "mxfp4_dequant",
        "mxfp4_dequant",
        "MXFP4 [N, K/2] + e8m0 [N, K/32] -> bf16 [N, K], the GEMM's exact "
        "nibble decode, one thread per 32-value block. For prefill M the "
        "plugin decodes once and runs hipBLASLt instead of re-decoding every "
        "weight tile once per 128-row M block",
        {"weight_bits": 4, "group_size": 32, "dtype": "bf16"},
        {"threads": 256},
    ),
    Kernel(
        "w4a16_dequant",
        "w4a16_dequant",
        "W4A16 ExLlama int4 [N, K/2] + bf16 group scales -> bf16 [N, K], the "
        "same decode and rounding as w4a16_wmma_gemm's staging. For prefill M "
        "the plugin decodes once and runs hipBLASLt",
        {"weight_bits": 4, "group_size": (32, 64, 128), "dtype": "bf16"},
        {"threads": 256},
    ),
    Kernel(
        "w8a8_wmma_gemm_cfg",
        "w8a8_tile_sweep",
        "bench-only: the W8A8 GEMM pinned to one (BMF, BNF, WAVES, raster) "
        "config instead of the production choice, for bench/kernels/w8a8_gemm.py. "
        "Same math and the same activation quantize; M must be a multiple of "
        "128 so every config's scale-tile stride lines up",
        {
            "weight_bits": 8,
            "rows": Range(lo=128),
            "dtype": ("bf16", "fp16"),
        },
        {
            "cfgs": "0 (8,1,8) BN128 | 1 (4,2,4) | 2 (4,1,8) | 3 (2,2,4) | "
            "4 (2,1,8) | 5 (8,1,8)+8xn raster | 6 (8,1,8)+4xn | 7 (4,2,4)+8xn"
        },
    ),
    # ---------------------------------------------------- unquantized dense
    Kernel(
        "row_sliced_gemm",
        "dense_gemm",
        "bf16 hipBLASLt GEMM run as 1024-row slices, which keep the "
        "activation panel near the 32 MB infinity cache: 31 -> 41 TFLOPS for "
        "27B MLP down (K = 17408) at 8k rows. Below 1024 rows the original "
        "call is used unchanged",
        {**_BF16, "k_min": Range(lo=16384)},
        {"rows": 1024},
    ),
    # ------------------------------------------------- Qwen3.8-Flash-Next
    Kernel(
        "moe_act_quant_i8",
        "moe_act_quant_i8",
        "per (row, 128-element group) int8 quantization for moe_gemm_w4a8: "
        "s = max|x| / 127 (1 for an all-zero group), q = clamp(rint(x / s), "
        "-127, 127). bf16 in, int8 + fp32 scales out",
        {**_BF16, "out_dtype": "int8", "group_size": (128,)},
        {"threads": 256, "teams": "one 16-lane team per group"},
    ),
    Kernel(
        "moe_gemm_w4a8",
        "moe_gemm_w4a8",
        "expert GEMM over the WNA16 int4 layout with int8 activations: the "
        "iu8 WMMA is ~2x the bf16 FLOP rate, the int4 weights become int8 at "
        "the LDS commit, and the i32 accumulator folds per 128-group with "
        "sa * sw. Takes over from moe_gemm_w4a16 for batch >= 1024 tokens: "
        "1.34x (w13) and 1.19x (w2) at 16k Qwen3.8-Flash-Next tokens",
        {
            "silu_mul": True,
            **_BF16,
            "activation_dtype": "int8",
            "weight_dtype": "uint8",
            "weight_bits": 4,
            "quant": "wna16",
            "group_size": (128,),
            "activation": ("silu", "gelu_tanh"),
        },
        {"block_m": (64, 128), "block_n": (128, 256), "best": "64x256, 2x4 waves"},
    ),
    Kernel(
        "qsa_union_attn_bf16",
        "qsa_union_attn_bf16",
        "Qwen4Exp QSA sparse attention for a single-request prefill: 16 "
        "consecutive query rows gather the union of their selected tokens "
        "once and mask each score by the row's membership bits (S^T = K Q^T, "
        "O^T = V^T P^T, two waves per head). 1.5x the gfx1151 Triton per-row "
        "kernel on real selections, more as the context grows",
        {
            "head_dim": 256,
            "gqa": 12,
            **_BF16,
            "requests": 1,
            "rows_min": 1024,
            "max_position": Range(hi=131072),
        },
        {"rows": (4, 8, 16), "block_n": (16, 32), "partials": "fp16"},
    ),
    Kernel(
        "qsa_kv_to_f16",
        "qsa_kv_to_f16",
        "rewrites one layer's visible K/V from the paged bf16 cache into "
        "dense fp16 rows [kv_heads * 256] for qsa_union_attn_bf16 (bf16 -> "
        "fp16 is exact for cache values); one pass per layer, ~0.5 ms at 16k "
        "tokens. Opt-in via STRIX_QSA_UNION_F16=1",
        {"head_dim": 256, **_BF16, "out_dtype": "fp16"},
        {"threads": 256},
    ),
    Kernel(
        "qsa_indexer_bf16",
        "qsa_indexer_bf16",
        "Qwen4Exp QSA indexer scores over the paged compressed-key cache: "
        "64 x 64 WMMA tiles, the key tile in LDS shared by the four heads, "
        "relu per head, -inf past each row's visible keys. 1.7-1.9x the "
        "tl.dot kernel from 4k to 128k context",
        {
            "indexer_heads": 4,
            "indexer_head_dim": 128,
            **_BF16,
            "requests": 1,
            "rows_min": 128,
        },
        {"block_m": 64, "block_n": 64, "waves": 4},
    ),
    Kernel(
        "hc_up_gate_mix_bf16",
        "hc_up_gate_mix_bf16",
        "Qwen4Exp hyper-connection up projection fused with the gated mix: "
        "y = mean_s sigmoid(bf16(a @ w_s^T)) * x_s over four streams, the "
        "gate never reaching memory (6.7 -> 5.2 ms per site at 16k rows)",
        {"hc_count": 4, **_BF16, "k_min": Range(lo=64), "rows_min": 512},
        {"block_m": 128, "block_n": 32, "waves": 8},
    ),
]

_selections: dict[tuple, dict[str, Any]] = {}


def ops() -> dict[str, list[str]]:
    """op -> names of the kernels implementing it, in priority order."""
    out: dict[str, list[str]] = {}
    for k in KERNELS:
        names = out.setdefault(k.op, [])
        if k.name not in names:
            names.append(k.name)
    return out


def geometries(op: str, *keys: str, **where: Any) -> set[tuple]:
    """The shape points `op` covers, as tuples of `keys` in order.

    `where` narrows the rows first, so `geometries("attn_prefill", "head_dim",
    "gqa", split_kv=True)` is the set of shapes the split-KV variant (the
    speculative-verify path) accepts. Rows whose value for a key is not a
    single value (a set or a Range) are not shape points and are skipped.
    """
    out = set()
    for k in KERNELS:
        if k.op != op or not k.accepts(where):
            continue
        point = []
        for key in keys:
            want = k.constraints.get(key)
            if not isinstance(want, _POINTS):
                break
            point.append(_norm(want))
        else:
            out.add(tuple(point))
    return out


def explain(op: str, **geometry: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": k.name,
            "ok": (r := k.check(geometry)) is None,
            "reason": r,
        }
        for k in KERNELS
        if k.op == op
    ]


def select(op: str, **geometry: Any) -> str | None:
    rows = explain(op, **geometry)
    hit = next((r["name"] for r in rows if r["ok"]), None)
    key = (op, tuple(sorted((k, _norm(v)) for k, v in geometry.items())))
    rec = _selections.setdefault(
        key,
        {
            "op": op,
            "query": dict(key[1]),
            "kernel": hit,
            "reason": None
            if hit
            else "; ".join(r["reason"] for r in rows) or "no kernel",
            "count": 0,
        },
    )
    rec["count"] += 1
    return hit


def selections() -> list[dict[str, Any]]:
    return list(_selections.values())


def kernels() -> list[dict[str, Any]]:
    return [
        {
            "name": k.name,
            "op": k.op,
            "doc": k.doc,
            "constraints": dict(k.constraints),
            "spec": dict(k.spec),
        }
        for k in KERNELS
    ]
