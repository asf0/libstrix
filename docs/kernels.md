# Kernels and the vLLM plugin

## Turning kernels on and off

Installing the package registers the `vllm.general_plugins` entry point
`strix`. It activates only on ROCm + gfx1151 and logs a single line when it
does. `STRIX_DISABLE=all`, or a comma-separated list of the names below, keeps
vLLM's own kernels for A/B runs.

`dense_w8` and `moe_w4a8` change numerics or memory and were measured on
Qwen3.8-Flash-Next only, so they turn on for that architecture
(`Qwen4Exp*`) alone and log `libstrix: <op> on/off for <arch>` at load.
`STRIX_ENABLE=dense_w8,moe_w4a8` turns them on for another model;
`STRIX_DISABLE` still wins. `dense_w8` stops adding int8 copies at
`STRIX_DENSE_W8_MAX_GB` (default 12).

`STRIX_ENABLE=npu` (off by default) moves Qwen3.8-Flash-Next's prefill input
projections to the XDNA2 NPU (`python/strix_vllm/qwen4_exp_npu.py`, the `npu/`
shim over hipfire's G80 BF16 trunk transport). A single-request prefill step of
N >= `STRIX_NPU_MIN_TOKENS` (4,096) tokens is split into two banks, the first
~N/2 tokens and the rest, each with its own attention / GDN / QSA metadata
(the model state's `prepare_attn` on sliced input batches; bank B sees bank A
as context). Layers 0 and 1 (PLE, a convolution across tokens) run on all
rows; from layer 2 on each layer runs bank A, then bank B, while the NPU
computes the other bank's input projection (GDN `in_proj_qkvz` + `in_proj_ba`,
QSA `qkv_proj`), so the GPU runs none from layer 3 on. A bank's rows past the
largest NPU program size (`STRIX_NPU_SIZES`, up to 8,192) are projected on the
GPU. The NPU part is W8A8 (int8 per output row and per token, grid15 scales,
one BF16 rounding), bit-equal to an exact CPU model of that contract. The step
runs eagerly (outside torch.compile); other steps keep the compiled path.
Docker: [docker.md](docker.md#npu-prefill).

| `STRIX_DISABLE` name | kernel | replaces | when |
| --- | --- | --- | --- |
| `rms_norm_gated` | `gdn_gated_rmsnorm_h128_bf16` | `RMSNormGated` (GDN output norm) | bf16, 128 channels |
| `moe_pad` | (weight layout) | MoE expert weight storage | padded row pitch at `create_weights`; vLLM's own ROCm padding is undone by #46009 |
| `moe_gemm` | `moe_gemm_bf16` | both expert GEMMs in `TritonExperts`, gate-and-mul fused into w13 | bf16, no quant/LoRA/bias/EP, SiLU or GELU-tanh |
| `moe_w4a16` | `moe_gemm_w4a16` | both expert GEMMs in `TritonWNA16Experts` (compressed-tensors int4), gate-and-mul fused into w13 | uint4, group 32/64/128, bf16 scales, no zero-point/bias/LoRA/EP; TRITON backend only |
| `moe_w8a16` | `moe_gemm_w8a16` | both expert GEMMs in the int8 MoE path (`oracle/int8.py`, `TritonExperts`), gate-and-mul fused into w13. **Opt-in (`STRIX_MOE_W8A16=1`): measured slower than Triton's int8 WMMA on Qwen3.6-35B-A3B-INT8-W8A8** | int8 [E, N, K] experts, one scale per output channel (fp32/bf16/fp16), bf16 activations (W8A16), no bias/LoRA/EP, K % 64 == 0, N % 128 == 0; TRITON backend only |
| `moe_fp8` | `moe_gemm_fp8_raw` (`moe_gemm_fp8` with `STRIX_DISABLE=moe_fp8_raw`) | both expert GEMMs in vLLM's Triton FP8 MoE (`oracle/fp8.py`, `TritonExperts`), gate-and-mul fused into w13; the experts class defers input quantization, so activations stay unrounded (W8A16) instead of being quantized to FP8. The raw kernel widens the e4m3 bytes to fp16 with integer ops, runs the fp16 WMMA and applies the scale to the fp32 accumulator | float8_e4m3fn [E, N, K] with fp32 128x128 block scales [E, N/128, K/128] (K % 128 == 0) or one scale per output channel (K % 64 == 0), no bias/LoRA/EP, N % 128 == 0; TRITON backend only. Expert rows are padded by 16 bytes as the weights are loaded: a K = 2048 pitch streamed w13 at 38 GB/s, 4.5x slower. Serving scripts that export `VLLM_ROCM_USE_AITER=1` also need `VLLM_ROCM_USE_AITER_MOE=0` |
| `gdn_chunk` | `gdn_chunk_fwd_k128_v128_c64_bf16` | `ChunkGatedDeltaRule` (FLA chunk prefill) | head 128, bf16, no in-kernel l2norm |
| `gdn_conv` | `gdn_causal_conv1d_w4_bf16`, `gdn_post_conv_bf16` | `causal_conv1d_fn` and `fused_post_conv_prep` in the GDN prefill path (module globals rebound) | width 4, bf16 x/state, heads 128, L2 norm on, plain g |
| `gemm_rows` | (hipBLASLt, sliced) | `rocm_unquantized_gemm` for bf16 linears with K >= 16384 | runs > 1024 rows as 1024-row slices (31 -> 41 TFLOPS on 27B MLP down at 8k rows) |
| `attn_prefill` | `attn_prefill_bf16` | `TRITON_ATTN` impl for prefill and speculative verify | head 64 GQA 4 (LFM2.5), head 128 GQA 8 (Flint, with attention sinks), head 256 GQA 2/6/8, head 512 GQA 8 (Gemma-4 full layers), bf16 or int8 KV (`int8_per_token_head`), causal, optional sliding window; sinks on unsplit launches only, so verify batches with sinks stay on Triton |
| `attn_decode` | `attn_decode_bf16` | `TRITON_ATTN` impl for pure-decode batches | head 128 GQA 8 (Flint, sinks, sliding window), head 256 GQA 2 (sliding window), head 256 GQA 8 (Qwen3.6-35B, <= 2 sequences), head 512 GQA 8 (Gemma-4), bf16 or int8 KV |
| `attn_decode` | fork's Triton `unified_attention` (vendored) | `unified_attention` inside `TRITON_ATTN` for every batch strix kernels skip | the fork's gfx1151 split-KV decode, pooled-KV buffer loads, prefill configs |
| `w8a16` | `w8a16_skinny_gemm` (M <= 5; the op takes up to 8), `w8a8_wmma_gemm` | compressed-tensors W8A16 linears (`MPLinearKernel`, first in the ROCm list) | uint8b128 symmetric, per-channel or group 128, K % 128 == 0 |
| `w8a8` | `w8a8_wmma_gemm` | compressed-tensors W8A8 int8 linears (`Int8ScaledMMLinearKernel`, first in the ROCm int8 list) | int8 per-channel weights, dynamic per-token activations (quantized per 128-group inside the op), symmetric, K % 128 == 0 |
| `w4a16` | `w4a8_wmma_gemm` (M > 5), `w4a16_wmma_gemm` (g32, M > 128), vLLM's `wvSplitK_int4_g` (M <= 5) | `RDNAHybridW4A16LinearKernel` prefill (Triton) for symmetric int4 | uint4b8, group 128 / 32 / per-channel, K % 128 == 0; vLLM's packed layout, no extra weight copy |
| `fp8` | `fp8_skinny_gemm` (M <= 5), `fp8_wmma_gemm_raw` (M <= 256; `STRIX_DISABLE=fp8_raw` keeps the bf16-staging `fp8_wmma_gemm`) and `fp8_wmma_gemm` (M < 768), `fp8_dequant` + hipBLASLt (M >= 768, in 1024-row slices when K >= 16384) | vLLM's block-scaled FP8 linear (`TritonFp8BlockScaledMMKernel`, no tuned config here: 0.57 tok/s decode on the 27B) | float8_e4m3fn weights with 128x128 block scales (K % 128 == 0) or per-channel scales (compressed-tensors `channel`, K % 64 == 0; vLLM's `_POSSIBLE_FP8_KERNELS` list, `STRIX_DISABLE=fp8_channel` keeps stock), bf16 activations (W8A16, the per-token activation quantize is skipped); weights stay one byte each, block scales are also kept expanded per row (3 % more) |
| `mxfp4` | `mxfp4_gemv` (M <= 2), `mxfp4_wmma_gemm` | `EmulationMxfp4LinearKernel` for Quark MXFP4 linears | fp4 weights + e8m0 scales used in place, bf16 activations (W4A16), K % 128 == 0 |
| `moe_w4a8` | `moe_gemm_w4a8` + `moe_act_quant_i8` | takes over the WNA16 expert GEMMs from `moe_w4a16` for batch >= 1024 tokens: int8 activations per (row, 128-group) on the iu8 WMMA, int4 -> int8 at the LDS commit (1.2-1.34x the W4A16 tile at 16k Qwen3.8-Flash-Next tokens) | uint4, **group 128**, no zero-point/bias/LoRA/EP, bf16 activations, K % 128 == 0; TRITON backend only |
| `qsa_union` | `qsa_union_attn_bf16` | Qwen4Exp QSA sparse attention (`qsa_sparse_paged_attention`) for a single-request prefill of >= 1024 rows: 16 consecutive rows share one union of their selected tokens (1.5x the gfx1151 Triton kernel, more as the context grows) | head_dim 256, GQA 12, bf16 or int8_per_token_head cache, one request of at most 131072 positions (its own length, whatever `--max-model-len` is; past that an int8 cache is dequantized to a bf16 copy for vLLM's kernel); an int8 cache is dequantized into a compact fp16 K/V copy that the gather reads (one pass per layer, a host sync for the length), as `STRIX_QSA_UNION_F16=1` does for bf16 |
| `qsa_indexer` | `qsa_indexer_bf16` | the QSA indexer's paged scores (`qsa_mqa_paged`) for a single-request prefill of >= 128 rows (1.7-1.9x the tl.dot kernel) | 4 indexer heads, head_dim 128, one KV head, bf16, one request |
| `hc_up_gate_mix` | `hc_up_gate_mix_bf16` | Qwen4Exp hyper-connection up projection fused with `hc_gate_mix` (6.7 -> 5.2 ms per site at 16k rows, no gate in memory) | hc_count 4, bf16, K % 64 == 0, batch >= 512 rows |
| `dense_w8` | `w8a16_skinny_gemm` on int8 copies | **decode** copies of unquantized bf16 linears and the LM head: the bf16 weight stays for prefill, batches of <= 5 rows read an int8 copy, and 6-8 rows (a DFlash 2 verify step) on weights of 8M elements or more (Qwen3.8-Flash-Next decode 51 -> 37 ms/step; GSM8K 93 vs 92; LM head at 7 rows 3.3 ms against 5.5 ms bf16) | bf16 2-D weights, K % 16 == 0; each copy is checked against bf16 at load |

Some swaps live in code that `torch.compile` traces. vLLM keys its compile
caches on its own env vars and config only, so the plugin adds
`STRIX_VLLM_CACHE_KEY` (a digest of `STRIX_DISABLE`, `STRIX_ENABLE` and the
plugin source) to vLLM's cache key; without it, a graph traced with the plugin
off could be replayed with it on.

## The kernel table

`python/strix/registry.py` is the source of truth for what this library can
do: one row per kernel, the geometry it takes (`constraints`) and how it is
built (`spec`: tile, staging, why).

- `strix.kernels()` dumps the table, `strix.ops()` groups it by op.
- `strix.explain(op, **geometry)` says why a shape matches a row or not.
- `strix.select(op, **geometry)` returns the kernel a caller should use, or
  `None` to keep vLLM's own path.
- `strix.geometries(op, *keys)` returns the shape points a family covers.

The plugin builds its dispatch gates from the table rather than repeating
them: `python/strix_vllm/attn.py` reads the (head_dim, GQA) pairs it routes
from `geometries()`, and `tests/test_registry.py` fails if a `torch.ops.strix`
registration has no row.

## Against upstream vLLM

The kernels need no patched vLLM sources: the plugin runs on a stock upstream
build. A few things the recipes use come from the
[strix-vllm](https://github.com/asf0/strix-vllm) fork (branch `gfx1151`)
instead, which the Docker image ships:

- streaming Qwen3.8-Flash-Next's n-gram table from disk
  ([docker.md](docker.md#recipes));
- `--use-replayssm` on Qwen GDN models (upstream rejects the flag for Qwen3.5
  layers);
- the KV cache layout for draft models: a draft's layers get their own cache
  pool at their own page size instead of padding every page of the target up
  to theirs (with ReplaySSM, the 35B with DFlash 2 goes from 1.45M to 3.92M
  tokens of KV).

`bench/upstream.sh <cmd>` runs a command against a stock upstream checkout
named by `VLLM_UPSTREAM` that has the installed build's `vllm/*.so` copied in
and its own `vllm/_version.py` (the version is part of the compile-cache
key). It also puts the ROCm SDK's bundled `amdsmi` on `PYTHONPATH`, which
upstream needs to detect ROCm on this APU.

## Hardware notes that shaped the kernels

- Rows 4-16 KiB apart read at the same column offset camp on the same DRAM
  channels (4-8x slower). Pad weight row pitch, or order the grid so the
  head / feature index varies fastest.
- Keep staging registers in `ext_vector_type` vectors and check
  `-Rpass-analysis=kernel-resource-usage` for scratch: `uint4` arrays spilled.
- Use `strix::barrier()` for LDS hand-offs (explicit `lgkmcnt(0)`).

## Adding a kernel

1. `csrc/<family>/<entry_point>.hip`, geometry in the name, checks that reject
   a mismatch.
2. Declare and register it in `csrc/torch_ops.cpp`; add a fake impl in
   `python/strix/__init__.py`.
3. Add a row to `python/strix/registry.py`.
4. Test against a torch reference in `tests/`, benchmark against vLLM's
   current kernel in `bench/`.
5. Hook it up in `python/strix_vllm/` behind `select()` with fallback to the
   parent class.
