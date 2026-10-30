# libstrix

HIP kernels tuned for **gfx1151** (Strix Halo, Radeon 8060S), plus a vLLM
plugin that swaps them into stock vLLM. No vLLM source changes.

Each kernel is compiled for one geometry, named in its entry point, and rejects
any shape it was not built for. `strix.select(op, **geometry)` reports whether
a kernel covers a shape; `None` means vLLM keeps its own path. The plugin is a
no-op on every other GPU.

## Requirements

- A gfx1151 APU (Strix Halo / Radeon 8060S) and a ROCm install that ships
  `hipcc`.
- The ROCm build of PyTorch, CMake >= 3.26 and Ninja.
- vLLM with `vllm.general_plugins` entry-point support.

Build against the same torch and ROCm that the vLLM process loads — the
extension links against that torch, and a mismatch fails at import.

## Build and install

```sh
export ROCM_PATH=/opt/rocm          # your ROCm install
source /path/to/venv/bin/activate   # the python that runs vLLM
./build.sh                          # -> python/strix/_C.so
pip install --no-deps --no-build-isolation -e .
```

`build.sh` compiles with `$ROCM_PATH/lib/llvm/bin/clang++`; `STRIX_GPU_ARCH`
overrides the target architecture (default `gfx1151`) and `MAX_JOBS` the build
parallelism (default 8).

## Using it with vLLM

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

| `STRIX_DISABLE` name | kernel | replaces | when |
| --- | --- | --- | --- |
| `rms_norm_gated` | `gdn_gated_rmsnorm_h128_bf16` | `RMSNormGated` (GDN output norm) | bf16, 128 channels |
| `moe_pad` | (weight layout) | MoE expert weight storage | padded row pitch at `create_weights`; vLLM's own ROCm padding is undone by #46009 |
| `moe_gemm` | `moe_gemm_bf16` | both expert GEMMs in `TritonExperts`, gate-and-mul fused into w13 | bf16, no quant/LoRA/bias/EP, SiLU or GELU-tanh |
| `moe_w4a16` | `moe_gemm_w4a16` | both expert GEMMs in `TritonWNA16Experts` (compressed-tensors int4), gate-and-mul fused into w13 | uint4, group 32/64/128, bf16 scales, no zero-point/bias/LoRA/EP; TRITON backend only |
| `moe_w8a16` | `moe_gemm_w8a16` | both expert GEMMs in the int8 MoE path (`oracle/int8.py`, `TritonExperts`), gate-and-mul fused into w13. **Opt-in (`STRIX_MOE_W8A16=1`): measured slower than Triton's int8 WMMA on Qwen3.6-35B-A3B-INT8-W8A8** | int8 [E, N, K] experts, one scale per output channel (fp32/bf16/fp16), bf16 activations (W8A16), no bias/LoRA/EP, K % 64 == 0, N % 128 == 0; TRITON backend only |
| `gdn_chunk` | `gdn_chunk_fwd_k128_v128_c64_bf16` | `ChunkGatedDeltaRule` (FLA chunk prefill) | head 128, bf16, no in-kernel l2norm |
| `gdn_conv` | `gdn_causal_conv1d_w4_bf16`, `gdn_post_conv_bf16` | `causal_conv1d_fn` and `fused_post_conv_prep` in the GDN prefill path (module globals rebound) | width 4, bf16 x/state, heads 128, L2 norm on, plain g |
| `gemm_rows` | (hipBLASLt, sliced) | `rocm_unquantized_gemm` for bf16 linears with K >= 16384 | runs > 1024 rows as 1024-row slices (31 -> 41 TFLOPS on 27B MLP down at 8k rows) |
| `attn_prefill` | `attn_prefill_bf16` | `TRITON_ATTN` impl for prefill and speculative verify | head 64 GQA 4 (LFM2.5), head 256 GQA 2/6/8, head 512 GQA 8 (Gemma-4 full layers), bf16 or int8 KV (`int8_per_token_head`), causal, no window/sinks |
| `attn_decode` | `attn_decode_bf16` | `TRITON_ATTN` impl for pure-decode batches | head 256 GQA 2 (sliding window), head 512 GQA 8 (Gemma-4), bf16 or int8 KV |
| `attn_decode` | fork's Triton `unified_attention` (vendored) | `unified_attention` inside `TRITON_ATTN` for every batch strix kernels skip | the fork's gfx1151 split-KV decode, pooled-KV buffer loads, prefill configs |
| `w8a16` | `w8a16_skinny_gemm` (M <= 5), `w8a8_wmma_gemm` | compressed-tensors W8A16 linears (`MPLinearKernel`, first in the ROCm list) | uint8b128 symmetric, per-channel or group 128, K % 128 == 0 |
| `w8a8` | `w8a8_wmma_gemm` | compressed-tensors W8A8 int8 linears (`Int8ScaledMMLinearKernel`, first in the ROCm int8 list) | int8 per-channel weights, dynamic per-token activations (quantized per 128-group inside the op), symmetric, K % 128 == 0 |
| `w4a16` | `w4a8_wmma_gemm` (M > 5), `w4a16_wmma_gemm` (g32, M > 128), vLLM's `wvSplitK_int4_g` (M <= 5) | `RDNAHybridW4A16LinearKernel` prefill (Triton) for symmetric int4 | uint4b8, group 128 / 32 / per-channel, K % 128 == 0; vLLM's packed layout, no extra weight copy |
| `mxfp4` | `mxfp4_gemv` (M <= 2), `mxfp4_wmma_gemm` | `EmulationMxfp4LinearKernel` for Quark MXFP4 linears | fp4 weights + e8m0 scales used in place, bf16 activations (W4A16), K % 128 == 0 |
| `moe_w4a8` | `moe_gemm_w4a8` + `moe_act_quant_i8` | takes over the WNA16 expert GEMMs from `moe_w4a16` for batch >= 1024 tokens: int8 activations per (row, 128-group) on the iu8 WMMA, int4 -> int8 at the LDS commit (1.2-1.34x the W4A16 tile at 16k Qwen3.8-Flash-Next tokens) | uint4, **group 128**, no zero-point/bias/LoRA/EP, bf16 activations, K % 128 == 0; TRITON backend only |
| `qsa_union` | `qsa_union_attn_bf16` | Qwen4Exp QSA sparse attention (`qsa_sparse_paged_attention`) for a single-request prefill of >= 1024 rows: 16 consecutive rows share one union of their selected tokens (1.5x the gfx1151 Triton kernel, more as the context grows) | head_dim 256, GQA 12, bf16, one request, positions < 131072; `STRIX_QSA_UNION_F16=1` gathers from a compact fp16 K/V copy instead (needs a host sync) |
| `qsa_indexer` | `qsa_indexer_bf16` | the QSA indexer's paged scores (`qsa_mqa_paged`) for a single-request prefill of >= 128 rows (1.7-1.9x the tl.dot kernel) | 4 indexer heads, head_dim 128, one KV head, bf16, one request |
| `hc_up_gate_mix` | `hc_up_gate_mix_bf16` | Qwen4Exp hyper-connection up projection fused with `hc_gate_mix` (6.7 -> 5.2 ms per site at 16k rows, no gate in memory) | hc_count 4, bf16, K % 64 == 0, batch >= 512 rows |
| `dense_w8` | `w8a16_skinny_gemm` on int8 copies | **decode** copies of unquantized bf16 linears and the LM head: the bf16 weight stays for prefill, batches of <= 5 rows read an int8 copy (Qwen3.8-Flash-Next decode 51 -> 37 ms/step; GSM8K 93 vs 92) | bf16 2-D weights, K % 16 == 0; each copy is checked against bf16 at load |

Some swaps live in code that `torch.compile` traces. vLLM keys its compile
caches on its own env vars and config only, so the plugin adds
`STRIX_VLLM_CACHE_KEY` (a digest of `STRIX_DISABLE`, `STRIX_ENABLE` and the
plugin source) to vLLM's cache key; without it, a graph traced with the plugin
off could be replayed with it on.

### The kernel table

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

### Against upstream vLLM

Nothing here needs patched vLLM sources. `bench/upstream.sh <cmd>` runs a
command against a stock upstream checkout named by `VLLM_UPSTREAM` that has
the installed build's `vllm/*.so` copied in and its own `vllm/_version.py`
(the version is part of the compile-cache key). It also puts the ROCm SDK's
bundled `amdsmi` on `PYTHONPATH`, which upstream needs to detect ROCm on this
APU.

### Qwen3.8-Flash-Next

This model is the exception: its 102 GB n-gram (engram) table does not fit in
128 GB next to the weights. The [strix-vllm](https://github.com/asf0/strix-vllm)
fork (branch `gfx1151`) streams the table from disk, so run it there with the
plugin installed. Use the AWQ W4A16 checkpoint; bf16 does not fit.

```sh
vllm serve /path/to/Qwen3.8-Flash-Next-AWQ-W4A16 \
  --engram-config '{"disk_offload": true}' --language-model-only \
  --gpu-memory-utilization 0.80 --max-model-len 131072 \
  --max-num-batched-tokens 16384 --no-enable-prefix-caching \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

- `--gpu-memory-utilization 0.80` leaves page cache for the table rows. A
  lower setting prefills slower; a higher one leaves too little host memory.
- 16k prefill chunks are the measured best; 32k is not faster.
- MTP 2 costs about 4 % prefill and more than doubles decode (18 -> 40-43
  tok/s on real text). Drop `--speculative-config` for prefill-only work.
- Keep `gdn_conv` enabled: with vLLM's own GDN causal conv this model fails at
  startup (out-of-range n-gram ids), so `STRIX_DISABLE=all` does not run.
- The fork's [engram docs](https://github.com/asf0/strix-vllm/blob/gfx1151/docs/features/engram.md#disk-offload)
  describe the disk path and its debugging switches.

## Results

Stock upstream vLLM (`899a1689e2`, unpatched) with and without the plugin on
one gfx1151 APU, single stream, int8 KV. Speedup is plugin over stock; the
last column is the plugin's own throughput. Full tables, batched decode,
serving and accuracy are in [docs/performance.md](docs/performance.md).

| model | prefill 2k / 64k | decode 2k / 64k | 128k prefill / decode tok/s |
| --- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 2.7× / 19.3× | 1.86× / 2.35× | 1,751 / 26.4 |
| Qwen3.6-35B-A3B W8A8 | 2.1× / 18.3× | 1.03× / 1.53× | 1,697 / 24.5 |
| Qwen3.8-27B bf16 | 1.7× / 14.3× | 1.01× / 1.14× | 435 / 3.9 |
| Qwen3.8-27B INT8 (W8A16) | 2.6× / 14.6× | 2.38× / 2.33× | 430 / 5.7 |
| Qwen3.8-27B Quark INT4 | 2.1× / 14.2× | 1.01× / 1.22× | 428 / 8.1 |
| LFM2.5-24B-A2B | 2.0× / 11.4× | 2.15× / 2.18× | 2,405 / 27.1 |
| gemma-4-26B-A4B | 4.4× / 31.1× | 1.28× / 3.33× | 1,588 / 21.5 |
| gemma-4-26B-A4B AWQ | 3.9× / 28.2× | 1.36× / 4.28× | 1,499 / 29.6 |
| gemma-4-31B MXFP4 | 2.7× / 21.0× | 6.21× / 6.99× | 296 / 7.3 |
| gemma-4-31B W4A16 | 3.0× / 22.5× | 1.32× / 2.99× | 292 / 8.0 |
| Qwen3.8-Flash-Next AWQ ¹ | 1.8× / 2.0× | 1.89× / 1.58× | 1,635 / 25.1 |

¹ On the strix-vllm fork with bf16 KV and 16k chunks, against everything-off
except `gdn_conv` (see [above](#qwen38-flash-next)). Decode is without MTP.
With MTP 2 on real text it decodes at 40-43 tok/s and prefills at
1,540-1,620 tok/s from 5.6k to 100k tokens. Batched decode gains 1.16-1.26×.

Prefill gains grow with context: at 256k the 35B prefills in 241 s against
7,568 s stock. Decode doubles where the plugin replaces the dominant op and
is at parity elsewhere; no configuration, single stream or batched, is
slower than stock. KLD against stock is at the noise floor on the Qwen3.6
and Qwen3.8-27B checkpoints (0.001-0.006, 97-99 % top-1). Qwen3.8-Flash-Next
sits higher (0.008-0.019, 97-98 % top-1, perplexity within 0.6 %), most likely because its
MoE GEMM quantizes activations to int8. gemma-4-26B drifts further (0.2),
which `docs/performance.md` explains. Serving the 27B INT8 + MTP at 40
sequences gives 123 output tok/s.

Earlier measurements (before the re-run above),
with MTP k=2 and the plugin (decode x1 / x4 / x16 tok/s): 35B 56 / 118 / 210,
27B bf16 10.6 / 40 / 132, Quark INT4 26.6 / 73 / 125. GSM8K (plugin off vs
on, paired, n = 250 unless noted) shows no significant change on any of
them: 35B 1254 / 1249 (n = 1319), 27B 237 / 235, Quark 241 / 238, Gemma 247
/ 247, LFM2.5 196 / 197, 35B + MTP 237 / 238.

Not covered: LFM2.5 NVFP4 (vLLM forces Marlin for weight-only NVFP4 on ROCm)
and `--use-replayssm` for Qwen GDN, which is a vLLM core change (upstream
rejects the flag for Qwen3.5 layers).

## Tests and benchmarks

```sh
python -m pytest tests -q          # per-kernel correctness vs torch references
bench/run_guarded.sh python bench/kernels/moe_gemm.py   # etc.; one GPU job at a time
```

`bench/run_guarded.sh` kills the whole process group if host `MemAvailable`
drops below `STRIX_MIN_AVAIL_GB` (default 10) or after `STRIX_TIMEOUT_S`
(default 3600): GPU memory is host RAM on this APU and an OOM takes the
machine down.

`bench/model/profile_model.py` + `tools/analyze_trace.py` give per-kernel phase
breakdowns under rocprofv3 (use `--enforce-eager` for models whose graph
replay hangs the profiler); the W8A16/W4A16 linears report their mainloop,
activation-quantize pass and split-K reduce as separate categories, since
which of the three dominates decides the next tile change.
`bench/model/e2e_check.py` and `bench/model/gsm8k_check.py` are greedy and
accuracy A/B checks with and without the plugin; `bench/model/kld_check.py`
needs a wikitext-2-raw directory via `--corpus` or `STRIX_WIKITEXT`.

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

## Layout

```
csrc/           HIP kernels (attn, gdn, gemm, moe, norm) + shared headers
python/strix/   torch.ops.strix bindings and the kernel/geometry table
python/strix_vllm/  the vLLM plugin: one module per swapped subsystem
bench/kernels/  per-kernel microbenchmarks against vLLM's own kernels
bench/model/    model-level checks: phase profiling, KLD, e2e and GSM8K A/B
tools/          trace analysis and kernel tuning helpers
tests/          per-kernel correctness against torch references
```

## License

MIT (`LICENSE`). Three files derived from
[vLLM](https://github.com/vllm-project/vllm) keep its Apache-2.0 license;
`NOTICE` lists them.
