# Docker recipes

`docker/recipes/` has one compose file per tested model, all built on
`ghcr.io/asf0/strix-vllm` (the strix-vllm fork, this plugin and ROCm). Host
setup and the quick start are in the [README](../README.md#docker).

## Recipes

Memory at the default `GPU_UTIL=0.85`, from each recipe's startup log: the
weights, the KV cache left after them, and how many full 131k-token requests
that cache holds at once. Multimodal checkpoints serve images
(`--mm-encoder-attn-backend=TRITON_ATTN`); the vision encoder and its input
budget take 1-2 GiB of KV. Every recipe marked ✓ answered a text and an image
request in this configuration.

| recipe | model | weights | KV cache | KV tokens | 131k requests | images |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| `qwen3.6-35b-a3b` | Qwen/Qwen3.6-35B-A3B + incoai/Qwen3.6-35B-A3B-DFlash2 (7 draft tokens) | 71.0 GiB | 27.3 GiB | 2,441,861 | 18.6 | ✓ |
| `qwen3.6-35b-a3b-w8a8` | Avesed/Qwen3.6-35B-A3B-INT8-W8A8 + incoai/Qwen3.6-35B-A3B-DFlash2 (7 draft tokens) | 37.5 GiB | 60.7 GiB | 5,422,458 | 41.4 | ✓ |
| `qwen3.6-35b-a3b-fp8-dflash2` | Qwen/Qwen3.6-35B-A3B-FP8 + incoai/Qwen3.6-35B-A3B-DFlash2 (7 draft tokens) | 35.9 GiB | 50.2 GiB | 3,923,703 | 29.9 | ✓ |
| `sage-a1.5-fp8` | InternScience/Sage-A1.5-FP8 + incoai/Qwen3.6-35B-A3B-DFlash2 (7 draft tokens) | 37.5 GiB | 58.8 GiB | 4,994,430 | 38.1 | ✓ |
| `minicpm-v-4.7-35b-a3b-fp8` | openbmb/MiniCPM-V-4.7-35B-A3B-FP8 + openbmb/MiniCPM-V-4.7-35B-A3B-DSpark (4 draft tokens) | 36.1 GiB | 61.9 GiB | 6,233,083 | 47.6 | ✓ |
| `qwen3.8-27b` | Qwen/Qwen3.8-27B + incoai/Qwen3.8-27B-DFlash2 (7 draft tokens) | 53.3 GiB | 49.2 GiB | 1,435,398 | 11.0 | ✓ |
| `qwen3.8-27b-int8` | lued/Qwen3.8-27B-INT8-W8A16-MTP + incoai/Qwen3.8-27B-DFlash2 (7 draft tokens, 40 sequences) | 30.6 GiB | 71.5 GiB | 2,085,962 | 15.9 | ✓ |
| `qwen3.8-27b-awq-int4` | amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 + incoai/Qwen3.8-27B-DFlash2 (7 draft tokens) | 17.7 GiB | 85.0 GiB | 2,641,486 | 20.2 | ✓ |
| `qwen3.8-27b-awq-mxfp4` | amd/Qwen3.8-27B-Quark-AWQ-MXFP4 + incoai/Qwen3.8-27B-DFlash2 (7 draft tokens, not benchmarked) | 17.9 GiB | 85.2 GiB | 2,647,654 | 20.2 | ✓ |
| `qwen3.8-27b-fp8-dflash2` | Qwen/Qwen3.8-27B-FP8 + incoai/Qwen3.8-27B-DFlash2 (7 draft tokens) | 33.6 GiB | 61.6 GiB | 1,737,503 | 13.3 | ✓ |
| `qwen3.8-flash-next-dflash2` | wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 (n-gram table streamed from disk) + incoai/Qwen3.8-Flash-Next-DFlash2 (4 draft tokens) | 77.0 GiB | 14.1 GiB | 627,020 | 4.8 | ✓ |
| `lfm2.5-24b-a2b` | LiquidAI/LFM2.5-24B-A2B | 46.2 GiB | 58.1 GiB | 5,719,435 | 43.6 | text model |
| `lfm2.5-24b-a2b-fp8-dspark` | LiquidAI/LFM2.5-24B-A2B-FP8 + LiquidAI/LFM2.5-24B-A2B-DSpark (4 draft tokens) | 24.1 GiB | 73.8 GiB | 4,837,780 | 36.9 | text model |
| `muse-glimmer-30b-fp8-dflash2` | RedHatAI/Muse-Glimmer-30B-FP8-block + meta-models/Muse-Glimmer-30B-assistant (15 draft tokens) | 38.5 GiB | 56.4 GiB | 5,380,274 | 41.0 | ✓ |
| `flint-fp8` | HawkBearPig/flint-FP8-BF16 + incoai/Flint-DFlash2 (3 draft tokens) | 42.9 GiB | 54.6 GiB | 2,667,556 | 20.4 | text model |
| `inkling-2-mini` | thinkingmachines/Inkling-2-Mini + its DSpark draft (2 drafts for one request; `GPU_UTIL` 0.80; `bench-inkling-2-mini` is the 32k benchmark variant) | 20.4 GiB | 75.6 GiB | 468,701 | 3.6 | ✓ |
| `gemma-4-26b-a4b` | google/gemma-4-26B-A4B-it + z-lab/gemma-4-26B-A4B-it-DFlash (15 draft tokens) | 50.5 GiB | 52.7 GiB | 2,041,996 | 15.6 | ✓ |
| `gemma-4-26b-a4b-awq-int4` | cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit + z-lab/gemma-4-26B-A4B-it-DFlash (15 draft tokens) | 16.6 GiB | 86.3 GiB | 3,340,475 | 25.5 | ✓ |
| `gemma-4-26b-a4b-fp8` | RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic + z-lab/gemma-4-26B-A4B-it-DFlash (15 draft tokens) | 25.8 GiB | 77.2 GiB | 2,990,822 | 22.8 | ✓ |
| `gemma-4-31b-mxfp4` | amd/gemma-4-31B-it-MXFP4 + z-lab/gemma-4-31B-it-DFlash (15 draft tokens) | 17.8 GiB | 84.3 GiB | 652,978 | 5.0 | text only |
| `gemma-4-31b-qat-int4` | google/gemma-4-31B-it-qat-w4a16-ct + z-lab/gemma-4-31B-it-DFlash (15 draft tokens) | 19.9 GiB | 79.3 GiB | 614,229 | 4.7 | ✓ |

Rows for recipes that gained a draft after their last smoke run still show the
no-draft weights/KV numbers; they are re-measured at the next smoke.

Overrides go in `.env` or the shell: `GPU_UTIL` (default 0.85), `MAX_MODEL_LEN`,
`MAX_NUM_SEQS`, `MAX_BATCHED_TOKENS`, `MTP` (draft tokens, `0` = off),
`SPEC_TOKENS` (DFlash and DSpark draft tokens), `REPLAYSSM` (`1` = one GDN state per
request when speculating; on in the Qwen recipes), `SERVED_MODEL_NAME`, `STRIX_DISABLE`,
`STRIX_ENABLE` (see [kernels.md](kernels.md)), `HF_TOKEN`,
`PORT`, and `EXTRA_ARGS` (appended to `vllm serve`). `CACHE` keeps the compile
caches, so later starts skip recompiling; the container runs as root, so
removing it needs `sudo`. For local test runs that should not write compile
artifacts to the SSD, point `CACHE` at a tmpfs directory such as
`/dev/shm/strix-cache`; `bench/container/run.sh` and the Python harness also
take `CACHE_TMPFS=1`, which mounts a sized tmpfs (`CACHE_SIZE`, default 8g)
for the container only. Models must be present under `MODELS` with a tmpfs: a
Hugging Face download will not fit.

`qwen3.8-flash-next-dflash2` needs the strix-vllm fork: its 102 GB n-gram
(engram) table does not fit next to the weights, so the fork streams it from
disk
([engram docs](https://github.com/asf0/strix-vllm/blob/gfx1151/docs/features/engram.md#disk-offload)).
Use the AWQ W4A16 checkpoint; bf16 does not fit.

- `GPU_UTIL=0.85` leaves page cache for the table rows; lower prefills slower.
  0.88 holds more KV but leaves only about 11 GiB of host memory free, 2 GiB
  above the memory guard.
- 16k prefill chunks (`MAX_BATCHED_TOKENS`) are the measured best.
- Speculation runs with the incoai/Qwen3.8-Flash-Next-DFlash2 draft: 19 %
  faster for one request and 16 % with 8 at once at the default 4 draft
  tokens ([speculative decoding](performance.md#speculative-decoding)).
  ReplaySSM is not supported for this architecture.
- Keep `gdn_conv` enabled: with vLLM's own GDN causal conv this model fails at
  startup, so `STRIX_DISABLE=all` does not run.
- fp8 KV (`--kv-cache-dtype=fp8`, unit scales): vLLM's fp8 QSA cache, which
  libstrix casts to bf16 per request for its union prefill kernel, at the
  bf16 kernel's speed (vLLM's own fp8 kernel is 4-7.6x slower per layer).
  KLD 0.0225 against bf16 KV at 2k
  ([checkpoint notes](performance.md#checkpoint-notes)).

`flint-fp8` and `lfm2.5-24b-a2b-fp8-dspark` need the
strix-vllm fork: it runs Flint on ROCm, and hands Flint's and LFM2.5's hidden
states to their draft models. Flint writes `<think>` blocks and `<tool_call>`
calls in the GLM layout, so its recipes use the `glm45` tool and reasoning
parsers.

Every recipe sets the tool-call and reasoning parsers its chat template needs
(`qwen3_coder` + `qwen3` for Qwen and Sage, `gemma4`, `lfm2` + `deepseek_r1`,
`glm45`, `muse_glimmer`, `qwen3_coder` + `minicpmv`), so chat clients that send
`tool_choice: auto` work and the thinking comes back as `reasoning` instead of
in the answer. `minicpm-v-4.7-35b-a3b-fp8` also sets
`--default-chat-template-kwargs` to turn thinking on: its template leaves
thinking off unless asked, and then the answer has no `</think>` for the
parser to split on.

Every recipe except Muse-Glimmer (greedy, `do_sample: false` in its
`generation_config.json`) sets the model card's sampling defaults with
`--override-generation-config`, so a client that sends no sampling fields gets
them even when the checkpoint folder has no `generation_config.json`. A
request's own values win.

| family | temperature | top_p | top_k | other |
| --- | ---: | ---: | ---: | --- |
| Qwen3.6-35B-A3B (bf16, W8A8, FP8), Sage-A1.5 | 1.0 | 0.95 | 20 | presence_penalty 1.5 |
| MiniCPM-V-4.7, Qwen3.8-27B (all), Qwen3.8-Flash-Next, Flint | 1.0 | 0.95 | 20 | |
| LFM2.5-24B-A2B (bf16, FP8) | 0.1 | | 50 | repetition_penalty 1.05 |
| gemma-4 (all) | 1.0 | 0.95 | 64 | |

Sage-A1.5 needs them for tool calls: greedy, it reasons in circles past 4,096
tokens without calling the tool; with its defaults it called it in 7 of 8
tries, after 52-251 tokens.

## NPU prefill

Qwen3.8-Flash-Next and Qwen3.8-27B, off by default
(`STRIX_ENABLE=npu`, see
[kernels.md](kernels.md#turning-kernels-on-and-off)): the XDNA2 NPU computes
the input projections of prefill steps of 4,096 tokens or more — one request
or several — while the GPU runs the rest of the layer: TTFT 7-8 % lower from
16k to 100k on Flash-Next, prefill about 7 % faster on the 27B (served,
8k-25k prompts; 6.7-7.3 % offline at 8k-16k; bf16 - the
FP8 and INT8 checkpoints are not covered, see below).
Layer the
override over the recipe and leave room for the NPU's pinned memory:

```sh
GPU_UTIL=0.82 docker compose -f qwen3.8-flash-next-dflash2.yaml \
  -f npu/qwen3.8-flash-next-dflash2.yaml up -d
GPU_UTIL=0.82 docker compose -f qwen3.8-27b.yaml \
  -f npu/qwen3.8-27b.yaml up -d
```

| needs | why |
| --- | --- |
| kernel 7.4+, `amdxdna` loaded (`/dev/accel/accel0`), the IOMMU enabled with the GPU in a passthrough (identity) group | the override passes the device in; `amd_iommu=off`, the GPU-only recommendation, keeps the NPU off |
| `ulimits: memlock: -1` (in the override) | weight panels and arenas are pinned host memory |
| `GPU_UTIL=0.78-0.82` (Flash-Next DFlash 2) | at 0.85 host MemAvailable falls to 9 GiB at the first long prompt, the memory guard's limit; 0.78 keeps 4.59 GiB of KV (203,692 tokens) and served a full benchmark, 0.82 keeps 412,699 KV tokens with the NPU, 471,150 without |
| `GPU_UTIL=0.82` (the 27B) | at 0.85 host MemAvailable falls to 6 GiB at startup, below the memory guard; at 0.82 the NPU opens during the startup profile run and its panels pin ~10.5 GiB before the KV cache is sized: 826,647 KV tokens with the NPU resident (1,121,559 without), 6.31 requests at 131k |

The log shows `libstrix: npu prefill on: 46 layers resident` (Flash-Next) or
`libstrix: npu prefill on (Qwen3.5): 47 GDN layers ...` (the 27B) after startup.

Steps of several requests take the NPU path too (the banks are token ranges;
a request straddling the boundary appears in both), and the QSA indexer's
projection rides the panels unless `STRIX_NPU_INDEXER=0`. A full benchmark
with the NPU on: 40 of 71 NPU steps were multi-request or ragged packs, with
the GPU never waiting at a chain. `STRIX_NPU_STATS=<file>` logs per-chain
relay stats and GPU windows for
[bench/model/npu_stats.py](../bench/model/npu_stats.py).
Without a usable NPU (no device, no driver, the memlock limit, no
`libstrix_npu.so`) it logs `libstrix: npu prefill off: <reason>` once and runs
the GPU path, with the full KV cache (471,150 tokens at 0.82). A failed NPU
chain during a step fails that request and turns the NPU off for later steps.
Speed and accuracy: [performance.md](performance.md#npu-prefill).

## Memory guard

GPU memory here is host RAM, so a run that asks for too much does not fail
with an out-of-memory error: the kernel OOM-kills the whole machine. The
entrypoint watches two numbers and stops vLLM (exit 137) before that happens:

| variable | default | stops vLLM when |
| --- | ---: | --- |
| `STRIX_MIN_AVAIL_GB` | 9 | host `MemAvailable` drops below this many GB |
| `STRIX_MIN_GPU_FREE_GB` | 14 | less than this many GB are left under the GPU memory cap (the driver fails a few GB before the cap, while `MemAvailable` still looks fine) |

`0` turns a check off. Every recipe stays within both at `GPU_UTIL=0.85` and
the 117 GiB cap.

## Building the image

About 40 minutes, mostly vLLM's C++: `docker/build-and-push.sh`, or
`docker build -f docker/Dockerfile .` from the repo root.

**Releases.** GitHub Actions builds the published image (`.github/workflows/image.yml`,
no GPU needed). Pushing a tag `vX.Y.Z` builds the strix-vllm commit pinned in
`docker/strix-vllm.ref`, pushes `ghcr.io/asf0/strix-vllm:X.Y.Z` and `:latest`, and
publishes a GitHub release whose notes are the top entry of
[CHANGELOG.md](../CHANGELOG.md) plus the libstrix and fork commits (the fork commit
is also tagged `strix/vX.Y.Z` in asf0/strix-vllm). The release carries
`libstrix-recipes-X.Y.Z.zip`: these recipes with the image pinned to `X.Y.Z`, ready to
unzip and `docker compose up` without cloning. Pin a version in
your recipes (`IMAGE=ghcr.io/asf0/strix-vllm:0.1.0`) to keep it across releases. A
manual run of the workflow (any fork branch or commit) builds a dev image tagged
`<libstrix commit>-<fork commit>`, without a release.

**What the image leaves out.** Only the build stage has the full ROCm SDK. The
runtime keeps `_rocm_sdk_core` and `_rocm_sdk_libraries` without the ROCm
compiler (`hipcc`, clang, lld), headers, static libraries and MLIR, and the
Python packages that compile kernels with `hipcc` at runtime: tilelang,
flydsl and aiter. No recipe needs them.

- DeepSeek V4 and GLM-5-Next use tilelang for their mHC layers on gfx1151;
  without it vLLM runs its PyTorch version of those layers (not measured).
- To serve them with tilelang, build an image that keeps the compiler and
  installs tilelang: keep `_rocm_sdk_core/lib/llvm/bin`,
  `_rocm_sdk_core/lib/llvm/include` and clang's `libclang_rt` builtins in the
  `/stage` step of `docker/Dockerfile`, link `_rocm_sdk_core/amdgcn` to
  `lib/llvm/amdgcn` (where clang looks for the device libraries), add the
  unversioned `lib*.so` names `hipcc` links against, and `pip install
  tilelang==0.1.15 z3-solver==4.15.4.0`.

`STRIX_VLLM_LOCAL=/path/to/strix-vllm docker/build-and-push.sh` builds from a
local fork checkout instead of GitHub: its committed HEAD goes in as the
BuildKit named context `strix-vllm-local`, the image is tagged
`strix-vllm:local`, and the script will not push it.
