# libstrix

**Fast vLLM on AMD Strix Halo** (Ryzen AI Max, Radeon 8060S, gfx1151).
One `docker compose` command serves Qwen3.6, Qwen3.8, gemma-4, LFM2.5,
MiniCPM-V, Muse-Glimmer and Flint behind an OpenAI-compatible API, with
prefill 11-31x faster than stock vLLM at 64k context and decode up to 2.4x
faster (more on gemma-4 and the FP8 27B).

```sh
git clone https://github.com/asf0/libstrix && cd libstrix/docker/recipes
cp .env.example .env              # set MODELS=/path/to/your/models
docker compose -f qwen3.6-35b-a3b-fp8.yaml up -d
```

Host setup (GPU memory cap, BIOS) is a one-time step; see [Docker](#docker).

Under the hood: HIP kernels tuned for gfx1151, a vLLM plugin that swaps them
into stock vLLM with no vLLM source changes, and an option to run prefill input
projections on the Strix Halo **XDNA2 NPU** while the GPU runs the rest of the
layer. Each kernel is compiled for one geometry, named in its entry point, and
rejects any shape it was not built for. `strix.select(op, **geometry)` reports
whether a kernel covers a shape; `None` means vLLM keeps its own path. The
plugin is a no-op on every other GPU.

## Results

Against stock upstream vLLM on one gfx1151 APU, single stream, int8 KV
(full tables in [docs/performance.md](docs/performance.md)):

- **Prefill** 1.7-4.4× faster at 2k and 11-31× at 64k (Flash-Next 2.4×,
  against a different baseline; Flint FP8 8.8× at 2k); at 256k the 35B
  prefills in 241 s against 7,568 s stock.
- **Decode** 1.9-2.4× on the 35B, LFM2.5 and the INT8 27B, 3-7× on gemma-4 at
  64k, 4.6× on Flint FP8, 13.5× on the FP8 27B, and at parity on the bf16 and
  AWQ 27B and the 35B W8A8. No configuration, single stream or batched, is
  slower than stock, apart from Qwen3.8-Flash-Next's short prompts on its
  disk-backed n-gram table.
- **Accuracy:** KLD against stock 0.001-0.006 on Qwen3.6 and Qwen3.8-27B (97-99 %
  top-1), higher on Flash-Next and gemma-4-26B; GSM8K unchanged within noise on
  every model checked.
- **Speculative decoding:** DFlash 2 decodes the FP8 27B at 29.6 tok/s (7.7
  without a draft, 16.4 with MTP 2) and the FP8 35B at 85.1 (45.7 / 63.5);
  LFM2.5's DSpark draft takes it from 51.7 to 66.7 tok/s.
- **NPU prefill** (Flash-Next, `STRIX_ENABLE=npu`, kernel 7.4+): TTFT 7-8 %
  lower from 16k to 100k tokens served and prefill 9-11 % faster at 8k-32k
  offline, decode and short prompts unchanged, GSM8K unchanged; the NPU also
  covers multi-request steps. See
  [docs/docker.md](docs/docker.md#npu-prefill).

## Docker

`ghcr.io/asf0/strix-vllm` bundles the [strix-vllm](https://github.com/asf0/strix-vllm)
fork (branch `gfx1151`), this plugin and ROCm 10.1 on Ubuntu 26.04 / Python 3.14.
Nothing to build. `docker/recipes/` has one compose file per tested model.

### Host setup (once)

The image cannot set these; they belong to the host kernel.

- **GPU memory.** By default the iGPU can map only part of the 128 GB, and
  every recipe asks for 85 % of it. Raise the cap on the kernel command line,
  but not to all of RAM: GPU memory is host memory the kernel cannot reclaim,
  so a cap near the total lets one run take the whole machine down. The
  recipes were tested with a 117 GiB cap (`amdgpu.gttsize` in MiB,
  `ttm.pages_limit` in 4 KiB pages). `amd_iommu=off` is also recommended for
  the GPU alone — every number in this README was measured with it — but it
  keeps the NPU off: the XDNA2 prefill path needs the IOMMU enabled, the GPU
  in a passthrough (identity) group, a kernel at 7.4 or newer, and the NPU's
  memory pinnable (`ulimits: memlock: -1`, set by the `npu/` recipe
  overrides). On Ubuntu, GPU-only:

  ```sh
  sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/&amdgpu.gttsize=120000 ttm.pages_limit=30720000 amd_iommu=off /' /etc/default/grub
  sudo update-grub && sudo reboot
  cat /sys/class/drm/card*/device/mem_info_gtt_total   # ~117 GiB afterwards
  ```

  With the NPU, drop `amd_iommu=off` and boot with the GPU's IOMMU group in
  passthrough (e.g. `iommu.passthrough=1`); `/dev/accel/accel0` must exist
  after the reboot (the `amdxdna` driver).

- **BIOS.** Keep the dedicated VRAM (UMA frame buffer) small, e.g. 2 GB; the
  GPU uses the shared memory above, so a large carve-out only takes it away.
- **Docker** with Compose v2 (`docker.io docker-compose-v2` on Ubuntu). The
  amdgpu kernel driver must expose `/dev/kfd` and `/dev/dri`; no ROCm install is
  needed on the host.

### Running a model

```sh
git clone https://github.com/asf0/libstrix && cd libstrix/docker/recipes
cp .env.example .env              # set MODELS=/path/to/your/models
docker compose -f qwen3.8-flash-next.yaml up -d
docker compose -f qwen3.8-flash-next.yaml logs -f   # first start compiles: minutes
curl localhost:8000/v1/models
```

Each recipe serves `/models/<folder>` and downloads the model's Hugging Face
repo into `CACHE` when the folder is missing (gated repos such as gemma-4 need
`HF_TOKEN` in `.env`). For local testing that should not write compile
artifacts to the SSD, point `CACHE` at a tmpfs directory such as
`/dev/shm/strix-cache`, or set `CACHE_TMPFS=1` for the bench scripts. Run one
model at a time; each takes most of the 128 GB.

See [docs/docker.md](docs/docker.md) for every recipe with its memory and KV
capacity, the settings you can override (`GPU_UTIL`, `MAX_MODEL_LEN`, `MTP`,
`SPEC_TOKENS`, ...), the memory guard and how to build the image.

The image ships the [strix-vllm](https://github.com/asf0/strix-vllm) fork
rather than stock vLLM: it streams Qwen3.8-Flash-Next's n-gram table from
disk and gives its QSA layers an int8 KV cache, runs Flint on ROCm, supports
`--use-replayssm` on the Qwen GDN models, hands Flint's and LFM2.5's hidden
states to their DFlash 2 and DSpark drafts, and gives draft models their own
KV cache pool. The kernels themselves run on stock vLLM.

## Building from source

### Requirements

- A gfx1151 APU (Strix Halo / Radeon 8060S) and a ROCm install that ships
  `hipcc`.
- The ROCm build of PyTorch, CMake >= 3.26 and Ninja.
- vLLM with `vllm.general_plugins` entry-point support.

Build against the same torch and ROCm that the vLLM process loads — the
extension links against that torch, and a mismatch fails at import.

### Build and install

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
does. `STRIX_DISABLE=all`, or a comma-separated list of kernel names, keeps
vLLM's own kernels for A/B runs; `STRIX_ENABLE` turns on the ones that are
off by default. [docs/kernels.md](docs/kernels.md) lists every kernel, what
it replaces and when it applies, plus how to add one.

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

Conventions for docs, recipes and model names are in [AGENTS.md](AGENTS.md).

## Support

libstrix is built and tested on one machine in spare time. If it saves you
time, [a Ko-fi tip](https://ko-fi.com/asfdev) helps keep the tests and
development going.

## License

MIT (`LICENSE`). Three files derived from
[vLLM](https://github.com/vllm-project/vllm) keep its Apache-2.0 license;
`NOTICE` lists them.
