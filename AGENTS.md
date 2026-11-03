# Working on libstrix

Conventions for anyone changing this repo, human or agent. Read before editing
docs, recipes or results.

## Where things go

| file | holds | does not hold |
| --- | --- | --- |
| `README.md` | what libstrix is, requirements, build and install, Docker host setup and quick start, one paragraph on the plugin, 4-5 headline result bullets, tests, layout, license (~150 lines) | tables of recipes, kernels or results |
| `docs/docker.md` | the recipe table (weights, KV, tokens, 131k requests, images), overrides, per-recipe notes, memory guard, building the image | speed numbers |
| `docs/kernels.md` | the `STRIX_DISABLE` kernel table, the kernel registry, what needs the strix-vllm fork, hardware notes, adding a kernel | model results |
| `docs/performance.md` | setup and method, checkpoint notes, summary, speed, accuracy, per-model tables | recipe settings |
| `NOTES.md`, `docs/handoff.md` | local working notes (gitignored, never published) | |

New material goes into the file that owns its topic. Add a new `docs/*.md`
only for a topic none of them covers, and link it from the README.

## Writing

- Describe the code as it is now. No "before the fix", "earlier
  measurements", or notes about an image that is not published yet: the image
  is rebuilt from the strix-vllm fork when the code is pushed. Replace stale
  numbers; do not footnote them.
- No footnotes shared across tables. Caveats about a checkpoint go in
  "Checkpoint notes" in `docs/performance.md`; a footnote is only for one
  table and is numbered within it.
- Prefer a short table or bullets with a bold lead over long paragraphs.
  Plain words, numbers with units, no filler.
- Every claim with a number must come from a measurement in a log or results
  file. When moving or restructuring text, check afterwards that every number
  is still present (compare the number tokens of the old and new file).
- Links between docs use relative paths (`docker.md#memory-guard`) and must
  resolve after any heading change.

## Names

One name per checkpoint everywhere in the docs: `<family>-<size> <QUANT>`,
plain name for bf16.

| checkpoint | name in docs | recipe |
| --- | --- | --- |
| Qwen/Qwen3.6-35B-A3B | Qwen3.6-35B-A3B | `qwen3.6-35b-a3b` |
| Avesed/Qwen3.6-35B-A3B-INT8-W8A8 | Qwen3.6-35B-A3B W8A8 | `qwen3.6-35b-a3b-w8a8` |
| Qwen/Qwen3.6-35B-A3B-FP8 | Qwen3.6-35B-A3B FP8 | `qwen3.6-35b-a3b-fp8`, `-fp8-dflash2` |
| InternScience/Sage-A1.5-FP8 | Sage-A1.5 FP8 | `sage-a1.5-fp8` |
| openbmb/MiniCPM-V-4.7-35B-A3B-FP8 | MiniCPM-V-4.7 FP8 | `minicpm-v-4.7-35b-a3b-fp8` |
| Qwen/Qwen3.8-27B | Qwen3.8-27B | `qwen3.8-27b` |
| lued/Qwen3.8-27B-INT8-W8A16-MTP | Qwen3.8-27B INT8 | `qwen3.8-27b-int8` |
| amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 | Qwen3.8-27B AWQ INT4 | `qwen3.8-27b-awq-int4` |
| amd/Qwen3.8-27B-Quark-AWQ-MXFP4 | Qwen3.8-27B AWQ MXFP4 | `qwen3.8-27b-awq-mxfp4` |
| Qwen/Qwen3.8-27B-FP8 | Qwen3.8-27B FP8 | `qwen3.8-27b-fp8`, `-fp8-dflash2` |
| wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 | Qwen3.8-Flash-Next AWQ | `qwen3.8-flash-next` |
| LiquidAI/LFM2.5-24B-A2B | LFM2.5-24B-A2B | `lfm2.5-24b-a2b` |
| LiquidAI/LFM2.5-24B-A2B-FP8 | LFM2.5-24B-A2B FP8 | `lfm2.5-24b-a2b-fp8`, `-fp8-dspark` |
| RedHatAI/Muse-Glimmer-30B-FP8-block | Muse-Glimmer-30B FP8 | `muse-glimmer-30b-fp8`, `-fp8-dflash2` |
| HawkBearPig/flint-FP8-BF16 | Flint FP8 | `flint-fp8`, `-fp8-dflash2` |
| google/gemma-4-26B-A4B-it | gemma-4-26B-A4B | `gemma-4-26b-a4b` |
| cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit | gemma-4-26B-A4B AWQ | `gemma-4-26b-a4b-awq-int4` |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | gemma-4-26B-A4B FP8 | `gemma-4-26b-a4b-fp8` |
| amd/gemma-4-31B-it-MXFP4 | gemma-4-31B MXFP4 | `gemma-4-31b-mxfp4` |
| google/gemma-4-31B-it-qat-w4a16-ct | gemma-4-31B QAT W4A16 | `gemma-4-31b-qat-int4` |

Recipe files are `<model>-<size>[-<quant>][-dflash2|-dspark].yaml`, lowercase, with the
compose service named like the file. Quant suffixes come from a fixed set:
none (bf16), `fp8`, `int8` (W8A16), `w8a8`, `awq-int4`, `awq-mxfp4`, `mxfp4`,
`qat-int4`. Do not copy repo-name noise (`-it`, `-quark`, `-ct`, `-dynamic`,
`-block`, `-mtp`). The model folder under `/models/` keeps the checkpoint's own
name.

## Order

Every table, list and appendix keeps families together in this order, bf16
first and then the quantized variants in the order above:

1. Qwen3.6-35B-A3B (bf16, W8A8, FP8), then the models on its base: Sage-A1.5,
   MiniCPM-V-4.7
2. Qwen3.8-27B (bf16, INT8, AWQ INT4, AWQ MXFP4, FP8), then Qwen3.8-Flash-Next
3. LFM2.5-24B-A2B (bf16, FP8)
4. Muse-Glimmer-30B FP8
5. Flint FP8
6. gemma-4-26B-A4B (bf16, AWQ, FP8), then gemma-4-31B (MXFP4, QAT W4A16)

A `-dflash2` or `-dspark` recipe sits directly under its base recipe.

## Recipes

- Each recipe `extends` `common.yaml` and sets only what differs. Defaults:
  `GPU_UTIL` 0.85, `int8_per_token_head` KV, `TRITON_ATTN`, 131,072 context
  unless the model needs otherwise.
- Speculation: `MTP: ${MTP:-2}` where the checkpoint has an MTP head.
  DFlash 2 and DSpark recipes set `MTP: 0` and pass `--speculative-config`
  with `"num_speculative_tokens": ${SPEC_TOKENS:-N}`, N the measured best.
  `REPLAYSSM: ${REPLAYSSM:-1}` on every Qwen GDN recipe that speculates (the
  entrypoint adds `--use-replayssm` for MTP and DFlash alike). Flash-Next does
  not support ReplaySSM.
- Multimodal checkpoints serve images with
  `--mm-encoder-attn-backend=TRITON_ATTN`; use `--language-model-only` only
  when the vision tower does not load.
- Sampling defaults come from the model card (thinking mode) or the
  checkpoint's `generation_config.json`, as the last command line:
  `'--override-generation-config={...}'`. List the family in the sampling
  table in `docs/docker.md`. Measurements stay greedy: `Server.chat` sends
  neutral sampling unless `recipe_sampling=True`.
- A new or changed recipe gets its row in `docs/docker.md` from its own
  startup log at `GPU_UTIL=0.85`: weights (`Model loading took`), KV cache
  (`Available KV cache memory`), KV tokens (`GPU KV cache size`) and 131k
  requests (`Maximum concurrency`). The images column says ✓ only after a
  text and an image request both answered; otherwise `untested`,
  `text model` or `text only`.
- Validate with `docker compose -f <recipe> config -q` before committing.

## Testing a model

Tests run against the Docker image, the build users get, not a vLLM installed
on the host. Build the image under test first:

- Python-only changes (plugin `python/`, the strix-vllm fork's `vllm/*.py`,
  `docker/entrypoint.sh`): `STRIX_VLLM=/path/to/strix-vllm
  docker/build-test-image.sh` layers them on the published image as
  `strix-vllm:test` in seconds.
- Anything compiled (`csrc/`, CMake, vLLM C++/CUDA): a full
  `docker/build-and-push.sh` (without `PUSH=1`), then `IMAGE=<that tag>`.

`bench/container/` has the tests. Served tests take a target,
`recipe.yaml[:ENV=VAL,...]`, start it in their own compose project (port
18200) and stop it afterwards; offline scripts run inside the image through
`bench/container/run.sh`. All need `MODELS` set. One GPU job at a time: the
served tests refuse to start while another container holds the GPU.

A new checkpoint (or a change that touches one) goes through these steps in
order; stop and fix before moving on. "Stock" is the same image with
`STRIX_DISABLE=all`.

1. **Smoke.** `python bench/container/smoke.py <recipe>` starts it, checks a
   text answer, an image answer (multimodal), a parsed tool call and parsed
   reasoning (when the recipe sets those parsers), lists the kernel choices
   from the startup log (a layer left on vLLM's path is the first thing to
   fix) and prints the memory numbers for its `docs/docker.md` row.
2. **Greedy A/B**, plugin off and on:

   ```sh
   STRIX_DISABLE=all bench/container/run.sh python bench/model/e2e_check.py --model /models/M --out /work/base.json
   bench/container/run.sh python bench/model/e2e_check.py --model /models/M --out /work/strix.json --compare /work/base.json
   ```

3. **KLD at depth**, 2k and 32k, plus 128k where the context allows; set
   `STRIX_WIKITEXT` to a wikitext-2-raw directory:

   ```sh
   STRIX_DISABLE=all bench/container/run.sh python bench/model/kld_check.py --model /models/M --depth 32768 --out /work/ref.json
   bench/container/run.sh python bench/model/kld_check.py --model /models/M --depth 32768 --out /work/new.json --compare /work/ref.json
   ```

   When stock cannot load the checkpoint (per-channel FP8), the reference is
   an exact bf16 copy of the same weights; say so in "Checkpoint notes".
4. **GSM8K, paired.** n = 250, stock and plugin, compared per question with
   McNemar:

   ```sh
   python bench/container/gsm8k.py recipe.yaml:STRIX_DISABLE=all --out stock.json
   python bench/container/gsm8k.py recipe.yaml --out strix.json
   python bench/container/gsm8k.py --compare stock.json strix.json
   ```

5. **Speed by depth**, single stream, bf16 and int8 KV, plugin on and off:

   ```sh
   bench/container/run.sh python bench/model/profile_model.py --model /models/M \
     --combined 2048,8192,16384,32768,65536,131072 --decode-batches 1 \
     --decode-steps 128 --max-model-len 133120 \
     --kv-cache-dtype int8_per_token_head --gpu-memory-utilization 0.80
   ```

   256k with int8 KV only, at `--gpu-memory-utilization 0.72`. Stock deep
   points take hours; stop stock at 64k unless the deep numbers are needed.
6. **Batched decode**, 4 and 16 sequences after 2k and 8k of context: the
   same script with `--prefill "" --decode-batches 4,16 --decode-ctx
   2048,8192 --decode-long-only --decode-steps 64`.
7. **Speculative decoding** (MTP head or DFlash 2 draft): no speculation,
   MTP and DFlash 2 at a few draft-token counts, side by side:

   ```sh
   python bench/container/spec_equiv.py --baseline recipe.yaml:MTP=0 \
     recipe.yaml:MTP=2 recipe.yaml:MTP=3 recipe-dflash2.yaml:SPEC_TOKENS=3 recipe-dflash2.yaml
   ```

   Each configuration reports output tok/s at 1 request and at 8 at once,
   mean tokens accepted per step, and the KV cache it leaves (tokens and GiB:
   a draft model takes some). It passes when every output matches the
   no-draft baseline except at the baseline's near-ties. The fastest passing
   setting becomes the recipe default; the table goes into "Speculative
   decoding" in `docs/performance.md`, with the KV tokens per configuration.
8. **KV under pressure**, for any recipe with a draft model and after any
   change to KV cache planning: `python bench/container/kv_pressure.py
   <recipe>` runs each prompt alone, then all at once on a cache sized to
   80 % of what they need together (50 % and 30 % if that does not
   preempt), and fails if a record copied right alone comes back with
   wrong contents under pressure, or if no preemption happened at all
   (lower `--fill` then).

Pass bar: plugin-on KLD at the stock-vs-stock noise floor (0.001-0.006 on the
Qwen checkpoints, 97-99 % top-1) unless the plugin changes the arithmetic on
purpose (explain it in "Reading it"); GSM8K McNemar p > 0.05 against stock;
no depth or batch slower than stock; spec_equiv and kv_pressure PASS.

Then record it: rows in every `docs/performance.md` table it was measured
in, in family order (see Order); its caveats in "Checkpoint notes"; a
per-model table in the appendix; the headline in the README only if it moves
a README bullet.

Before pushing an image, `python bench/container/smoke.py` (every recipe) must
pass on it.

## Measuring

- One GPU job at a time; GPU memory is host RAM, and an overcommitted run
  takes the machine down. The container's memory guard covers served tests;
  use `bench/run_guarded.sh` for anything run on the host.
- Run `python -m pytest tests -q` after any kernel change.

## Git

Commit and push only when asked. The Docker image is built from the
strix-vllm fork and this repo at push time, so docs describe what that build
will contain.
