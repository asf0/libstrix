# Changelog

One entry per release, newest first, covering both libstrix and the vLLM fork
the image is built with ([asf0/strix-vllm](https://github.com/asf0/strix-vllm)).
Versions follow [semantic versioning](https://semver.org): the minor version
for new models and features, the patch version for fixes and tuning; 0.x
until the API is called stable.

Changes land under **Unreleased**. To release: rename that heading to
`vX.Y.Z (date): summary`, set `version` in `pyproject.toml`, pin the fork
commit in `docker/strix-vllm.ref`, commit, and push the tag `vX.Y.Z`. The
image workflow then publishes `ghcr.io/asf0/strix-vllm:X.Y.Z` (and `:latest`)
and a GitHub release with this entry and the two commits it was built from.

## Unreleased

Recipes:

- Every recipe except `lfm2.5-24b-a2b` speculates with a draft by default:
  the incoai DFlash 2 drafts (Qwen3.6/3.8, Flint, Flash-Next), the
  meta-models Muse-Glimmer assistant, the z-lab gemma-4 DFlash drafts and the
  DSpark drafts (LFM2.5 FP8, MiniCPM-V-4.7); `MTP=<n>` still selects a
  checkpoint's own MTP head.
- Removed the draftless duplicates: `qwen3.6-35b-a3b-fp8`, `qwen3.8-27b-fp8`,
  `qwen3.8-27b-uncensored-fp8`, `qwen3.8-flash-next`, `muse-glimmer-30b-fp8`,
  `lfm2.5-24b-a2b-fp8`; `flint-fp8-dflash2` is now `flint-fp8`. The NPU
  overlay `npu/qwen3.8-flash-next.yaml` went with `qwen3.8-flash-next`.
- New `inkling-2-mini`: Inkling 2 Mini with its DSpark draft, 131k context
  (smoke: text, image, tool call and reasoning pass).

## v0.1.0 (2026-11-23): Inkling 2 Mini, fork on upstream main

Fork commit: `5c214f256f` (branch `v0.1.x`).

### Fork (asf0/strix-vllm, branch `v0.1.x`)

- Follows upstream vLLM `main` instead of the v0.34.0 release; the image
  reports `0.35.0.dev0+strix.g<fork commit>`. The `gfx1151` branch (v0.34.0
  based) is unchanged.
- KV-cache subpools are gone: hybrid KV blocks are sized for the speculative
  drafter's page when the lcm with the target's would be over twice it, so GDN
  pages pay ~1.5% padding instead of 2x.
- Inkling 2 Small (upstream PR #67685) and Inkling 2 Mini (the looped model)
  on ROCm and gfx1151: NVFP4/MXFP8 experts without FP8 hardware, gfx1151
  attention launch shapes, DSpark drafting only the steps its batch-size
  schedule verifies, and hooks for libstrix's kernels.
- Adaptive verification profiles verify costs with routed experts.

### libstrix

Added:

- Inkling 2 Mini (`python/strix_vllm/inkling2*.py`; switches `inkling2_moe`,
  `inkling2_loop`, `inkling2_attention`, `inkling2_kv_conv`, `inkling2_linear`,
  `inkling2_draft_head`, see [docs/kernels.md](docs/kernels.md)):
  - MoE: fp16-WMMA NVFP4 / MXFP8 grouped GEMMs for prefill (derived from CIRU
    AI's kernel, see NOTICE) with the softcapped SwiGLU and routing weights in
    the epilogue, and a HIP NVFP4 GEMV for decode-sized tiles.
  - Attention: `inkling_rel_attn_prefill`, prefill relative attention with the
    output gate and per-token sink in its epilogue; window layers compute
    their relative logits in the kernel instead of reading a 315 MB table.
  - `inkling2_kv_conv`: the attention prologue's K/V short convolutions.
  - Loop message controllers for decode and prefill; int4 / int8 decode
    copies of the attention projections and the DSpark draft head.
- Recipe `bench-inkling-2-mini` (DSpark, 2 drafts for a single request).
- Tests: `tests/test_inkling_rel_attn.py`,
  `bench/container/test_inkling2_conv.py`,
  `bench/container/test_inkling2_nvfp4_bmm.py`.

Changed:

- `attn_prefill_kernel` takes the query tiles per block (`QT`) and Inkling's
  relative-attention mode as template parameters; one-wave heads no longer
  allocate the cross-wave score buffer (same outputs on every existing
  config).
- `dense_w8`: fp32-head logits read the int8 LM head copy in decode; on for
  Inkling 2 as well as Qwen3.8-Flash-Next and Flint.
- Image: `xgrammar` 0.2.9 (vLLM main pins it); the QSA indexer wrapper
  forwards vLLM's extra kwargs; the Flash-Next DFlash 2 draft runs on
  TRITON_ATTN.

Inkling 2 Mini on this image: prefill 2,674 / 2,688 / 2,570 tok/s at 2.7k /
10.8k / 21.5k tokens; decode 47 tok/s at one request (56 with DSpark), 94 /
111 / 155 tok/s total at 4 / 8 / 16; GSM8K 213-216 of 250.

## Before v0.1.0

Images were built from the fork's `gfx1151` branch (vLLM v0.34.0 plus the
fork's commits) and libstrix `main` up to `a83589d`, without a changelog; see
the git history of both repositories.
