# Performance

libstrix against stock upstream vLLM on one gfx1151 APU (Strix Halo, 128 GB
unified memory). Your ROCm and vLLM versions will move these numbers; the
ratios are the point.

## Setup and method

- vLLM: upstream `899a1689e2`, unpatched.
- torch 2.14.0+rocm10.1.0, HIP 7.16.
- "stock" is vLLM with the plugin disabled, using its own kernels (Triton
  attention, Triton and `fused_moe` GEMMs). "strix" is the plugin on.
- One request per depth in a fresh engine per config: prompt of exactly that
  many tokens, then 128 decode steps (64 at 256k). Prefill throughput is
  depth / TTFT; decode is 1000 / ms per step. Depths below 64k run twice and
  report the second.
- `--max-num-seqs 1`, prefix caching off, prefill chunk 8,192 tokens,
  `--gpu-memory-utilization 0.80` (0.72 at 256k). The setting does not move
  single-stream speed: Qwen3.6-35B-A3B FP8 at 0.80 and 0.85 (int8 KV, same
  session) measured 0.578 / 0.577 s prefill and 21.20 / 21.18 ms decode at 2k,
  2.094 / 2.095 s and 21.71 / 21.69 ms at 8k.
- Batched: N sequences decode together after a 2k or 8k context, 64 steps.
- Qwen3.8-27B FP8: its int8-KV depths 2k-128k and its 256k point are
  separate runs; a first int8-KV pass measured 7-15% slow on one session and
  was repeated.
- Stock is measured to 64k on every model and at 128k and 256k on a
  subset: a single stock request at those depths takes 30 minutes to 4
  hours.

## Checkpoint notes

What differs per checkpoint, for every table below:

- **LFM2.5-24B-A2B, LFM2.5-24B-A2B FP8, Muse-Glimmer-30B FP8** cap context at
  131,072; their deepest point is 130,048.
- **LFM2.5-24B-A2B FP8, Muse-Glimmer-30B FP8** are compressed-tensors block
  FP8; stock speed was not measured.
- **Qwen3.6-35B-A3B FP8** is block-scaled FP8 (128x128), dense and expert
  GEMMs. Stock was measured at 2k and 8k only: 341 / 309 tok/s prefill, 4.4
  tok/s decode.
- **Qwen3.8-27B FP8** is block-scaled FP8. Stock decodes it at 0.57 tok/s
  (untuned Triton FP8 GEMM, one short-context run); no other stock points
  were run.
- **Sage-A1.5 FP8, MiniCPM-V-4.7 FP8, gemma-4-26B-A4B FP8** are
  compressed-tensors per-channel FP8. Stock vLLM has no FP8 kernel for this
  format on gfx1151 and fails at start-up, so there is no stock speed; for KLD
  the stock side is an exact bf16 copy of the same weights. MiniCPM-V-4.7 is
  measured text only. gemma-4-26B-A4B FP8 is scored against the plugin on that
  bf16 copy, so only the FP8 path differs; it runs the bf16-activation FP8
  kernels (its activations overflow fp16, and the fp16 raw kernels put it at
  KLD 0.23).
- **gemma-4-31B MXFP4**: stock runs MXFP4 as fp4 × fp4 emulation on this GPU,
  strix as fp4 weights × bf16 activations, so the ratio is against a slow path.
- **Qwen3.8-Flash-Next AWQ** is measured differently: on the strix-vllm fork's
  Docker image, 16k prefill chunks, the n-gram table read from disk (prefill
  is the fastest run per depth: a request whose rows are not cached is
  slower), and a baseline that keeps the libstrix GDN conv. See
  [its per-model table](#qwen38-flash-next-awq). Its int8 KV (the strix-vllm
  fork's int8 QSA layers, the served setting) scores KLD 0.0207 against bf16
  KV at 2k and 0.0083 at 32k. At 2k any numeric change costs about that much
  (K alone 0.0204, V alone 0.0197, V on an 11-bit grid 0.0215; two bf16 runs
  are bit-identical); at 32k two bf16 runs already differ by 0.0075.
- **Flint FP8** (block FP8 experts, bf16 elsewhere) needs the strix-vllm
  fork, which runs it on ROCm. It is measured on the Docker image at
  `--gpu-memory-utilization 0.85` at every depth, 256k included; stock to 64k.
  Its experts run the FP8 kernel with a bf16 intermediate (the fused
  silu(gate) * up reaches 90,112, past the fp16 range).
- **LFM2.5-24B-A2B FP8** greedy output depends on the batch shape even
  without speculation: the same 8 prompts alone and 8 at once diverge at
  top-1 / top-2 gaps up to 1.375 nats (its logprobs move in bf16 steps of
  0.125). Its speculative-decoding check therefore accepts divergences at
  gaps up to 1.5 instead of 1.0.

## Summary

Single stream, `int8_per_token_head` KV (the serving dtype). Speedup is
strix over stock; the last two columns are strix's own throughput.

| model | 2k prefill | 64k prefill | 2k decode | 64k decode | strix 128k prefill / decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 2.70× | 19.3× | 1.86× | 2.35× | 1,751 / 26.4 |
| Qwen3.6-35B-A3B W8A8 | 2.14× | 18.3× | 1.03× | 1.53× | 1,697 / 24.5 |
| Qwen3.6-35B-A3B FP8 | 10.4× | - | 10.8× | - | 1,688 / 34.8 |
| Sage-A1.5 FP8 | - | - | - | - | 1,708 / 30.6 |
| MiniCPM-V-4.7 FP8 | - | - | - | - | 1,716 / 30.5 |
| Qwen3.8-27B | 1.67× | 14.3× | 1.01× | 1.14× | 435 / 3.9 |
| Qwen3.8-27B INT8 | 2.56× | 14.6× | 2.38× | 2.33× | 430 / 5.7 |
| Qwen3.8-27B AWQ INT4 | 2.05× | 14.2× | 1.01× | 1.22× | 428 / 8.1 |
| Qwen3.8-27B FP8 | - | - | 13.5× | - | 434 / 6.4 |
| Qwen3.8-Flash-Next AWQ | 1.93× | 2.37× | 1.52× | 1.72× | 1,518 / 24.2 |
| LFM2.5-24B-A2B | 2.02× | 11.4× | 2.15× | 2.18× | 2,405 / 27.1 |
| LFM2.5-24B-A2B FP8 | - | - | - | - | 2,244 / 33.6 |
| Muse-Glimmer-30B FP8 | - | - | - | - | 76 / 6.5 |
| Flint FP8 | 8.78× | 16.7× | 4.58× | 4.44× | 1,344 / 39.7 |
| gemma-4-26B-A4B | 4.35× | 31.1× | 1.28× | 3.33× | 1,588 / 21.5 |
| gemma-4-26B-A4B AWQ | 3.87× | 28.2× | 1.36× | 4.28× | 1,499 / 29.6 |
| gemma-4-26B-A4B FP8 | - | - | - | - | 1,520 / 29.8 |
| gemma-4-31B MXFP4 | 2.67× | 21.0× | 6.21× | 6.99× | 296 / 7.3 |
| gemma-4-31B QAT W4A16 | 2.99× | 22.5× | 1.32× | 2.99× | 292 / 8.0 |

What it means:

- **Prefill** is where the plugin matters most. Stock prefill time grows
  about 4× per doubling of context on this APU; strix's about 3×. At
  256k the 35B takes 241 s with strix and 7,568 s stock (31×).
- **Decode** doubles on the 35B, LFM2.5 and the 27B INT8 checkpoint, and on
  every gemma at depth. On the 27B bf16, the 27B AWQ INT4 and the 35B W8A8,
  decode is at parity with stock: nothing the plugin replaces dominates their
  decode step.
- **FP8** decodes from the one-byte weights (W8A16). The 27B FP8 goes from
  0.57 to 7.7 tok/s and matches the INT8 27B at every depth. The FP8 MoE models
  decode 1.2-1.5× faster than their bf16 versions (35B 47 vs 32.8 tok/s at 2k),
  with prefill 1-14 % behind bf16 (the FP8 weights are decoded to bf16 for the
  large prefill GEMMs; gemma-4 also runs the slower bf16-activation kernels).
- **No configuration is slower than stock**, single stream or batched
  (lowest: 0.98× at x16 on the 35B W8A8, inside run-to-run noise), except
  Qwen3.8-Flash-Next's short prompts: when their n-gram rows come from disk,
  2k prefill runs below stock (median 541 against 716 tok/s on the Docker
  image; the fastest run, 1,387).

## Speed

### Depth

Prefill and decode tok/s, single stream, `--gpu-memory-utilization 0.80`
(0.72 at 256k, which leaves the host 10 GB of RAM at that depth; GPU memory is
host RAM here). The memory setting sizes the KV cache only; it does not change
these single-stream numbers (see [Setup and method](#setup-and-method)). The KV-capacity table in
[docker.md](docker.md) uses 0.85, the Docker recipes' setting.

int8 KV (`int8_per_token_head`, the serving dtype):

| model | prefill 2k / 32k / 128k / 256k | decode 2k / 32k / 128k / 256k |
| --- | --- | --- |
| Qwen3.6-35B-A3B | 3,589 / 3,331 / 1,751 / 1,085 | 32.8 / 31.0 / 26.4 / 22.1 |
| Qwen3.6-35B-A3B W8A8 | 3,221 / 3,133 / 1,697 / 1,059 | 29.7 / 28.4 / 24.5 / 20.9 |
| Qwen3.6-35B-A3B FP8 | 3,543 / 3,102 / 1,688 / 1,052 | 47.2 / 43.2 / 34.8 / 27.7 |
| Sage-A1.5 FP8 | 3,750 / 3,189 / 1,708 / 1,066 | 40.0 / 37.1 / 30.6 / 25.0 |
| MiniCPM-V-4.7 FP8 | 3,795 / 3,196 / 1,716 / 1,067 | 39.6 / 36.9 / 30.5 / 24.8 |
| Qwen3.8-27B | 709 / 611 / 435 / - | 4.4 / 4.3 / 3.9 / - |
| Qwen3.8-27B INT8 | 692 / 603 / 430 / 312 | 7.8 / 7.2 / 5.7 / 5.5 |
| Qwen3.8-27B AWQ INT4 | 688 / 598 / 428 / 312 | 13.0 / 11.4 / 8.1 / 5.8 |
| Qwen3.8-27B FP8 | 658 / 610 / 434 / 318 | 7.7 / 7.4 / 6.4 / 5.5 |
| Qwen3.8-Flash-Next AWQ ¹ | 1,387 / 1,708 / 1,518 / 1,423 | 24.4 / 24.4 / 24.2 / 26.9 |
| LFM2.5-24B-A2B | 5,124 / 4,613 / 2,405 / - | 38.3 / 34.7 / 27.1 / - |
| LFM2.5-24B-A2B FP8 | 4,948 / 4,052 / 2,244 / - | 52.5 / 46.3 / 33.6 / - |
| Muse-Glimmer-30B FP8 | 499 / 220 / 76 / - | 7.4 / 6.9 / 6.5 / - |
| Flint FP8 | 4,137 / 3,001 / 1,344 / 778 | 63.0 / 54.6 / 39.7 / 28.5 |
| gemma-4-26B-A4B | 3,877 / 3,071 / 1,588 / 971 | 25.9 / 24.7 / 21.5 / 18.5 |
| gemma-4-26B-A4B AWQ | 3,358 / 2,769 / 1,499 / 934 | 38.2 / 35.6 / 29.6 / 24.1 |
| gemma-4-26B-A4B FP8 | 3,327 / 2,802 / 1,520 / 945 | 38.7 / 36.1 / 29.8 / 24.4 |
| gemma-4-31B MXFP4 | 542 / 462 / 296 / 200 | 9.7 / 8.9 / 7.3 / 5.9 |
| gemma-4-31B QAT W4A16 | 535 / 452 / 292 / 198 | 10.8 / 10.0 / 8.1 / 6.4 |

¹ The Docker image. Prefill is the fastest run at each depth: its n-gram table
is read from disk, and a request that waits on it prefills more slowly, down
to about 540 tok/s at 2k (see [its per-model table](#qwen38-flash-next-awq)).
Decode is the median of every run. bf16 KV on the same image: 1,515 / 1,695 /
1,555 at 2k / 32k / 128k.

bf16 KV (256k does not fit with bf16 KV):

| model | prefill 2k / 32k / 128k | decode 2k / 32k / 128k |
| --- | --- | --- |
| Qwen3.6-35B-A3B | 3,581 / 3,393 / 1,799 | 32.6 / 29.5 / 22.8 |
| Qwen3.6-35B-A3B W8A8 | 3,214 / 3,173 / 1,733 | 29.3 / 27.1 / 21.4 |
| Qwen3.6-35B-A3B FP8 | 3,531 / 3,137 / 1,722 | 46.6 / 40.5 / 28.8 |
| Sage-A1.5 FP8 | 3,738 / 3,236 / 1,749 | 39.6 / 35.2 / 25.9 |
| MiniCPM-V-4.7 FP8 | 3,785 / 3,247 / 1,754 | 39.2 / 34.8 / 25.8 |
| Qwen3.8-27B | 677 / 604 / 406 | 4.4 / 4.2 / 3.7 |
| Qwen3.8-27B INT8 | 694 / 608 / 413 | 7.7 / 6.9 / 5.0 |
| Qwen3.8-27B AWQ INT4 | 690 / 604 / 408 | 12.9 / 10.6 / 6.6 |
| Qwen3.8-27B FP8 | 664 / 638 / 421 | 7.7 / 7.1 / 5.8 |
| Qwen3.8-Flash-Next AWQ | 1,515 / 1,695 / 1,555 | 25.4 / 25.3 / 25.3 |
| LFM2.5-24B-A2B | 5,007 / 4,593 / 2,404 | 38.3 / 34.7 / 26.8 |
| LFM2.5-24B-A2B FP8 | 4,972 / 4,149 / 2,261 | 52.7 / 46.0 / 33.0 |
| Muse-Glimmer-30B FP8 | 580 / 328 / 136 | 7.5 / 7.1 / 6.7 |
| Flint FP8 | 4,163 / 3,121 / 1,437 | 61.2 / 47.5 / 28.2 |
| gemma-4-26B-A4B | 3,591 / 3,063 / 1,485 | 24.6 / 23.2 / 18.5 |
| gemma-4-26B-A4B AWQ | 3,395 / 2,864 / 1,554 | 37.2 / 33.2 / 24.6 |
| gemma-4-26B-A4B FP8 | 3,378 / 2,968 / 1,581 | 37.7 / 33.6 / 24.8 |
| gemma-4-31B MXFP4 | 492 / 469 / 303 | 9.4 / 8.3 / 6.1 |
| gemma-4-31B QAT W4A16 | 542 / 460 / 299 | 10.4 / 9.1 / 6.6 |

Decode here is without speculation; the MTP heads are not used in these runs.

Sage-A1.5 decodes slower than the 35B FP8 (40 vs 47 tok/s at 2k) because its
checkpoint keeps the linear-attention projections in bf16. Muse-Glimmer's
prefill falls off with depth because of its attention, not the FP8 linears.

### Batched decode

Total decode tok/s over the batch, stock → **strix** (speedup), at 2k and 8k
context. int8 KV for every model; bf16 KV too for the 35B checkpoints, whose
decode-attention routing depends on batch size (strix takes decode up to 2
sequences on head 256 / GQA 8, stock Triton above that).

| model | KV | x4 @2k | x4 @8k | x16 @2k | x16 @8k |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | bf16 | 41 → **93** (2.25×) | 38 → **79** (2.12×) | 97 → **164** (1.68×) | 83 → **141** (1.69×) |
| Qwen3.6-35B-A3B | int8 | 39 → **97** (2.51×) | 37 → **82** (2.20×) | 92 → **167** (1.82×) | 84 → **152** (1.82×) |
| Qwen3.6-35B-A3B W8A8 | bf16 | 80 → **80** (1.00×) | 68 → **70** (1.04×) | 170 → **167** (0.98×) | 146 → **153** (1.05×) |
| Qwen3.6-35B-A3B W8A8 | int8 | 78 → **84** (1.08×) | 72 → **78** (1.08×) | 183 → **180** (0.98×) | 153 → **167** (1.09×) |
| Qwen3.6-35B-A3B FP8 | bf16 | - → **103** (-) | - → **100** (-) | - → **243** (-) | - → **193** (-) |
| Qwen3.6-35B-A3B FP8 | int8 | - → **107** (-) | - → **107** (-) | - → **255** (-) | - → **213** (-) |
| Sage-A1.5 FP8 | int8 | - → **122** (-) | - → **101** (-) | - → **231** (-) | - → **215** (-) |
| MiniCPM-V-4.7 FP8 | int8 | - → **101** (-) | - → **101** (-) | - → **294** (-) | - → **206** (-) |
| Qwen3.8-27B | int8 | 17 → **17** (1.02×) | 16 → **17** (1.06×) | 57 → **60** (1.06×) | 49 → **58** (1.17×) |
| Qwen3.8-27B INT8 | int8 | 13 → **29** (2.30×) | 12 → **28** (2.30×) | 46 → **96** (2.09×) | 40 → **84** (2.08×) |
| Qwen3.8-27B AWQ INT4 | int8 | 47 → **50** (1.05×) | 41 → **45** (1.11×) | 100 → **122** (1.21×) | 79 → **104** (1.31×) |
| Qwen3.8-27B FP8 | int8 | - → **29** (-) | - → **28** (-) | - → **100** (-) | - → **93** (-) |
| Qwen3.8-Flash-Next AWQ | bf16 | 37 → **44** (1.18×) | 32 → **40** (1.26×) | 81 → **96** (1.18×) | 78 → **90** (1.16×) |
| LFM2.5-24B-A2B | int8 | 36 → **78** (2.17×) | 34 → **74** (2.15×) | 111 → **181** (1.63×) | 90 → **153** (1.71×) |
| LFM2.5-24B-A2B FP8 | int8 | - → **127** (-) | - → **108** (-) | - → **291** (-) | - → **255** (-) |
| Muse-Glimmer-30B FP8 | int8 | - → **30** (-) | - → **29** (-) | - → **112** (-) | - → **109** (-) |
| gemma-4-26B-A4B | int8 | 40 → **67** (1.67×) | 33 → **67** (2.02×) | 86 → **158** (1.85×) | 74 → **175** (2.36×) |
| gemma-4-26B-A4B AWQ | int8 | 56 → **99** (1.77×) | 43 → **86** (1.99×) | 100 → **230** (2.29×) | 78 → **225** (2.89×) |
| gemma-4-26B-A4B FP8 | int8 | - → **102** (-) | - → **103** (-) | - → **254** (-) | - → **263** (-) |
| gemma-4-31B MXFP4 | int8 | 6 → **29** (5.19×) | 5 → **28** (5.46×) | 14 → **92** (6.43×) | 12 → **83** (6.81×) |
| gemma-4-31B QAT W4A16 | int8 | 21 → **39** (1.84×) | 15 → **37** (2.38×) | 22 → **104** (4.72×) | 17 → **93** (5.45×) |

On Qwen3.8-Flash-Next, `dense_w8` (int8 decode copies) applies only up to 8
rows (5 on small weights), and decode attention (QSA) and the GDN recurrence
stay on vLLM's kernels without MTP. The batched gain is therefore small, unlike the
single-stream decode gain (1.5-1.7×).

### Serving

Qwen3.8-27B INT8 W8A16 with MTP k=2, int8 KV, 131k context, 40 concurrent
sequences, gmu 0.7, driven by `vllm bench serve` with 80 requests of 2,048
tokens in / 1,024 out.

| metric | 27B INT8 W8A16 + MTP | 27B FP8 + MTP | 35B-A3B FP8 + MTP | Sage-A1.5 FP8 + MTP |
| --- | ---: | ---: | ---: | ---: |
| output throughput | 123.3 tok/s | 115.4 tok/s | 300.1 tok/s | 227.3 tok/s |
| total throughput | 369.9 tok/s | 346.1 tok/s | 900.2 tok/s | 681.8 tok/s |
| TPOT mean / median / p99 | 240 / 228 / 552 ms | 259 / 238 / 561 ms | 109 / 106 / 176 ms | 143 / 133 / 248 ms |
| TTFT median | 13.8 s | 10.8 s | 2.3 s | 2.1 s |

p99 TPOT is the slowest one or two of 80 requests and varies between runs
(27B FP8: 462-561 ms over four runs, mean within 3 %). Sage trails the 35B FP8
for the reason above and a lower MTP acceptance on random prompts (1.62 vs 2.25
tokens per step).

FP8 MoE checkpoints need `VLLM_ROCM_USE_AITER_MOE=0` when the server script
exports `VLLM_ROCM_USE_AITER=1`: vLLM then demands the AITER FP8 MoE backend,
which does not support this GPU, and the engine fails at start-up.

### Speculative decoding

FP8 checkpoints with the plugin, on 8 real prompts (4 GSM8K, plus code,
explanation and summary), greedy, up to 512 tokens. Each cell is output tok/s
at 1 request / 8 at once, with the mean tokens accepted per step in brackets.
MTP uses the checkpoint's own MTP head; DFlash 2 is inco.ai's block-diffusion
draft model (`-dflash2` recipes) and DSpark Liquid AI's draft model for LFM2.5
(`-dspark`). "–" is a method the checkpoint has no head or draft for.

| model | none | MTP 2 | MTP 3 | DFlash 2 | DSpark |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 45.7 / 127.4 | 63.5 / 167.6 (2.57) | 62.7 / 188.1 (3.15) | **85.1 / 188.5** (4.45), 7 tokens | – |
| Qwen3.8-27B | 7.7 / 46.7 | 16.4 / 90.8 (2.65) | 18.5 / 98.4 (3.26) | **29.6 / 132.9** (4.90), 7 tokens | – |
| LFM2.5-24B-A2B | 51.7 / 139.3 | – | – | – | **66.7** / 177.7 (2.94), 4 tokens; 63.9 / **180.2** (3.29), 6 tokens |
| Muse-Glimmer-30B | 7.5 / 58.5 | – | – | **24.1** / 80.8 (4.50), 15 tokens; 21.6 / **107.3** (3.94), 7 tokens | – |
| Flint | 62.7 / 144.4 | – | – | **80.6** / 144.9 (2.72), 3 tokens; 76.6 / **145.5** (2.27), 2 tokens | – |

KV cache each configuration leaves (tokens, `GPU_UTIL=0.85`, recipe defaults;
the draft model's own cache comes out of the same memory):

| model | none | MTP 2 | MTP 3 | DFlash 2 | DSpark |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 5,684,779 | 4,736,198 | 4,710,648 | 3,923,703 (7 tokens) | – |
| Qwen3.8-27B | 2,022,253 | 1,864,797 | 1,870,062 | 1,769,472 (7 tokens) | – |
| LFM2.5-24B-A2B | 7,423,011 | – | – | – | 4,853,614 (4 tokens) |
| Muse-Glimmer-30B | 7,046,883 | – | – | 5,380,274 (15 tokens) | – |
| Flint | 2,981,019 | – | – | 2,678,872 (3 tokens) | – |

- **Draft tokens:** 7 beats 3 on the Qwen models (35B 75.4 / 187.4, 27B 20.4
  / 106.9). Muse is fastest at 15 for one request and at 7 for several
  (`SPEC_TOKENS=7`). LFM2.5 is fastest at 4 for one request and at 6 for
  several; 9 (the draft's block) gives 59.8 / 173.6 (3.59).
- **Flint** gains least: a longer verify batch reads more of its 256 experts
  per step than the extra accepted tokens pay for. 5 tokens gives 64.2 / 140.3
  (3.10), 7 gives 58.5 / 124.0 (3.23).
- **Accuracy with the draft:** GSM8K (n = 250, 768 tokens, one request at a
  time) without / with the draft: Flint 196 / 200 (3 tokens, McNemar
  p = 0.57), LFM2.5 197 / 198 (4 tokens, p = 1.0).
- **KV under pressure** (`kv_pressure.py`, 12 requests copying back 140
  records on a cache too small for all of them): no record copied right
  alone comes back wrong under pressure. Qwen3.6-35B-A3B FP8 + DFlash 2
  (1 preemption) and + MTP 2 with ReplaySSM (58), Flint + DFlash 2 (1, at
  30 % of the tokens needed), LFM2.5 + DSpark (9). Qwen3.8-Flash-Next is not
  covered: with a cache that small it runs one request at a time, so nothing
  is preempted.
- **Qwen3.8-Flash-Next AWQ** (needs the strix-vllm fork: it records the mean
  of the four hyper-connection streams after layers 5/15/25/35/45 for the
  draft) on the same 8 prompts with 256-token answers, int8 KV, output tok/s
  at 1 request / 8 at once (tokens accepted per step) and the KV it leaves:

  | configuration | tok/s | KV tokens |
  | --- | ---: | ---: |
  | none | 24.9-25.2 / 54.5-60.0 | 1,391,594 |
  | MTP 2 | 41.1 / 70.6 (2.60) | 605,314 |
  | DFlash 2, 4 tokens | **48.8 / 81.8** (3.64) | 627,020 |
  | DFlash 2, 5 tokens | 48.8 / 79.4 (3.92) | 586,463 |
  | DFlash 2, 6 tokens | 47.4 / 77.7 (4.12) | 527,484 |
  | DFlash 2, 7 tokens | 45.1 / 78.4 (4.25) | 530,456 |

  4 tokens is the recipe default. Each token checked reads up to 10 more of
  the 512 experts, so longer drafts stop paying past 4-5 although they
  accept more. MTP 2 and DFlash 7 diverge from the no-draft output at one
  confident position (baseline margin 2.6, same prompt and token); MTP 2 on
  the previous image (v0.32) does the same, so it is not the draft.

## Accuracy

### KLD against stock

Each pair scores a 512-token teacher-forced tail of wikitext after
2k, 32k or 128k tokens of context and stores the top-20 logprobs at every
scored position, once stock and once with the plugin. KLD is the mean
per-position KL(stock ‖ strix) over the top-20; top-1 is how often both pick
the same argmax. PPL deltas are only comparable within a row.

| model | depth | top-1 | PPL stock → strix | KLD mean | KLD p95 |
| --- | ---: | ---: | --- | ---: | ---: |
| Qwen3.6-35B-A3B | 2k | 97.1% | 3.688 → 3.694 | 0.0049 | 0.0200 |
| | 32k | 97.0% | 3.306 → 3.304 | 0.0038 | 0.0111 |
| | 128k | 97.9% | 2.683 → 2.698 | 0.0027 | 0.0078 |
| Qwen3.6-35B-A3B FP8 | 2k | 97.0% | 3.648 → 3.671 | 0.0080 | 0.030 |
| | 32k | 97.0% | 3.324 → 3.331 | 0.0060 | 0.018 |
| Sage-A1.5 FP8 | 2k | 96.9% | 3.869 → 3.882 | 0.0049 | 0.015 |
| | 32k | 97.3% | 3.401 → 3.408 | 0.0038 | 0.013 |
| MiniCPM-V-4.7 FP8 | 2k | 97.6% | 3.720 → 3.739 | 0.0052 | 0.022 |
| | 32k | 97.8% | 3.420 → 3.435 | 0.0025 | 0.0095 |
| Qwen3.8-27B | 2k | 99.0% | 3.798 → 3.800 | 0.0008 | 0.0029 |
| | 32k | 99.3% | 3.360 → 3.376 | 0.0036 | 0.0025 |
| Qwen3.8-27B INT8 | 2k | 98.5% | 3.813 → 3.832 | 0.0022 | 0.0073 |
| | 32k | 97.6% | 3.375 → 3.394 | 0.0018 | 0.0063 |
| | 128k | 98.3% | 2.673 → 2.673 | 0.0022 | 0.0068 |
| Qwen3.8-27B AWQ INT4 | 2k | 98.3% | 3.865 → 3.874 | 0.0065 | 0.0067 |
| | 32k | 98.2% | 3.469 → 3.476 | 0.0015 | 0.0052 |
| Qwen3.8-27B FP8 | 2k | 97.4% | 3.807 → 3.806 | 0.0032 | 0.0092 |
| Qwen3.8-Flash-Next AWQ | 2k | 96.9% | 1.670 → 1.670 | 0.019 | 0.095 |
| | 32k | 96.8% | 3.056 → 3.074 | 0.0091 | 0.037 |
| | 128k | 98.1% | 2.442 → 2.443 | 0.0078 | 0.029 |
| LFM2.5-24B-A2B | 2k | 94.2% | 6.488 → 6.509 | 0.034 | 0.094 |
| | 32k | 95.5% | 5.711 → 5.798 | 0.023 | 0.082 |
| LFM2.5-24B-A2B FP8 | 2k | 90.2% | 6.404 → 6.687 | 0.077 | 0.25 |
| | 32k | 91.1% | 5.867 → 6.025 | 0.080 | 0.20 |
| Muse-Glimmer-30B FP8 | 2k | 98.3% | 3.894 → 3.893 | 0.0030 | 0.0096 |
| | 32k | 98.3% | 3.324 → 3.329 | 0.0023 | 0.0079 |
| gemma-4-26B-A4B | 2k | 85.5% | 16.33 → 16.50 | 0.28 | 1.28 |
| | 32k | 88.4% | 51.95 → 55.06 | 0.22 | 1.00 |
| | 128k | 88.5% | 28.95 → 33.80 | 0.20 | 0.98 |
| gemma-4-26B-A4B AWQ | 2k | 82.4% | 16.73 → 19.73 | 0.34 | 1.83 |
| | 32k | 88.3% | 19.59 → 22.02 | 0.23 | 0.89 |
| | 128k | 83.3% | 7.462 → 7.725 | 0.26 | 1.35 |
| gemma-4-26B-A4B FP8 | 2k | 100% | 14.34 → 14.34 | 0.0000 | 0.0000 |
| | 32k | 90.6% | 42.97 → 43.00 | 0.17 | 0.68 |
| gemma-4-31B MXFP4 | 2k | 59.7% | 13.44 → 13.29 | 1.54 | 5.58 |
| | 32k | 53.1% | 51.81 → 29.45 | 1.58 | 5.84 |
| gemma-4-31B QAT W4A16 | 2k | 92.0% | 14.78 → 14.94 | 0.028 | 0.091 |
| | 32k | 95.9% | 12.93 → 13.21 | 0.0062 | 0.020 |

Reading it:

- **Qwen3.6 and Qwen3.8-27B are at the noise floor.** Two stock runs of the 27B INT8 differ by
  0.0006 on their own; every Qwen row is within a few times that, with
  97-99% top-1 agreement at every depth.
- **Qwen3.6-35B-A3B FP8** (0.006-0.008) is above the bf16 35B (0.003-0.005) for
  the same reason as the 27B FP8: stock quantizes activations to FP8, the plugin
  keeps them in bf16. Against the stock bf16 model the plugin is the closer of
  the two (table below). The speed fixes that kept the arithmetic (row padding, tile table)
  left every number bit-identical, since each output sums over K in the same
  order whatever the tile; the raw-decode kernels (exact fp16 weights, scale on
  the fp32 accumulator) moved KLD by less than 0.001.
- **Qwen3.8-27B FP8** (0.0032 at 2k) is slightly above the INT8 row. The two
  sides are different arithmetic: vLLM quantizes activations to FP8 and
  multiplies FP8 × FP8, the plugin keeps them in bf16. Perplexity is unchanged
  (-0.04%). Stock was run at 2k only; a 32k stock prefill was not attempted.
  The table below scores the plugin against stock bf16 instead, which is the
  quantization error plus the kernels.
- **Qwen3.8-Flash-Next** (0.008-0.019) is above the 27B floor. Its MoE GEMM
  quantizes activations to int8 per 128-group (`moe_w4a8`), which is a different
  arithmetic, not a bf16 reordering. Perplexity is unchanged at 2k and 128k
  and 0.6% higher at 32k. The reference is near-stock
  (every kernel off but `gdn_conv`; see Checkpoint notes).
- **LFM2.5** (0.02-0.03) is a real, small difference owned by attention.
- **gemma-4-26B** (0.2-0.3) is real but not a kernel defect. The plugin's
  attention is within 2-4 bf16 ulp of an fp32 reference at every context
  length, and on short chat text each replaced op moves the model by only
  0.02-0.03 nats with no top-1 flips. Over long context gemma amplifies
  bf16-level differences between two equally valid computations; Qwen,
  given the same kernels, does not. GSM8K is unchanged (247 / 247). Which
  side is closer to an fp32 model is not measured.
- **gemma-4-31B MXFP4** compares two different arithmetics (fp4 × fp4
  emulation against fp4 × bf16), so its KLD is not a pass/fail number.
- **gemma-4-31B qat** sits at its own floor: stock's Triton W4A16 kernel is
  split-K with a varying reduction order, so two stock runs differ by 0.014.

### FP8 against stock bf16

#### Qwen3.8-27B FP8

The plugin on the FP8 checkpoint, scored against the stock bf16 Qwen3.8-27B
reference sequences (same wikitext tails, top-20 logprobs). This is quantization
plus kernels, not kernel error alone.

| depth | top-1 | PPL bf16 stock → FP8 strix | KLD mean | KLD p95 |
| ---: | ---: | --- | ---: | ---: |
| 2k | 97.3% | 3.798 → 3.813 | 0.0043 | 0.0122 |
| 32k | 97.9% | 3.360 → 3.390 | 0.0051 | 0.0095 |

#### Qwen3.6-35B-A3B FP8

Same method: the plugin and stock vLLM on the FP8 checkpoint, each scored against
the stock bf16 Qwen3.6-35B-A3B reference sequences.

| depth | side | top-1 | PPL (bf16 stock → FP8) | KLD mean | KLD p95 |
| ---: | --- | ---: | --- | ---: | ---: |
| 2k | strix | 95.9% | 3.688 → 3.695 | 0.0090 | 0.031 |
| 2k | stock | 96.3% | 3.688 → 3.681 | 0.0114 | 0.043 |
| 32k | strix | 96.3% | 3.306 → 3.307 | 0.0070 | 0.021 |
| 32k | stock | 95.6% | 3.306 → 3.308 | 0.0084 | 0.027 |

The plugin is closer to bf16 than stock FP8 at both depths (KLD 17-21% lower).
The 128k point was not run for this model.

### The compressed-tensors FP8 checkpoints

- **LFM2.5 FP8** (0.08) differs because stock rounds activations to FP8 and the
  plugin does not: against the bf16 LFM2.5 at 2k the plugin is the closer of
  the two (KLD 0.078 vs 0.092 for stock FP8).
- **gemma-4-26B FP8** is bit-identical to its bf16 weights at 2k. At 32k the
  scored tail is a 512-token last chunk, which the fused FP8 kernel takes (it
  applies the scale in fp32 to the exact FP8 values; the bf16 copy holds
  rounded weights). Sending those rows through the dequant + hipBLASLt path
  instead makes 32k bit-identical too (KLD 0.0000, 100 % top-1), so the 0.17 is
  that rounding difference, which gemma amplifies as it does for the bf16
  checkpoint (0.2-0.3 above).
- **Sage-A1.5 and MiniCPM-V-4.7** (0.004-0.005) sit at the bf16 35B's level.

### GSM8K

Greedy, n = 250 unless noted; "-" is a side not run.

| model | stock | strix |
| --- | ---: | ---: |
| Qwen3.6-35B-A3B (n = 1319) | 1254 | 1249 |
| Qwen3.6-35B-A3B + MTP 2 | 237 | 238 |
| Qwen3.6-35B-A3B FP8 | - | 240 |
| Sage-A1.5 FP8 | - | 236 |
| MiniCPM-V-4.7 FP8 | - | 242 |
| Qwen3.8-27B | 237 | 235 |
| Qwen3.8-27B AWQ INT4 | 241 | 238 |
| Qwen3.8-27B FP8 | - | 238 |
| Qwen3.8-Flash-Next AWQ, bf16 / int8 KV | - | 242 / 244 |
| LFM2.5-24B-A2B | 196 | 197 |
| LFM2.5-24B-A2B FP8 | - | 202 |
| Muse-Glimmer-30B FP8 | - | 224 |
| Flint FP8 | - | 224 |
| gemma-4-26B-A4B | 247 | 247 |
| gemma-4-26B-A4B FP8 | - | 242 |

No paired difference is significant. Stock was not run on the FP8
checkpoints: per-channel FP8 does not load, and the 27B FP8 decodes at 0.57
tok/s, which would take hours.

## Per-model tables

Prefill and decode tok/s, single stream; "x" is strix over stock; "-" is a depth
not measured.

### Qwen3.6-35B-A3B

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1,583 | 3,581 | 2.26x | 1,331 | 3,589 | 2.70x |
| 8,192 | 1,124 | 4,284 | 3.81x | 735 | 4,167 | 5.67x |
| 16,384 | 774 | 3,903 | 5.04x | 447 | 3,888 | 8.70x |
| 32,768 | 469 | 3,393 | 7.23x | 250 | 3,331 | 13.32x |
| 65,536 | 260 | 2,617 | 10.08x | 132 | 2,558 | 19.32x |
| 131,072 | 139 | 1,799 | 12.98x | 68.30 | 1,751 | 25.64x |
| 261,120 | - | - | - | 34.50 | 1,085 | 31.44x |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 17.81 | 32.55 | 1.83x | 17.62 | 32.84 | 1.86x |
| 8,192 | 17.18 | 31.89 | 1.86x | 16.96 | 32.37 | 1.91x |
| 16,384 | 16.40 | 31.10 | 1.90x | 16.16 | 31.91 | 1.97x |
| 32,768 | 15.00 | 29.54 | 1.97x | 14.67 | 30.96 | 2.11x |
| 65,536 | 12.83 | 26.92 | 2.10x | 12.45 | 29.26 | 2.35x |
| 131,072 | 10.00 | 22.75 | 2.28x | 9.58 | 26.36 | 2.75x |
| 261,120 | - | - | - | 6.68 | 22.06 | 3.30x |

### Qwen3.6-35B-A3B W8A8

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1,824 | 3,214 | 1.76x | 1,504 | 3,221 | 2.14x |
| 8,192 | 1,162 | 3,950 | 3.40x | 751 | 3,931 | 5.23x |
| 16,384 | 792 | 3,646 | 4.60x | 453 | 3,611 | 7.97x |
| 32,768 | 475 | 3,173 | 6.68x | 252 | 3,133 | 12.46x |
| 65,536 | 263 | 2,485 | 9.46x | 133 | 2,432 | 18.30x |
| 131,072 | 138 | 1,733 | 12.58x | 68.41 | 1,697 | 24.81x |
| 261,120 | - | - | - | 34.61 | 1,059 | 30.61x |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 29.32 | 29.32 | 1.00x | 28.99 | 29.74 | 1.03x |
| 8,192 | 27.78 | 28.92 | 1.04x | 27.26 | 29.28 | 1.07x |
| 16,384 | 25.84 | 28.31 | 1.10x | 25.28 | 28.99 | 1.15x |
| 32,768 | 22.64 | 27.07 | 1.20x | 22.09 | 28.37 | 1.28x |
| 65,536 | 18.16 | 24.96 | 1.37x | 17.59 | 26.98 | 1.53x |
| 131,072 | 13.00 | 21.38 | 1.65x | 12.47 | 24.52 | 1.97x |
| 261,120 | - | - | - | 7.90 | 20.88 | 2.64x |

### Qwen3.6-35B-A3B FP8

Block-scaled FP8 (128x128), dense and expert GEMMs. Stock is measured only at 2k and 8k
(int8 KV, one pass each): 341 / 309 tok/s prefill and 4.37 / 4.32 tok/s decode, against
the strix int8 column below. The deeper stock points were not run.

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | - | 3,531 | - | 341 | 3,541 | 10.4x |
| 8,192 | - | 3,906 | - | 309 | 3,884 | 12.6x |
| 16,384 | - | 3,611 | - | - | 3,581 | - |
| 32,768 | - | 3,137 | - | - | 3,102 | - |
| 65,536 | - | 2,464 | - | - | 2,423 | - |
| 131,072 | - | 1,722 | - | - | 1,688 | - |
| 261,120 | - | - | - | - | 1,052 | - |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | - | 46.63 | - | 4.37 | 47.18 | 10.8x |
| 8,192 | - | 45.08 | - | 4.32 | 46.08 | 10.7x |
| 16,384 | - | 43.51 | - | - | 45.05 | - |
| 32,768 | - | 40.52 | - | - | 43.19 | - |
| 65,536 | - | 35.75 | - | - | 39.98 | - |
| 131,072 | - | 28.77 | - | - | 34.78 | - |
| 261,120 | - | - | - | - | 27.67 | - |

### Sage-A1.5 FP8

Per-channel FP8, Qwen3.6-35B-A3B shapes, linear-attention projections in bf16.

prefill tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 3,738 | 3,750 |
| 8,192 | 4,044 | 4,053 |
| 16,384 | 3,729 | 3,707 |
| 32,768 | 3,236 | 3,189 |
| 65,536 | 2,510 | 2,471 |
| 131,072 | 1,749 | 1,708 |
| 261,120 | - | 1,066 |

decode tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 39.62 | 40.01 |
| 8,192 | 38.50 | 39.19 |
| 16,384 | 37.36 | 38.49 |
| 32,768 | 35.17 | 37.14 |
| 65,536 | 31.51 | 34.71 |
| 131,072 | 25.93 | 30.62 |
| 261,120 | - | 25.03 |

### MiniCPM-V-4.7 FP8

Per-channel FP8, Qwen3.6-35B-A3B language model.

prefill tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 3,785 | 3,795 |
| 8,192 | 4,082 | 4,049 |
| 16,384 | 3,757 | 3,712 |
| 32,768 | 3,247 | 3,196 |
| 65,536 | 2,524 | 2,475 |
| 131,072 | 1,754 | 1,716 |
| 261,120 | - | 1,067 |

decode tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 39.18 | 39.60 |
| 8,192 | 38.12 | 38.86 |
| 16,384 | 36.99 | 38.21 |
| 32,768 | 34.84 | 36.86 |
| 65,536 | 31.24 | 34.50 |
| 131,072 | 25.75 | 30.49 |
| 261,120 | - | 24.83 |

### Qwen3.8-27B

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 514 | 677 | 1.32x | 423 | 709 | 1.67x |
| 8,192 | 317 | 674 | 2.13x | 208 | 686 | 3.29x |
| 16,384 | 200 | 647 | 3.23x | 127 | 657 | 5.18x |
| 32,768 | 109 | 604 | 5.56x | 70.75 | 611 | 8.64x |
| 65,536 | 54.93 | 533 | 9.70x | 37.58 | 539 | 14.34x |
| 131,072 | 27.47 | 406 | 14.77x | 19.41 | 435 | 22.39x |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 4.33 | 4.36 | 1.01x | 4.34 | 4.37 | 1.01x |
| 8,192 | 4.24 | 4.32 | 1.02x | 4.26 | 4.35 | 1.02x |
| 16,384 | 4.12 | 4.27 | 1.04x | 4.16 | 4.32 | 1.04x |
| 32,768 | 3.90 | 4.17 | 1.07x | 3.97 | 4.26 | 1.07x |
| 65,536 | 3.53 | 3.99 | 1.13x | 3.64 | 4.14 | 1.14x |
| 131,072 | 2.96 | 3.66 | 1.24x | 3.12 | 3.93 | 1.26x |

### Qwen3.8-27B INT8

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 304 | 694 | 2.28x | 270 | 692 | 2.56x |
| 8,192 | 217 | 679 | 3.13x | 161 | 675 | 4.19x |
| 16,384 | 155 | 652 | 4.20x | 109 | 648 | 5.94x |
| 32,768 | 92.68 | 608 | 6.56x | 65.60 | 603 | 9.19x |
| 65,536 | 49.99 | 537 | 10.74x | 36.55 | 532 | 14.56x |
| 131,072 | 25.78 | 413 | 16.01x | - | 430 | - |
| 261,120 | - | - | - | - | 312 | - |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 3.28 | 7.74 | 2.36x | 3.26 | 7.77 | 2.38x |
| 8,192 | 3.21 | 7.58 | 2.36x | 3.21 | 7.65 | 2.38x |
| 16,384 | 3.14 | 7.32 | 2.33x | 3.15 | 7.52 | 2.39x |
| 32,768 | 3.02 | 6.85 | 2.27x | 3.03 | 7.20 | 2.37x |
| 65,536 | 2.80 | 6.08 | 2.17x | 2.84 | 6.62 | 2.33x |
| 131,072 | 2.42 | 4.95 | 2.04x | - | 5.71 | - |
| 261,120 | - | - | - | - | 5.54 | - |

### Qwen3.8-27B AWQ INT4

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 380 | 690 | 1.81x | 335 | 688 | 2.05x |
| 8,192 | 174 | 674 | 3.88x | 141 | 669 | 4.74x |
| 16,384 | 132 | 647 | 4.91x | 101 | 642 | 6.35x |
| 32,768 | 83.91 | 604 | 7.20x | 64.38 | 598 | 9.28x |
| 65,536 | 47.32 | 534 | 11.28x | 37.26 | 528 | 14.17x |
| 131,072 | - | 408 | - | - | 428 | - |
| 261,120 | - | - | - | - | 312 | - |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 12.84 | 12.94 | 1.01x | 12.84 | 12.98 | 1.01x |
| 8,192 | 12.03 | 12.40 | 1.03x | 12.19 | 12.68 | 1.04x |
| 16,384 | 11.09 | 11.72 | 1.06x | 11.42 | 12.26 | 1.07x |
| 32,768 | 9.62 | 10.56 | 1.10x | 10.11 | 11.43 | 1.13x |
| 65,536 | 7.61 | 8.82 | 1.16x | 8.22 | 10.06 | 1.22x |
| 131,072 | - | 6.63 | - | - | 8.10 | - |
| 261,120 | - | - | - | - | 5.84 | - |

### Qwen3.8-27B FP8

Block-scaled FP8 (128x128). Stock is not measured: its Triton FP8 GEMM has no
tuned config here and decodes at 0.57 tok/s, so the stock
sweep was skipped.

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | - | 664 | - | - | 658 | - |
| 8,192 | - | 711 | - | - | 682 | - |
| 16,384 | - | 684 | - | - | 656 | - |
| 32,768 | - | 638 | - | - | 610 | - |
| 65,536 | - | 560 | - | - | 537 | - |
| 131,072 | - | 421 | - | - | 434 | - |
| 261,120 | - | - | - | - | 318 | - |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | - | 7.68 | - | - | 7.70 | - |
| 8,192 | - | 7.59 | - | - | 7.65 | - |
| 16,384 | - | 7.44 | - | - | 7.58 | - |
| 32,768 | - | 7.14 | - | - | 7.41 | - |
| 65,536 | - | 6.60 | - | - | 7.05 | - |
| 131,072 | - | 5.75 | - | - | 6.45 | - |
| 261,120 | - | - | - | - | 5.48 | - |

### Qwen3.8-Flash-Next AWQ

Measured on the [strix-vllm](https://github.com/asf0/strix-vllm) fork's
Docker image rather than unpatched upstream: only the fork streams the 102 GB
n-gram table from disk (`engram_config={"disk_offload": true}`; it does not fit
in 128 GB alongside the model) and gives the QSA layers an int8 KV cache. The
served settings: int8 KV, 16,384-token prefill chunks,
`--gpu-memory-utilization 0.85`; no MTP in these runs.

"stock" is every libstrix kernel off except `gdn_conv`. With stock vLLM's GDN
causal conv the engine fails at startup: the next n-gram lookup reads
out-of-range ids, at 8k and at 16k chunks. Strix's own conv is a small part of
the prefill step, so the ratios are close to plugin-off.

Prefill is the fastest run at each depth; decode is the median of every run.
bf16 stock was not run on the image.

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | - | 1,515 | - | 717 | 1,387 | 1.93x |
| 8,192 | - | 1,521 | - | 725 | 1,746 | 2.41x |
| 16,384 | - | 1,717 | - | 615 | 1,676 | 2.73x |
| 32,768 | - | 1,695 | - | 677 | 1,708 | 2.52x |
| 65,536 | - | 1,655 | - | 698 | 1,652 | 2.37x |
| 131,072 | - | 1,555 | - | 710 | 1,518 | 2.14x |
| 261,120 | - | 1,538 | - | 702 | 1,423 | 2.03x |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | - | 25.40 | - | 16.09 | 24.42 | 1.52x |
| 8,192 | - | 24.93 | - | 16.77 | 24.52 | 1.46x |
| 16,384 | - | 25.70 | - | 14.95 | 25.96 | 1.74x |
| 32,768 | - | 25.31 | - | 14.84 | 24.41 | 1.64x |
| 65,536 | - | 28.47 | - | 16.02 | 27.53 | 1.72x |
| 131,072 | - | 25.26 | - | 17.22 | 24.16 | 1.40x |
| 261,120 | - | 27.64 | - | 16.97 | 26.90 | 1.59x |

Reading it:

- **Disk reads.** Each token reads 16 rows of the table, and only about 17 GB
  of it stays in the page cache. A request whose rows are not cached waits on
  the drive, which serves about 100k random reads a second: about 0.3 s at
  2k, where such a run prefills at about 540 tok/s, below stock. Now and then
  a request stalls much longer (a 32k prefill once took 215.8 s instead of
  about 20 s). About 40 % of a new prompt's lookups are rows not read before,
  each a random read.
- **A row cache would not help.** Replaying 6 million tokens of real
  conversations (3,476 chats, code and maths) against 17 GiB of cache, the
  page cache hits 59.8 % of row reads (11,072 disk reads per conversation); a
  row cache of the same size hits 66.5 % (9,234). The conversations touch
  34.8 million distinct rows (10.4 GiB), so the row-cache figure is exact. That
  is 17 % fewer reads, about 0.05 s of the 0.3 s a 2k prompt waits, 3-4 % of
  TTFT at 2k and less for longer prompts. Only more RAM or a faster drive
  removes the rest.
- **Decode gather.** A decode step reads 16 rows, often uncached; tasks of 4
  rows fault them in parallel, which makes a decode step 5.5 % faster at 2k.
- **bf16 against int8 KV**: the same prefill. Interleaved in one session
  (`--max-model-len 133120`), TTFT at 32k was 20.0 and 22.8 s with int8
  against 20.1 and 23.1 s with bf16, at 128k 83.5 and 86.5 s against 85.3 and
  84.4 s. Decode is about 3 % faster with bf16 (35.1 against 36.3 ms per step
  at 128k).
- **MTP 2** (the recipe default) decodes at 40-43 tok/s on real text (the vLLM
  docs as prompts); the random-token prompts here would accept fewer drafts.

#### NPU prefill

`STRIX_ENABLE=npu` (see [kernels.md](kernels.md#turning-kernels-on-and-off)),
kernel 7.4.0-rc1, int8 KV, n-gram table from disk.

Served (DFlash 2 recipe, a full image build, `GPU_UTIL=0.82` for both, TTFT
of the second send of each prompt, n-gram rows cached; plugin 4 server runs,
NPU 3, alternated):

| prompt | plugin | NPU |
| --- | ---: | ---: |
| ~16k tokens (three prompts) | 7.65-7.79 s, mean 7.70 | 7.08-7.20 s, mean 7.14 (-7.3 %) |
| ~30k tokens (two prompts) | 14.50-14.55 s, mean 14.53 | 13.45-13.47 s, mean 13.46 (-7.4 %) |
| an image + ~8k tokens of text | 4.41-4.82 s, "Red." | 4.61-4.94 s, "The color of the square is red." |

Other lengths, three server runs each (TTFT of the second send; prefill
rate = prompt tokens / TTFT):

| prompt | plugin | NPU |
| --- | ---: | ---: |
| 7,643 tokens | 4.16 / 4.16 / 4.15 s (1,839 tok/s) | 4.02 / 4.01 / 4.13 s (1,886 tok/s, -2.5 %) |
| 57,053 tokens | 32.51 / 32.30 / 32.64 s (1,757 tok/s) | 30.02 / 30.00 / 30.06 s (1,900 tok/s, -7.6 %) |
| 85,032 tokens | 49.75 / 49.57 / 49.86 s (1,710 tok/s) | 45.66 / 45.64 / 45.75 s (1,861 tok/s, -8.1 %) |
| an image + ~30k tokens of text | 15.34 / 15.44 / 15.22 s | 14.29 / 14.31 / 14.34 s (-6.7 %), the same answer |
| two ~16k prompts at once (time to both answers) | 28.57, 23.26, 23.25 s | 22.94, 22.82, 22.75 s, the same answers |

Two requests in one step take the GPU path, so the concurrent case barely
moves (the first plugin run was slower).

Offline (host venv, 16,384-token chunks, `--gpu-memory-utilization 0.80`,
prefill time of the second pass over the same prompt):

| depth | plugin (three runs) | plugin, eager | NPU |
| ---: | ---: | ---: | ---: |
| 8,192 | 4.195 / 4.203 / 4.217 s | 4.157 s | 3.871 s (-7.6 %) |
| 16,384 | 9.260 / 9.003 / 9.008 s | 8.942 s | 7.784 s (-14.2 %) |
| 32,768 | 18.175 / 18.158 / 18.164 s | 18.057 s | 16.271 s (-10.3 %) |

Reading it:

- **Accuracy** (KLD at 16k against the plugin, 2,048 scored positions): NPU
  0.0115 mean (p95 0.046), top-1 95.8 %, perplexity +0.47 %; the plugin
  against itself 0.0090 (p95 0.036), 96.7 %, +0.23 %. The difference is the
  W8A8 projections, about what the int8 KV cache costs.
- **The NPU is slower per row but hidden.** A GDN projection of 8,192 rows
  takes 44.2 ms on the NPU (15.6 TOPS) against 15.8 ms on the GPU; with two
  banks it runs while the GPU does the other bank's layer.
- **Memory.** The NPU pins ~3 GB (weight panels and two banks of arenas),
  opened during vLLM's memory profile: 412,699 KV tokens at `GPU_UTIL=0.82`
  against 471,150 without the NPU. At 0.85 the memory guard stopped vLLM
  (host MemAvailable 9 GiB); 32k prefill chunks do not fit with the NPU
  (1.33 GiB of KV cache left at 0.80).
- **Which steps.** Steps below 4,096 tokens, batches of several requests and
  graph capture keep the GPU path, so decode and short prompts do not change.

Qwen3.8-27B bf16 (`STRIX_ENABLE=npu`, two K halves, 47 GDN layers; prefill of
an 8,192-token prompt, second pass, three runs each, alternating):

| configuration | prefill | tok/s |
| --- | --- | ---: |
| plugin, compiled | 11.29 / 10.88 / 11.35 s (mean 11.17) | 733 |
| plugin, eager | 12.19 / 12.25 s | 670 |
| NPU + GPU, eager loop | 11.44 s | 717 |
| NPU + GPU, compiled loop | 10.20 / 10.35 / 10.50 s (mean 10.35, **-7.3 %**) | 792 |

Depth (same method, one run each):

| depth | plugin, compiled | NPU + GPU, compiled loop |
| ---: | ---: | ---: |
| 8,192 | 11.21 s | 10.46 s (-6.7 %) |
| 16,384 | 23.60 s | 22.02 s (-6.7 %) |
| 32,768 | 49.81 s | 49.07 s (-1.5 %) |

Served (the [npu overlay](docker.md#npu-prefill) over the recipe,
`GPU_UTIL=0.82` for both, MTP 2, alternated; request time of a fresh prompt,
`max_tokens` 1 - the first request at a bank shape pays the dense regions'
one-time compile, so the steady runs are the later ones at each length):

| prompt | plugin | NPU |
| --- | ---: | ---: |
| ~8.2k tokens | 11.48 / 11.52 s | 10.68 s (**-7.3 %**) |
| ~16.4k tokens | 23.76 / 23.80 s | 22.07 / 22.13 s (**-7.1 %**) |
| ~24.6k tokens | 37.15 s | 34.33 s (**-7.6 %**) |

Reading it:

- **Accuracy.** KLD at 8k 0.0106 (p95 0.0147), top-1 97.3 %, perplexity
  +0.74 % against the plugin; at 9,216, a split of 3,072 + 6,144 rows,
  0.00268 (p95 0.0053), 98.5 %, -0.05 %. One activation scale over both
  K halves instead of one per half is worse: KLD 0.0140, +2.14 %.
- **The NPU is slower per row but hidden.** On a 4,096-row bank the two
  chains take 51.7 ms (13.4 TOPS) against 15.5 ms for the GPU's own GEMM;
  each chain runs while the GPU does the other bank's layer.
- **The split follows the served chunks.** The engine chunks an 8.2k
  prompt at 8,080 tokens and a 16.4k one at 16,160, sizes that do not
  halve cleanly, so the step splits at the boundary that covers the most
  NPU rows (`STRIX_NPU_SIZES`, default 2,048-8,192): 8,080 gives banks of
  4,096 + 3,984 rows (89 % covered), 16,160 gives 8,192 + 7,968 (95 %). A
  plain half-split won 2.2 % served (11.23 against 11.48 s at 8.2k).
  16,384 tokens is the last depth a single step covers fully; at 32,768
  the NPU takes each bank's first 8,192 rows and the rest runs on the
  GPU, and the win shrinks to 1.5 %.
- **Which steps.** Steps below 4,096 tokens, batches of several requests,
  graph capture and non-first pipeline ranks keep the GPU path, so decode
  and short prompts do not change. The loop's dense regions are
  torch.compiled (`STRIX_NPU_COMPILE=0` keeps the whole loop eager - the
  11.44 s row in the table).
- **Memory.** The panels (~10.5 GiB) pin during vLLM's memory profile, so
  they are counted before the KV cache is sized: 826,647 KV tokens at
  `GPU_UTIL=0.82` against 1,121,559 without, host `MemAvailable` 18.4 GiB
  with the NPU resident (the guard stops vLLM at 10 GiB). At 0.85 the
  guard stopped the served engine at startup.
- **Scope.** The panels are built for K = 5,120 bf16 projections;
  quantized checkpoints and any other shape turn the path off (both
  quantized ones were tried with dequantized panels and measured
  break-even). The other bf16 recipes gain nothing: Qwen3.6-35B-A3B turns
  off by shape, LFM2.5-24B-A2B and gemma-4-26B-A4B are not hooked.

### LFM2.5-24B-A2B

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 3,208 | 5,007 | 1.56x | 2,530 | 5,124 | 2.02x |
| 8,192 | 3,032 | 5,748 | 1.90x | 1,756 | 5,883 | 3.35x |
| 16,384 | 1,919 | 5,332 | 2.78x | 1,060 | 5,422 | 5.11x |
| 32,768 | 888 | 4,593 | 5.17x | 590 | 4,613 | 7.82x |
| 65,536 | 304 | 3,615 | 11.87x | 311 | 3,540 | 11.39x |
| 130,048 | 121 | 2,404 | 19.80x | 150 | 2,405 | 16.01x |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 18.00 | 38.33 | 2.13x | 17.80 | 38.28 | 2.15x |
| 8,192 | 17.77 | 37.49 | 2.11x | 17.40 | 37.48 | 2.15x |
| 16,384 | 17.60 | 36.47 | 2.07x | 16.98 | 36.53 | 2.15x |
| 32,768 | 17.14 | 34.68 | 2.02x | 16.06 | 34.72 | 2.16x |
| 65,536 | 16.25 | 31.50 | 1.94x | 14.51 | 31.69 | 2.18x |
| 130,048 | 14.88 | 26.77 | 1.80x | 12.28 | 27.11 | 2.21x |

### LFM2.5-24B-A2B FP8

Block FP8, 64 experts (top-4).

prefill tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 4,972 | 4,948 |
| 8,192 | 5,052 | 5,002 |
| 16,384 | 4,723 | 4,634 |
| 32,768 | 4,149 | 4,052 |
| 65,536 | 3,291 | 3,196 |
| 130,048 | 2,261 | 2,244 |

decode tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 52.72 | 52.53 |
| 8,192 | 51.15 | 51.26 |
| 16,384 | 49.32 | 49.52 |
| 32,768 | 46.02 | 46.31 |
| 65,536 | 40.54 | 41.11 |
| 130,048 | 33.04 | 33.64 |

### Muse-Glimmer-30B FP8

Dense, block FP8.

prefill tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 580 | 499 |
| 8,192 | 495 | 374 |
| 16,384 | 421 | 300 |
| 32,768 | 328 | 220 |
| 65,536 | 223 | 141 |
| 130,048 | 136 | 76 |

decode tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 7.47 | 7.45 |
| 8,192 | 7.38 | 7.34 |
| 16,384 | 7.26 | 7.17 |
| 32,768 | 7.06 | 6.92 |
| 65,536 | 6.92 | 6.75 |
| 130,048 | 6.71 | 6.51 |

### Flint FP8

Block FP8, 256 experts (top-6), attention sinks, 513-token sliding window on
most layers. Docker image, `--gpu-memory-utilization 0.85`.

prefill tok/s

| depth | bf16 strix | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 4,163 | 471 | 4,137 | 8.78x |
| 8,192 | 4,381 | 469 | 4,278 | 9.12x |
| 16,384 | 3,886 | 343 | 3,775 | 11.00x |
| 32,768 | 3,121 | 222 | 3,001 | 13.54x |
| 65,536 | 2,221 | 127 | 2,121 | 16.71x |
| 131,072 | 1,437 | - | 1,344 | - |
| 261,120 | - | - | 778 | - |

decode tok/s

| depth | bf16 strix | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 61.20 | 13.75 | 63.01 | 4.58x |
| 8,192 | 57.74 | 13.23 | 60.90 | 4.60x |
| 16,384 | 53.91 | 12.70 | 58.65 | 4.62x |
| 32,768 | 47.48 | 12.01 | 54.64 | 4.55x |
| 65,536 | 38.71 | 10.97 | 48.66 | 4.44x |
| 131,072 | 28.20 | - | 39.65 | - |
| 261,120 | - | - | 28.46 | - |

### gemma-4-26B-A4B

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1,548 | 3,591 | 2.32x | 890 | 3,877 | 4.35x |
| 8,192 | 813 | 3,602 | 4.43x | 430 | 4,011 | 9.33x |
| 16,384 | 487 | 3,377 | 6.93x | 246 | 3,629 | 14.75x |
| 32,768 | 270 | 3,063 | 11.34x | 142 | 3,071 | 21.64x |
| 65,536 | 140 | 2,108 | 15.09x | 75.09 | 2,336 | 31.11x |
| 131,072 | - | 1,485 | - | 38.38 | 1,588 | 41.37x |
| 261,120 | - | - | - | 18.75 | 971 | 51.80x |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 22.77 | 24.62 | 1.08x | 20.28 | 25.88 | 1.28x |
| 8,192 | 19.25 | 24.83 | 1.29x | 14.00 | 25.62 | 1.83x |
| 16,384 | 17.59 | 24.20 | 1.38x | 12.10 | 25.28 | 2.09x |
| 32,768 | 14.61 | 23.24 | 1.59x | 9.16 | 24.66 | 2.69x |
| 65,536 | 12.08 | 21.20 | 1.76x | 7.07 | 23.53 | 3.33x |
| 131,072 | - | 18.54 | - | 4.80 | 21.53 | 4.49x |
| 261,120 | - | - | - | 2.99 | 18.45 | 6.17x |

### gemma-4-26B-A4B AWQ

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1,356 | 3,395 | 2.50x | 867 | 3,358 | 3.87x |
| 8,192 | 702 | 3,641 | 5.19x | 414 | 3,528 | 8.52x |
| 16,384 | 444 | 3,335 | 7.51x | 252 | 3,232 | 12.83x |
| 32,768 | 256 | 2,864 | 11.19x | 142 | 2,769 | 19.48x |
| 65,536 | 138 | 2,225 | 16.16x | 76.15 | 2,151 | 28.24x |
| 131,072 | 71.48 | 1,554 | 21.75x | 39.44 | 1,499 | 38.00x |
| 261,120 | - | - | - | 20.08 | 934 | 46.49x |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 32.55 | 37.19 | 1.14x | 28.06 | 38.21 | 1.36x |
| 8,192 | 25.75 | 36.31 | 1.41x | 17.42 | 37.50 | 2.15x |
| 16,384 | 22.89 | 35.26 | 1.54x | 14.44 | 36.82 | 2.55x |
| 32,768 | 18.41 | 33.16 | 1.80x | 10.56 | 35.56 | 3.37x |
| 65,536 | 14.58 | 29.70 | 2.04x | 7.79 | 33.32 | 4.28x |
| 131,072 | 10.29 | 24.57 | 2.39x | 5.12 | 29.58 | 5.78x |
| 261,120 | - | - | - | 3.12 | 24.08 | 7.73x |

### gemma-4-26B-A4B FP8

Per-channel FP8, 128 experts (top-8), bf16-activation FP8 kernels.

prefill tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 3,378 | 3,327 |
| 8,192 | 3,647 | 3,517 |
| 16,384 | 3,480 | 3,228 |
| 32,768 | 2,968 | 2,802 |
| 65,536 | 2,288 | 2,189 |
| 131,072 | 1,581 | 1,520 |
| 261,120 | - | 945 |

decode tok/s

| depth | bf16 strix | int8 strix |
| ---: | ---: | ---: |
| 2,048 | 37.72 | 38.72 |
| 8,192 | 36.89 | 38.17 |
| 16,384 | 35.70 | 37.44 |
| 32,768 | 33.62 | 36.10 |
| 65,536 | 30.00 | 33.70 |
| 131,072 | 24.75 | 29.83 |
| 261,120 | - | 24.37 |

### gemma-4-31B MXFP4

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 289 | 492 | 1.70x | 203 | 542 | 2.67x |
| 8,192 | 166 | 550 | 3.31x | 100 | 542 | 5.40x |
| 16,384 | 106 | 518 | 4.89x | 60.82 | 511 | 8.41x |
| 32,768 | 61.62 | 469 | 7.61x | 34.56 | 462 | 13.37x |
| 65,536 | 33.48 | 397 | 11.85x | 18.55 | 389 | 20.96x |
| 131,072 | - | 303 | - | - | 296 | - |
| 261,120 | - | - | - | - | 200 | - |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1.59 | 9.36 | 5.89x | 1.56 | 9.69 | 6.21x |
| 8,192 | 1.56 | 9.10 | 5.84x | 1.48 | 9.50 | 6.43x |
| 16,384 | 1.53 | 8.80 | 5.74x | 1.43 | 9.30 | 6.52x |
| 32,768 | 1.46 | 8.27 | 5.66x | 1.30 | 8.93 | 6.88x |
| 65,536 | 1.39 | 7.42 | 5.33x | 1.19 | 8.30 | 6.99x |
| 131,072 | - | 6.13 | - | - | 7.29 | - |
| 261,120 | - | - | - | - | 5.87 | - |

### gemma-4-31B QAT W4A16

prefill tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 245 | 542 | 2.21x | 179 | 535 | 2.99x |
| 8,192 | 115 | 538 | 4.68x | 78.64 | 531 | 6.75x |
| 16,384 | 82.95 | 508 | 6.12x | 51.33 | 500 | 9.75x |
| 32,768 | 53.22 | 460 | 8.64x | 30.69 | 452 | 14.74x |
| 65,536 | 30.86 | 389 | 12.61x | 16.95 | 382 | 22.53x |
| 131,072 | - | 299 | - | - | 292 | - |
| 261,120 | - | - | - | - | 198 | - |

decode tok/s

| depth | bf16 stock | bf16 strix | x | int8 stock | int8 strix | x |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 8.96 | 10.39 | 1.16x | 8.16 | 10.76 | 1.32x |
| 8,192 | 8.00 | 10.10 | 1.26x | 6.30 | 10.58 | 1.68x |
| 16,384 | 7.41 | 9.76 | 1.32x | 5.48 | 10.36 | 1.89x |
| 32,768 | 6.02 | 9.14 | 1.52x | 3.96 | 9.95 | 2.52x |
| 65,536 | 4.97 | 8.11 | 1.63x | 3.09 | 9.22 | 2.99x |
| 131,072 | - | 6.61 | - | - | 8.05 | - |
| 261,120 | - | - | - | - | 6.41 | - |
