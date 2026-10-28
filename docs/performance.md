# Performance

libstrix against stock upstream vLLM on one gfx1151 APU (Strix Halo, 128 GB
unified memory).

- vLLM: upstream `899a1689e2`, unpatched.
- torch 2.14.0+rocm10.1.0, HIP 7.16.
- "stock" is vLLM with the plugin disabled, using its own kernels (Triton
  attention, Triton and `fused_moe` GEMMs). "strix" is the plugin on.

Your ROCm and vLLM versions will move these numbers; the ratios are the point.

## Summary

Single stream, `int8_per_token_head` KV (the serving dtype). Speedup is
strix over stock; the last two columns are strix's own throughput.

| model | 2k prefill | 64k prefill | 2k decode | 64k decode | strix 128k prefill / decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B (bf16) | 2.70× | 19.3× | 1.86× | 2.35× | 1,751 / 26.4 |
| Qwen3.6-35B-A3B INT8 W8A8 | 2.14× | 18.3× | 1.03× | 1.53× | 1,697 / 24.5 |
| Qwen3.8-27B (bf16) | 1.67× | 14.3× | 1.01× | 1.14× | 435 / 3.9 |
| Qwen3.8-27B INT8 W8A16 + MTP | 2.56× | 14.6× | 2.38× | 2.33× | 430 / 5.7 |
| Qwen3.8-27B Quark AWQ INT4 | 2.05× | 14.2× | 1.01× | 1.22× | 428 / 8.1 |
| LFM2.5-24B-A2B (bf16) | 2.02× | 11.4× | 2.15× | 2.18× | 2,405 / 27.1 ¹ |
| gemma-4-26B-A4B (bf16) | 4.35× | 31.1× | 1.28× | 3.33× | 1,588 / 21.5 |
| gemma-4-26B-A4B AWQ 4-bit | 3.87× | 28.2× | 1.36× | 4.28× | 1,499 / 29.6 |
| gemma-4-31B MXFP4 ² | 2.67× | 21.0× | 6.21× | 6.99× | 296 / 7.3 |
| gemma-4-31B qat W4A16 | 2.99× | 22.5× | 1.32× | 2.99× | 292 / 8.0 |

¹ LFM2.5 caps context at 131,072; its deep point is 130,048.
² Stock vLLM runs MXFP4 as fp4 × fp4 emulation on this GPU; strix runs it as
fp4 weights × bf16 activations. The ratio is against a slow path, not a
tuned one.

What it means:

- **Prefill** is where the plugin matters most. Stock prefill time grows
  about 4× per doubling of context on this APU; strix's about 3×. At
  256k the 35B takes 241 s with strix and 7,568 s stock (31×).
- **Decode** doubles on the 35B, LFM2.5 and the 27B INT8 checkpoint, and on
  every gemma at depth. On the 27B bf16, the Quark AWQ 27B and the 35B W8A8,
  decode is at parity with stock: nothing the plugin replaces dominates their
  decode step.
- **No configuration is slower than stock**, single stream or batched
  (lowest: 0.98× at x16 on the 35B W8A8, inside run-to-run noise).

## Depth, strix only

Prefill and decode tok/s, single stream, int8 KV. The 256k runs use
`--gpu-memory-utilization 0.72` (the default 0.80 leaves under 10 GB of host
memory at that depth). The 27B bf16 checkpoint does not fit 256k.

| model | prefill 2k / 32k / 128k / 256k | decode 2k / 32k / 128k / 256k |
| --- | --- | --- |
| Qwen3.6-35B-A3B | 3,589 / 3,331 / 1,751 / 1,085 | 32.8 / 31.0 / 26.4 / 22.1 |
| Qwen3.6-35B-A3B W8A8 | 3,221 / 3,133 / 1,697 / 1,059 | 29.7 / 28.4 / 24.5 / 20.9 |
| Qwen3.8-27B (bf16) | 709 / 611 / 435 / - | 4.4 / 4.3 / 3.9 / - |
| Qwen3.8-27B INT8 + MTP | 692 / 603 / 430 / 312 | 7.8 / 7.2 / 5.7 / 5.5 |
| Qwen3.8-27B Quark AWQ | 688 / 598 / 428 / 312 | 13.0 / 11.4 / 8.1 / 5.8 |
| LFM2.5-24B-A2B | 5,124 / 4,613 / 2,405 / - | 38.3 / 34.7 / 27.1 / - |
| gemma-4-26B-A4B | 3,877 / 3,071 / 1,588 / 971 | 25.9 / 24.7 / 21.5 / 18.5 |
| gemma-4-26B-A4B AWQ | 3,358 / 2,769 / 1,499 / 934 | 38.2 / 35.6 / 29.6 / 24.1 |
| gemma-4-31B MXFP4 | 542 / 462 / 296 / 200 | 9.7 / 8.9 / 7.3 / 5.9 |
| gemma-4-31B qat W4A16 | 535 / 452 / 292 / 198 | 10.8 / 10.0 / 8.1 / 6.4 |

Decode here is without speculation; the 27B checkpoint's MTP head is not
used in these runs.

## Batched decode

Total decode tok/s over the batch, stock → **strix** (speedup), at 2k and 8k
context. int8 KV for every model; bf16 KV too for the 35B checkpoints, whose
decode-attention routing depends on batch size (strix takes decode up to 2
sequences on head 256 / GQA 8, stock Triton above that).

| model | KV | x4 @2k | x4 @8k | x16 @2k | x16 @8k |
| --- | --- | ---: | ---: | ---: | ---: |
| LFM2.5-24B-A2B | int8 | 36 → **78** (2.17×) | 34 → **74** (2.15×) | 111 → **181** (1.63×) | 90 → **153** (1.71×) |
| Qwen3.6-35B-A3B | bf16 | 41 → **93** (2.25×) | 38 → **79** (2.12×) | 97 → **164** (1.68×) | 83 → **141** (1.69×) |
| Qwen3.6-35B-A3B | int8 | 39 → **97** (2.51×) | 37 → **82** (2.20×) | 92 → **167** (1.82×) | 84 → **152** (1.82×) |
| Qwen3.6-35B-A3B-INT8-W8A8 | bf16 | 80 → **80** (1.00×) | 68 → **70** (1.04×) | 170 → **167** (0.98×) | 146 → **153** (1.05×) |
| Qwen3.6-35B-A3B-INT8-W8A8 | int8 | 78 → **84** (1.08×) | 72 → **78** (1.08×) | 183 → **180** (0.98×) | 153 → **167** (1.09×) |
| Qwen3.8-27B | int8 | 17 → **17** (1.02×) | 16 → **17** (1.06×) | 57 → **60** (1.06×) | 49 → **58** (1.17×) |
| Qwen3.8-27B-INT8-W8A16-MTP | int8 | 13 → **29** (2.30×) | 12 → **28** (2.30×) | 46 → **96** (2.09×) | 40 → **84** (2.08×) |
| Qwen3.8-27B-Quark-AWQ-INT4-W4A16 | int8 | 47 → **50** (1.05×) | 41 → **45** (1.11×) | 100 → **122** (1.21×) | 79 → **104** (1.31×) |
| gemma-4-26B-A4B-it | int8 | 40 → **67** (1.67×) | 33 → **67** (2.02×) | 86 → **158** (1.85×) | 74 → **175** (2.36×) |
| gemma-4-26B-A4B-it-AWQ-4bit | int8 | 56 → **99** (1.77×) | 43 → **86** (1.99×) | 100 → **230** (2.29×) | 78 → **225** (2.89×) |
| gemma-4-31B-it-MXFP4 | int8 | 6 → **29** (5.19×) | 5 → **28** (5.46×) | 14 → **92** (6.43×) | 12 → **83** (6.81×) |
| gemma-4-31B-it-qat-w4a16-ct | int8 | 21 → **39** (1.84×) | 15 → **37** (2.38×) | 22 → **104** (4.72×) | 17 → **93** (5.45×) |

## Serving

Qwen3.8-27B INT8 W8A16 with MTP k=2, int8 KV, 131k context, 40 concurrent
sequences, gmu 0.7, driven by `vllm bench serve` with 80 requests of 2,048
tokens in / 1,024 out.

| metric | strix |
| --- | ---: |
| output throughput | 123.3 tok/s |
| total throughput | 369.9 tok/s |
| TPOT mean / median / p99 | 240 / 228 / 552 ms |
| TTFT median | 13.8 s |

## Accuracy: KLD against stock

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
| Qwen3.8-27B (bf16) | 2k | 99.0% | 3.798 → 3.800 | 0.0008 | 0.0029 |
| | 32k | 99.3% | 3.360 → 3.376 | 0.0036 | 0.0025 |
| Qwen3.8-27B INT8 + MTP | 2k | 98.5% | 3.813 → 3.832 | 0.0022 | 0.0073 |
| | 32k | 97.6% | 3.375 → 3.394 | 0.0018 | 0.0063 |
| | 128k | 98.3% | 2.673 → 2.673 | 0.0022 | 0.0068 |
| Qwen3.8-27B Quark AWQ | 2k | 98.3% | 3.865 → 3.874 | 0.0065 | 0.0067 |
| | 32k | 98.2% | 3.469 → 3.476 | 0.0015 | 0.0052 |
| LFM2.5-24B-A2B | 2k | 94.2% | 6.488 → 6.509 | 0.034 | 0.094 |
| | 32k | 95.5% | 5.711 → 5.798 | 0.023 | 0.082 |
| gemma-4-26B-A4B | 2k | 85.5% | 16.33 → 16.50 | 0.28 | 1.28 |
| | 32k | 88.4% | 51.95 → 55.06 | 0.22 | 1.00 |
| | 128k | 88.5% | 28.95 → 33.80 | 0.20 | 0.98 |
| gemma-4-26B-A4B AWQ | 2k | 82.4% | 16.73 → 19.73 | 0.34 | 1.83 |
| | 32k | 88.3% | 19.59 → 22.02 | 0.23 | 0.89 |
| | 128k | 83.3% | 7.462 → 7.725 | 0.26 | 1.35 |
| gemma-4-31B MXFP4 | 2k | 59.7% | 13.44 → 13.29 | 1.54 | 5.58 |
| | 32k | 53.1% | 51.81 → 29.45 | 1.58 | 5.84 |
| gemma-4-31B qat W4A16 | 2k | 92.0% | 14.78 → 14.94 | 0.028 | 0.091 |
| | 32k | 95.9% | 12.93 → 13.21 | 0.0062 | 0.020 |

Reading it:

- **Qwen is at the noise floor.** Two stock runs of the 27B INT8 differ by
  0.0006 on their own; every Qwen row is within a few times that, with
  97-99% top-1 agreement at every depth.
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

## Method

- One request per depth in a fresh engine per config: prompt of exactly that
  many tokens, then 128 decode steps (64 at 256k). Prefill throughput is
  depth / TTFT; decode is 1000 / ms per step. Depths below 64k run twice and
  report the second.
- `--max-num-seqs 1`, prefix caching off, prefill chunk 8,192 tokens,
  `--gpu-memory-utilization 0.80` (0.72 at 256k).
- Batched: N sequences decode together after a 2k or 8k context, 64 steps.
- Stock is measured to 64k on every model and at 128k and 256k on a
  subset: a single stock request at those depths takes 30 minutes to 4
  hours.

## Per-model tables

Prefill and decode tok/s, single stream; "x" is strix over stock; "-" is a depth
not measured.

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

### Qwen3.6-35B-A3B-INT8-W8A8

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

### Qwen3.8-27B-INT8-W8A16-MTP

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

### Qwen3.8-27B-Quark-AWQ-INT4-W4A16

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

### gemma-4-26B-A4B-it

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

### gemma-4-26B-A4B-it-AWQ-4bit

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

### gemma-4-31B-it-MXFP4

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

### gemma-4-31B-it-qat-w4a16-ct

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
