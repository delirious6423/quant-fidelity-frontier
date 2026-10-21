---
license: apache-2.0
library_name: mtplx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- macos
- m1
- m2
- fp16
- speculative-decoding
- multi-token-prediction
- qwen
- qwen3.8
- mtp
- mtplx
- local-ai
- coding
- qwen3-8
- qwen-3.8
- local-llm
- llm
- m5
- m5-max
- m4
- m3
- macbook-pro
- mac-studio
- opencode
- claude-code
- 27b
- qwen3.8-27b
- qwen3-8-27b
---

**[MTPLX](https://mtplx.com): the fastest way to run Qwen 3.8 on a Mac. Native multi-token-prediction speculative decoding on Apple Silicon, two to three times the speed of plain decoding, exact at any temperature.**

# Qwen 3.8 27B Optimized Speed FP16

**4-bit dynamic quant. Great coding speeds and good quality. Recommended.** This is the M1 and M2 build.

The FP16 precision sibling of
[Qwen 3.8 27B Optimized Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed).
M1 and M2 Macs do not run bf16 well, so this build keeps every quantized weight
byte-identical to the parent and stores the remaining floating tensors (scales,
biases, norms, the GDN convolution and state parameters, and the MTP head) in
fp16 instead of bf16. Same layout, same tuned depth and draft settings, same
context window. On an M3 or newer Mac use the parent instead.

[MTPLX](https://mtplx.com) picks the right one for you: the app and
`mtplx start` route M1 and M2 Macs to the FP16 builds and everything newer to
the parents. It is the default MTPLX picks on an M1 or M2 Mac with 32 GB or more.

## Measured on MTPLX 2.11.3 (16 September 2026)

This is a Qwen3.8-27B MLX pack for MTPLX, the fastest way to run Qwen 3.8 27B on a Mac. All numbers on a MacBook Pro M5 Max, fans verified at maximum, sampled at the model's own settings. Conditions and sources: [mtplx.com/benchmarks](https://mtplx.com/benchmarks/).

| Run | tok/s |
|---|---|
| Optimized Speed rewriting a file it just wrote, stock settings (MTPLX 2.10.0) | 87.6 |
| Bare Speed on a fresh coding task at official Qwen 3.8 sampling (MTPLX 2.7.0) | 65.2 |
| Optimized Speed on a 3k-token chat answer (MTPLX 2.10.0) | 64.3 |
| The 27B record on a fresh generation: Qwen 3.6 27B Optimized Speed, 192-token bench, 2 July 2026, raw logs published | 81.74 |

Pack quality against the bf16 Qwen3.8-27B checkpoint, teacher-forced over 2,389 positions of code, prose, JSON and a multilingual notice (16 September 2026): Optimized Speed (4-bit dynamic) 96.0 percent top-1 agreement and KL 0.012; Optimized Quality (8-bit dynamic) 99.3 percent and KL 0.0005. Exactness on this release: a thousand four-token draws from the fast path match a thousand from the plain path within the plain path's own noise, at temperature 1, top-p 0.95, top-k 20. Details: [MTPLX 2.11.3 release notes](https://mtplx.com/releases/2.11.3/).

Runs on Apple Silicon Macs with 32 GB of unified memory or more (36 GB for Optimized Quality): MacBook Pro, MacBook Air, Mac mini and Mac Studio on M1 to M5. Guide: [Run Qwen 3.8 27B on a Mac](https://mtplx.com/models/qwen3.8-27b/). Comparison pages: [MTPLX vs mlx-serve](https://mtplx.com/compare/mtplx-vs-mlx-serve/), [MTPLX vs oMLX](https://mtplx.com/compare/mtplx-vs-omlx/), [MTPLX vs LM Studio](https://mtplx.com/compare/mtplx-vs-lm-studio/), [MTPLX vs Ollama](https://mtplx.com/compare/mtplx-vs-ollama/).

## Speeds

The numbers we publish for the parent were measured on an M5 Max: 58.7 tok/s on the coding task and 35.1 to 37.3 tok/s on long xhigh reasoning,
official Qwen 3.8 sampling, generation running to the model's own stop. This
FP16 build has the same weights and runs the same MTPLX turbo path, so the
speculative math is identical; absolute tok/s on an M1 or M2 depends on that
chip. We have not published M1 or M2 numbers for it yet.

| | |
|---|---|
| Download | 20.4 GB |
| Peak unified memory (parent, measured on M5 Max) | 23.6 GB |
| Context window | 262,144 tokens |
| MTP depth | 3 |
| Sampling | temperature 1.0, top-p 0.95, top-k 20 (the official Qwen 3.8 contract) |

`MTPLX_FP16_CONVERSION_MANIFEST.json` in the repo lists every tensor that was
cast and every tensor that was preserved, with sha256 for each shard.
Speculation in MTPLX is exact at any temperature: drafts are accepted with the
probability-ratio rule plus residual resampling.

## Use it

Mac app: download at [mtplx.com](https://mtplx.com), pick "Qwen 3.8 27B Optimized Speed FP16".

Command line:

```bash
pip install mtplx
mtplx serve --model Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16
```

The other two FP16 builds:
[Bare Speed FP16](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed-FP16),
[Optimized Quality FP16](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16).
