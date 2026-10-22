---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: exllamav3
tags:
- exl3
- quantized
- tabbyapi
- vision-language
- moe
---

# DeepSeek-V4.1-Flash-1017 — EXL3 2.50 bpw (H5, HQ5bpw, SH)

EXL3 quantization of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), a mixture-of-experts vision-language model with 384 routed experts per layer, engram n-gram memory and a 1M-token context.

The routed experts, which hold almost all of the weights, are quantized to 2.5 bpw. Everything every token passes through is kept at a high bitrate: attention and shared experts at 5.0 bpw (HQ5bpw: a fixed 5 bpw floor via -hqb 5, where the stock --hq flag would give 4 bpw here), the head at 5.0 bpw (H5), with the quilter's scaled-Hessian option enabled (-sh). The engram n-gram tables are not quantized at all: they are the original FP8 tables, copied byte for byte (-ngb 16), split into one file per layer as in the original repo.

## Quantization details

| Parameter | Value |
|---|---|
| Quant method | EXL3 (exllamav3) |
| Average bitrate | 2.50 bpw target (achieved 2.53 over all quantized tensors) |
| Routed experts | 2.5 bpw |
| Attention and shared experts (-hqb) | 5.0 bpw |
| Head (LM output) bitrate | 5.0 bpw |
| MTP layers bitrate | 4.0 bpw |
| Vision tower bitrate | 6.0 bpw |
| Engram n-gram tables | original FP8 (E4M3 with E8M0 scales), unmodified (-ngb 16) |
| Scaled Hessian (-sh) | enabled |
| Codebook | mul1 |
| Output scales | auto |
| Calibration | 250 rows × 2048 columns (default corpus mix) |
| Quantized with | exllamav3 conversion format version 1.6.3 |

## Measured quality

Measured against the original FP8/FP4 model on wikitext (100 rows × 2048 tokens). Size on disk is the full repo footprint; both include the 202.8 GB of engram tables, which are read from disk or system RAM and do not take VRAM.

| Model | Size on disk | Mean KLD | Median KLD | p90 KLD | Perplexity (increase) |
|---|---|---|---|---|---|
| [DeepSeek-V4.1-Flash-1017 (original, reference)](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 510.3 GB | — | — | — | 2.9959 |
| **[2.50 H5 HQ5bpw -sh](https://huggingface.co/Derpyhue/DeepSeek-V4.1-Flash-1017-exl3-2.50bpw-H5-HQ5bpw-SH)** | **388.4 GB** | **0.1111** | **0.0179** | **0.289** | **3.1429 (+4.9%)** |

The quant picks the same top token as the original 89.6% of the time; the original predicts the actual next token as its top choice 73.87% of the time, this quant 72.85%.

## Usage

Requires [ExLlamaV3](https://github.com/turboderp/exllamav3) with DeepSeek-V4.1 support (at the time of writing [PR #449](https://github.com/turboderp-org/exllamav3/pull/449), not yet merged). The easiest way to run it is [tabbyAPI](https://github.com/theroyallab/tabbyAPI):

```yaml
# config.yml
model:
  model_dir: models
  model_name: DeepSeek-V4.1-Flash-1017-exl3-2.50bpw-H5-HQ5bpw-SH
```

The included `tabby_config.yml` sets `reasoning_effort: max` as the default chat template variable; a client can still send its own.

The engram tables are streamed from disk by default, so a fast NVMe drive helps. V4.1 uses a new DSML tool-call format (`<｜DSML｜ calls>`), so tool calling needs a parser for that format.

## Base model

See the [base model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) for full details: 40-layer mixture-of-experts with 384 routed experts and 1 shared expert (6 active per token), hyper-connections, engram n-gram memory in layers 1 and 14, 3 multi-token prediction layers, image input and a 1M-token context.

## Disclaimer

Quantized models lose some accuracy relative to the full-precision original. Bitrates were chosen as a size/quality tradeoff; your use case may tolerate more or less.
