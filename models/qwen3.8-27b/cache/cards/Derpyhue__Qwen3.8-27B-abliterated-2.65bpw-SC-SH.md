---
license: apache-2.0
base_model:
- huihui-ai/Huihui-Qwen3.8-27B-abliterated
- Qwen/Qwen3.8-27B
library_name: exllamav3
tags:
- exl3
- quantized
- tabbyapi
- vision-language
- abliterated
---

# Qwen3.8-27B-abliterated — EXL3 2.65 bpw (SC, SH)

EXL3 quantization of [Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated), an uncensored version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) created with abliteration.

Like the [main SC series](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.65bpw-SC-SH), bitrates are allocated per tensor by noise-attribution optimization (sc_optimize) with the quilter's scaled-Hessian option enabled (-sh), at head 4.0 / MTP 4.0 / vision 6.0 bpw. Unlike the main series, everything was re-derived for this model: the noise attribution was measured on the abliterated weights themselves, and calibration used traces collected from this model (217 rows × 2048 columns) instead of the generic corpus. All measurements below are therefore against the abliterated BF16 reference and are not comparable with the main-series tables.

## Quantization details

| Parameter | Value |
|---|---|
| Quant method | EXL3 (exllamav3) |
| Bitrate allocation | per-tensor, noise-attribution optimized (sc_optimize, measured on the abliterated model) |
| Average bitrate | 2.65 bpw (achieved 2.65) |
| Predicted KL divergence (sc_optimize estimate) | 0.0616 |
| Scaled Hessian (-sh) | enabled |
| Head (LM output) bitrate | 4.0 bpw |
| MTP layers bitrate | 4.0 bpw |
| Vision tower bitrate | 6.0 bpw |
| Codebook | mul1 |
| Output scales | auto |
| Calibration | 217 rows × 2048 columns (custom traces from the abliterated model) |
| Tensor tying | k_proj+v_proj, gate_proj+up_proj |
| Quantized with | exllamav3 conversion format version 1.5.4 |

## Measured quality

Measured against the abliterated BF16 reference. Weights (GiB) counts quantized tensors only; this quant totals ~12.11 GB on disk.

| Model | Weights (GiB) | Mean KLD | Median KLD | p90 KLD | Perplexity (increase) |
|---|---|---|---|---|---|
| [Abliterated BF16 (reference)](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) | 47.7 | — | — | — | 1.3418 |
| [2.55 H4 -sh](https://huggingface.co/Derpyhue/Qwen3.8-27B-abliterated-2.55bpw-SC-SH) | 7.87 | 0.0442 | 0.0026 | 0.128 | 1.4098 (+5.1%) |
| [2.60 H4 -sh](https://huggingface.co/Derpyhue/Qwen3.8-27B-abliterated-2.60bpw-SC-SH) | 8.01 | 0.0417 | 0.0023 | 0.122 | 1.4048 (+4.7%) |
| **[2.65 H4 -sh](https://huggingface.co/Derpyhue/Qwen3.8-27B-abliterated-2.65bpw-SC-SH)** | **8.15** | **0.0377** | **0.0021** | **0.109** | **1.3986 (+4.2%)** |

## Usage

Requires [ExLlamaV3](https://github.com/turboderp/exllamav3). The easiest way to run it is [tabbyAPI](https://github.com/theroyallab/tabbyAPI):

```yaml
# config.yml
model:
  model_dir: models
  model_name: Qwen3.8-27B-abliterated-2.65bpw-SC-SH
```

## Base model

[Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) is a refusal-removed variant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B native vision-language model: 64-layer hybrid architecture (Gated DeltaNet + Gated Attention), 262K native context (extensible to 1M), image and video understanding, and multi-token prediction training.

## Disclaimer

This model has had its refusal behavior removed and will comply with requests a base model would decline; the safety caveats of the original Qwen card apply with extra force. Quantized models also lose some accuracy relative to their full-precision original; bitrates here were chosen as a size/quality tradeoff.
