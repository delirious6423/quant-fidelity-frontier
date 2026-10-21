---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: exllamav3
tags:
- exl3
- quantized
- tabbyapi
- vision-language
---

# Qwen3.8-27B — EXL3 2.75 bpw (SC, SH)

EXL3 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B native vision-language model with flexible thinking control, built on the Qwen3.5 architecture.

This is [Qwen3.8-27B-exl3-2.75bpw-SC](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.75bpw-SC) with the quilter's scaled-Hessian option enabled (-sh): the same per-tensor noise-attribution recipe (sc_optimize), the same head 4.0 / MTP 4.0 / vision 6.0 bpw, the same calibration and the same ~12.41 GB footprint, at slightly lower mean KLD and perplexity (see the table below, "2.75 H4 -sh").

## Quantization details

| Parameter | Value |
|---|---|
| Quant method | EXL3 (exllamav3) |
| Bitrate allocation | per-tensor, noise-attribution optimized (sc_optimize) |
| Average bitrate | 2.75 bpw (achieved 2.7498) |
| Predicted KL divergence (sc_optimize estimate) | 0.0514 |
| Scaled Hessian (-sh) | enabled |
| Head (LM output) bitrate | 4.0 bpw |
| MTP layers bitrate | 4.0 bpw |
| Vision tower bitrate | 6.0 bpw |
| Codebook | mul1 |
| Output scales | auto |
| Calibration | 250 rows × 2048 columns (default corpus mix) |
| Tensor tying | k_proj+v_proj, gate_proj+up_proj |
| Quantized with | exllamav3 conversion format version 1.5.4 |

## Measured quality

Measured against the BF16 original; the row for this quant is bold. Size on disk is the full repo footprint.

| Quant | Size on disk | Mean KLD | Median KLD | p90 KLD | Perplexity |
|---|---|---|---|---|---|
| [2.65 H4](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.65bpw-SC) | 12.11 GB | 0.032903 | 0.001696 | 0.095268 | 1.3795 |
| [2.65 H4 -sh](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.65bpw-SC-SH) | 12.11 GB | 0.032541 | 0.001560 | 0.092880 | 1.3739 |
| [2.67 H4 -sh](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.67bpw-SC-SH) | 12.17 GB | 0.031061 | 0.001562 | 0.089724 | 1.3748 |
| [2.70 H4](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.70bpw-SC) | 12.26 GB | 0.030083 | 0.001484 | 0.087552 | 1.3746 |
| [2.70 H4 -sh](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.70bpw-SC-SH) | 12.26 GB | 0.030040 | 0.001454 | 0.086279 | 1.3748 |
| [2.75 H4](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.75bpw-SC) | 12.41 GB | 0.027894 | 0.001434 | 0.081578 | 1.3725 |
| **[2.75 H4 -sh](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.75bpw-SC-SH)** | **12.41 GB** | **0.026945** | **0.001279** | **0.077931** | **1.3685** |
| [2.80 H4](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.80bpw-SC) | 12.57 GB | 0.025571 | 0.001298 | 0.074292 | 1.3690 |
| [2.80 H4 -sh](https://huggingface.co/Derpyhue/Qwen3.8-27B-exl3-2.80bpw-SC-SH) | 12.57 GB | 0.025098 | 0.001232 | 0.073292 | 1.3688 |

## Usage

Requires [ExLlamaV3](https://github.com/turboderp/exllamav3). The easiest way to run it is [tabbyAPI](https://github.com/theroyallab/tabbyAPI):

```yaml
# config.yml
model:
  model_dir: models
  model_name: Qwen3.8-27B-exl3-2.75bpw-SC-SH
```

## Base model

See the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B) for full details: 64-layer hybrid architecture (Gated DeltaNet + Gated Attention), 262K native context (extensible to 1M), image and video understanding, and multi-token prediction training.

## Disclaimer

Quantized models lose some accuracy relative to the full-precision original. Bitrates were chosen as a size/quality tradeoff; your use case may tolerate more or less.
