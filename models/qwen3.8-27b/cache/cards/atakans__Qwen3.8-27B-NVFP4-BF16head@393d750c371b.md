---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: vllm
pipeline_tag: image-text-to-text
tags: [nvfp4, modelopt, quantized, blackwell, dgx-spark]
---

# Qwen3.8-27B NVFP4 BF16head

Qwen3.8-27B with the MLP layers in NVFP4, attention in FP8 and lm_head in BF16.
Calibrated with ModelOpt 0.47.0 Local-Hessian. Made and measured on a DGX Spark (GB10).

## Layers

| Part | Format |
|---|---|
| MLP | NVFP4 (static weight scales, Local-Hessian, FP8 scale sweep) |
| Full and linear attention | FP8 |
| lm_head, embeddings, vision, MTP | BF16 |

## Quality against BF16

WikiText-2 test, 100 × 512 tokens, top-20 logprobs. BF16 perplexity 8.30.
All rows measured by me with the same script and the same vLLM 0.30.0.

| | ppl ratio | KLD mean | KLD p99 | top-1 agree |
|---|---|---|---|---|
| [FP8 (Qwen)](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 1.002 | 0.0046 | 0.045 | 96.5% |
| **This model** | **1.022** | **0.0295** | **0.309** | **91.1%** |
| [NVFP4 (NVIDIA)](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | 1.029 | 0.0398 | 0.348 | 88.7% |

## Size and speed on one DGX Spark

| | Size | Decode, 1 request | TTFT, 8k prompt |
|---|---|---|---|
| FP8 (Qwen) | 29 GiB | 8.5 tok/s | 3.73 s |
| **This model** | **22 GiB** | **10.9 tok/s** | **2.97 s** |
| NVFP4 (NVIDIA) | 21 GiB | 12.5 tok/s | 2.90 s |

vLLM 0.30.0. The BF16 lm_head adds 1.7 GiB and about 7 ms per token.
It keeps the top-1 token closer to BF16.

## Difference from NVIDIA's NVFP4

Both use Local-Hessian. Two things differ:

- lm_head: BF16 here, NVFP4 there. BF16 costs 1.7 GiB and about 13% decode speed.
- Calibration data: 512 cnn_dailymail + 512 Nemotron post-training v1 here,
  2048 Nemotron post-training v3 there.

Pick this model for output closer to BF16. Pick NVIDIA's for speed and size.

## Run

```
vllm serve atakans/Qwen3.8-27B-NVFP4-BF16head --max-model-len 131072 --reasoning-parser qwen3
```

## Recipe

recipe.yaml in this repo. Calibration: 512 cnn_dailymail + 512 Nemotron post-training v1 samples.
