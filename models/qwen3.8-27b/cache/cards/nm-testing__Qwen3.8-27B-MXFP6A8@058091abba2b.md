---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- compressed-tensors
- mxfp6
- mxfp8
- fp8
---

# Qwen3.8-27B-MXFP6A8

A calibration-free [compressed-tensors](https://github.com/vllm-project/compressed-tensors)
quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), sized to fit a
single 32 GB GPU. It follows the layout of
[gaunernst/Qwen3.8-27B-MXFP6](https://huggingface.co/gaunernst/Qwen3.8-27B-MXFP6), stored in the
compressed-tensors format.

| Layers | Quantization |
|---|---|
| Dense MLP | MXFP6 (E2M3) weights, dynamic MXFP8 activations (`mxfp6-pack-quantized`) |
| Attention (full + GDN) and MTP projections | MXFP8 weights and activations |
| LM head | FP8 per-channel weights, dynamic per-token FP8 activations |
| Embeddings, vision tower, GDN `in_proj_a/b`, MTP `fc`, norms | BF16 |

`quantize.py` reproduces this checkpoint with LLM Compressor's `model_free_ptq` (round-to-nearest,
no calibration data).

## Evaluation

GSM8K, 5-shot, greedy, all 1319 questions, with vLLM's `tests/evals/gsm8k/gsm8k_eval.py` on one B300:

| Model | Accuracy |
|---|---|
| Qwen/Qwen3.8-27B (BF16) | 0.782 |
| Qwen3.8-27B-MXFP6A8 | 0.770 |

## Usage

Requires compressed-tensors and vLLM with MXFP6 support (in progress).

```bash
vllm serve nm-testing/Qwen3.8-27B-MXFP6A8
```
