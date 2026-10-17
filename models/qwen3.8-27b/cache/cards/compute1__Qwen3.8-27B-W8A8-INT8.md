---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: text-generation
license: other
language:
  - multilingual
tags:
  - compressed-tensors
  - quantized
  - W8A8
  - INT8
---

# Qwen3.8-27B W8A8 INT8 (compressed-tensors)

Quantized from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) using [llm-compressor](https://github.com/neuralmagic/llm-compressor).

## Quantization Config

- **Format**: `int-quantized` (compressed-tensors)
- **Weights**: INT8, channel-wise, symmetric, MSE observer
- **Activations**: INT8, token-wise, symmetric, dynamic
- **Ignored**: `lm_head`, embeddings, vision encoder, MTP layers, GDN linear-attn sub-modules (`in_proj_a`, `in_proj_b`, `conv1d`, `norm`, `A_log`, `dt_bias`), all layernorms, q_norm, k_norm

## Architecture

Qwen3.8-27B uses a hybrid architecture with 48 Gated DeltaNet (GDN) linear-attention layers and 16 full-attention layers (every 4th layer). The GDN `in_proj_a`/`in_proj_b` projections are kept in BF16 because they require special handling for fused loading.

## Intended Use

Optimized for inference on Ampere GPUs (RTX 3090, A100) with SGLang/vLLM Marlin kernels. Weight-only INT8 with BF16 activations.