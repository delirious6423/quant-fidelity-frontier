---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - qwen3
  - quantized
  - nvfp4
  - llmcompressor
---

# Qwen3.8-27B-NVFP4

NVFP4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) via `oneshot` calibration.

Vision encoder, gated-delta attention (conv1d / linear_attn), `lm_head`, embeddings, and norm layers are kept in original precision.
