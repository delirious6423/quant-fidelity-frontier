---
model_name: Qwen3.8-27B 6-bit MLX Text-Only
library_name: mlx
pipeline_tag: text-generation
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - mlx
  - mlx-lm
  - safetensors
  - qwen
  - qwen3.8
  - qwen3.8-27b
  - qwen3_5
  - text-generation
  - text-only
  - conversational
  - reasoning
  - tool-use
  - long-context
  - multilingual
  - quantized
  - weight-only-quantization
  - affine
  - rtn
  - group-size-64
  - 6-bit
---

# lukaskremla/Qwen3.8-27B-6bit-MLX-TextOnly

This quantization strips the model's vision tower and preserves only text-to-text capabilities. For a vision capable version check [lukaskremla/Qwen3.8-27B-6bit-MLX](https://huggingface.co/lukaskremla/Qwen3.8-27B-6bit-MLX).

You can download MTP heads separately from this collection [Qwen 3.8 27B MLX-Quants (Vision, Text-Only & MTP)](https://huggingface.co/collections/lukaskremla/qwen-38-27b-mlx-quants-vision-text-only-and-mtp).

Hugging Face might render incorrect parameter counts for this model, it is a common display bug for MLX quants.

## Credits

Converted to MLX format from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) using mlx-lm version **0.31.2**.
