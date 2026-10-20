---
model_name: Ornith-1.5-35B-A3B 4-bit MLX Text-Only
library_name: mlx
pipeline_tag: text-generation
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
  - mlx
  - mlx-lm
  - safetensors
  - ornith
  - ornith-1.5
  - ornith-1.5-35b-a3b
  - qwen
  - qwen3_5
  - qwen3_5_moe
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
  - 4-bit
---

# lukaskremla/Ornith-1.5-35B-A3B-4bit-MLX-TextOnly

This quantization strips the model's vision tower and preserves only text-to-text capabilities. For a vision capable version check [lukaskremla/Ornith-1.5-35B-A3B-4bit-MLX](https://huggingface.co/lukaskremla/Ornith-1.5-35B-A3B-4bit-MLX).

You can download MTP heads separately from this collection [Ornith 1.5 35B-A3B MLX-Quants (Vision, Text-Only & MTP)](https://huggingface.co/collections/lukaskremla/ornith-15-35b-a3b-mlx-quants-vision-text-only-and-mtp).

Hugging Face might render incorrect parameter counts for this model, it is a common display bug for MLX quants.

## Credits

Converted to MLX format from [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) using mlx-lm version **0.31.3**.
