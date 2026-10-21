---
model_name: Qwen3.8-27B 2-bit MLX
library_name: mlx
pipeline_tag: image-text-to-text
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - mlx
  - mlx-vlm
  - safetensors
  - qwen
  - qwen3.8
  - qwen3.8-27b
  - qwen3_5
  - image-text-to-text
  - multimodal
  - vision-language
  - vision
  - video
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
  - 2-bit
  - bf16-vision-tower
---
# lukaskremla/Qwen3.8-27B-2bit-MLX

This quantization preserves the model's BF16 vision tower. For a text-only version check [lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly](https://huggingface.co/lukaskremla/Qwen3.8-27B-2bit-MLX-TextOnly).

You can download MTP heads separately from this collection [Qwen 3.8 27B MLX-Quants (Vision, Text-Only & MTP)](https://huggingface.co/collections/lukaskremla/qwen-38-27b-mlx-quants-vision-text-only-and-mtp).

Hugging Face might render incorrect parameter counts for this model, it is a common display bug for MLX quants.

## Credits

Converted to MLX format from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) using mlx-vlm version **0.6.13**.
