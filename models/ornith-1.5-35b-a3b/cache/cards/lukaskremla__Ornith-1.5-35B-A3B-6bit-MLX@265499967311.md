---
model_name: Ornith-1.5-35B-A3B 6-bit MLX
library_name: mlx
pipeline_tag: image-text-to-text
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
  - mlx
  - mlx-vlm
  - safetensors
  - ornith
  - ornith-1.5
  - ornith-1.5-35b-a3b
  - qwen
  - qwen3_5
  - qwen3_5_moe
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
  - 6-bit
  - bf16-vision-tower
---
# lukaskremla/Ornith-1.5-35B-A3B-6bit-MLX

This quantization preserves the model's BF16 vision tower. For a text-only version check [lukaskremla/Ornith-1.5-35B-A3B-6bit-MLX-TextOnly](https://huggingface.co/lukaskremla/Ornith-1.5-35B-A3B-6bit-MLX-TextOnly).

You can download MTP heads separately from this collection [Ornith 1.5 35B-A3B MLX-Quants (Vision, Text-Only & MTP)](https://huggingface.co/collections/lukaskremla/ornith-15-35b-a3b-mlx-quants-vision-text-only-and-mtp).

Hugging Face might render incorrect parameter counts for this model, it is a common display bug for MLX quants.

## Credits

Converted to MLX format from [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) using mlx-vlm version **0.6.17**.
