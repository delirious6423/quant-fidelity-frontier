---
base_model: Qwen/Qwen3.8-27B
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- quantized
---

# Qwen3.8-27B 6-bit MLX

A 6-bit affine MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), using a group size of 64.

## Usage

```bash
pip install -U mlx-vlm
mlx_vlm.generate --model timothyckl/Qwen3.8-27B-6bit-MLX --prompt "Hello!"
```

See the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B) for model details, intended uses, and limitations.
