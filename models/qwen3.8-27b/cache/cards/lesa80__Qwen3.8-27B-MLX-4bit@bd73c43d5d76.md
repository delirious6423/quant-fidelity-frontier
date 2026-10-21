---
tags:
- mlx
- qwen
- qwen3.8
- vision-language-model
- vlm
- image-text-to-text
- reasoning
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
library_name: mlx-vlm
---

# Qwen3.8-27B-MLX-4bit

MLX 4-bit quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

- **BF16 version:** [lesa80/Qwen3.8-27B-MLX-bf16](https://huggingface.co/lesa80/Qwen3.8-27B-MLX-bf16)

**All capabilities preserved:**
- ✅ **Vision** — image + video (27-layer ViT encoder)
- ✅ **Reasoning** — thinking mode (токены `thinking` / `response`)
- ✅ **Длинный контекст** — 262K токенов
- ✅ **Multilingual** — русский, английский, китайский

| Property | Value |
|----------|-------|
| Quantization | 4-bit affine, group_size=64 |
| Model size | ~16 GB |
| RAM required | 16-24 GB |
| License | Apache 2.0 |

## Usage

```bash
hf download lesa80/Qwen3.8-27B-MLX-4bit --local-dir ./qwen-mlx --local-dir-use-symlinks False
```

```python
from mlx_vlm import load, generate
model, processor = load("lesa80/Qwen3.8-27B-MLX-4bit")
output = generate(model, processor, "Describe this image.", image=["image.jpg"])
print(output)
```

```bash
mlx_vlm.generate --model lesa80/Qwen3.8-27B-MLX-4bit --prompt "Hello" --max-tokens 200
```

## Links

- [Original: Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [BF16: lesa80/Qwen3.8-27B-MLX-bf16](https://huggingface.co/lesa80/Qwen3.8-27B-MLX-bf16)