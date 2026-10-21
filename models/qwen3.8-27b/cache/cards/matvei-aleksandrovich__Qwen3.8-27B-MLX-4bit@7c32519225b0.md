---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - mlx
  - mlx-vlm
  - qwen
  - qwen3.8
  - apple-silicon
  - 4-bit
---

# Qwen3.8-27B MLX 4-bit

This is a 4-bit affine MLX conversion of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple
silicon. The original tokenizer, chat template, and image/video processor
files are included.

## Details

- Architecture: `qwen3_5` / `Qwen3_5ForConditionalGeneration`
- Quantization: 4-bit affine, group size 64
- Native context length: 262,144 tokens
- Weight size: approximately 15 GiB
- Conversion code: `erokhins/mlx-vlm`, commit `77f16c25`

## Generate

```bash
python -m mlx_vlm.generate \
  --model matvei-aleksandrovich/Qwen3.8-27B-MLX-4bit \
  --prompt "Explain linear attention in simple terms." \
  --max-tokens 256
```

For speculative decoding, use the matching
[`Qwen3.8-27B-MTP-4bit`](https://huggingface.co/matvei-aleksandrovich/Qwen3.8-27B-MTP-4bit)
drafter. A drafter from another Qwen version is not a supported substitute.

## Conversion

```bash
mlx_vlm.convert \
  --hf-path Qwen/Qwen3.8-27B \
  --mlx-path ./Qwen3.8-27B-MLX-4bit \
  --quantize --q-bits 4
```

This community conversion is not an official Qwen release. Validate quality,
performance, and memory use on your own workload.

