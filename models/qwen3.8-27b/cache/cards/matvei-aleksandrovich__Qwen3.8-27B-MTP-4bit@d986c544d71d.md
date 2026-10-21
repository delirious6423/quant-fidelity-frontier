---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
  - mlx
  - speculative-decoding
  - mtp
  - qwen
  - qwen3.8
  - apple-silicon
  - 4-bit
---

# Qwen3.8-27B MTP 4-bit drafter

This repository contains the native Multi-Token Prediction weights extracted
from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and
quantized to 4-bit affine MLX weights.

It is not a standalone language model. Use it only as the MTP drafter for
[`matvei-aleksandrovich/Qwen3.8-27B-MLX-4bit`](https://huggingface.co/matvei-aleksandrovich/Qwen3.8-27B-MLX-4bit).

## Details

- Model type: `qwen3_5_mtp`
- Draft kind: `mtp`
- Block size: 3
- Quantization: 4-bit affine, group size 64
- Weight size: approximately 228 MiB
- Extraction code: `erokhins/mlx-vlm`, based on commit `77f16c25`

## Generate with the matching target

```bash
python -m mlx_vlm.generate \
  --model matvei-aleksandrovich/Qwen3.8-27B-MLX-4bit \
  --draft-model matvei-aleksandrovich/Qwen3.8-27B-MTP-4bit \
  --draft-kind mtp \
  --prompt "Write a short Python example." \
  --max-tokens 256
```

The target model verifies every proposed token. Measure the acceptance rate
and end-to-end speed on representative prompts before production use.

