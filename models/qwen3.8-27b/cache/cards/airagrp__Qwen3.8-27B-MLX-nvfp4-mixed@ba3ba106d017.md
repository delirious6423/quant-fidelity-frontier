---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
language: en
tags:
  - mlx
  - mlx-vlm
  - nvfp4
  - mixed-precision
  - qwen3_5
  - multimodal
  - vision
  - video
  - text-generation
  - mtp
  - speculative-decoding
---
This repository contains [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) converted to MLX format with a mixed-precision quantization recipe, using [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) **0.6.17**.

## Quantization recipe

| Module | Precision |
|---|---|
| MLP `gate_proj` / `up_proj` / `down_proj` (64 layers) | nvfp4 (group_size=16, bits=4) |
| Full attention `q_proj` / `k_proj` / `v_proj` / `o_proj` (16 layers) | bfloat16 |
| Linear (GDN) attention `in_proj_*` / `out_proj` (48 layers) | bfloat16 |
| Token embeddings (`embed_tokens`) | bfloat16 |
| Output head (`lm_head`) | bfloat16 |
| MTP head | bfloat16 |
| Vision tower | bfloat16 |

- Effective size: ~31 GB (8.9 bits per weight), base model is ~54 GB in bfloat16.
- Quantized modules are stored as packed nvfp4 weights (E2M1 codes, 8 per uint32) with per-16 E4M3 block scales; bfloat16 modules are stored as-is. Per-module precision is detected from the presence of `.scales` tensors; the global mode is set in `config.json` (`quantization.mode = "nvfp4"`).

## MTP

The native MTP head is **merged into this checkpoint** as `language_model.mtp.*` tensors (15 tensors, bfloat16, norms in the MLX +1 convention), stored in `mtp.safetensors` and referenced from `model.safetensors.index.json` — it is not a separate drafter model. Use it for speculative decoding (`--draft-kind mtp` in mlx-vlm) or ignore it; base inference is unaffected.

## Use with mlx-vlm

```bash
pip install mlx-vlm
```

```python
import mlx_vlm

model, processor = mlx_vlm.load("airagrp/Qwen3.8-27B-MLX-nvfp4-mixed")
response, _ = mlx_vlm.generate(
    model,
    processor,
    prompts="In one sentence, what is MLX?",
    max_tokens=64,
)
print(response)
```

```bash
mlx_vlm.generate --model airagrp/Qwen3.8-27B-MLX-nvfp4-mixed --prompt "In one sentence, what is MLX?" --max-tokens 64
```

## Use with MLX directly

Load with the standard MLX safetensors layout; weights use the MLX nvfp4 block-scale format (`group_size=16`, `bits=4`).

## Citations / license

Apache-2.0. Refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B) for architecture details, benchmarks, and usage guidelines.
