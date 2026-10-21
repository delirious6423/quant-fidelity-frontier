---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
language: en
tags:
  - mlx
  - mlx-vlm
  - mxfp8
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
| MLP `gate_proj` / `up_proj` / `down_proj` (64 layers) | mxfp8 (group_size=32, bits=8) |
| Full attention `q_proj` / `k_proj` / `v_proj` / `o_proj` (16 layers) | bfloat16 |
| Linear (GDN) attention `in_proj_*` / `out_proj` (48 layers) | bfloat16 |
| Token embeddings (`embed_tokens`) | bfloat16 |
| Output head (`lm_head`) | bfloat16 |
| MTP head | bfloat16 |
| Vision tower | bfloat16 |

- Effective size: ~39 GB (11.2 bits per weight), base model is ~54 GB in bfloat16.
- Quantized modules are stored as packed mxfp8 weights with per-32 E8M0 block scales; bfloat16 modules are stored as-is. Per-module precision is detected from the presence of `.scales` tensors; the global mode is set in `config.json` (`quantization.mode = "mxfp8"`).

## MTP

The native MTP head is **merged into this checkpoint** as `language_model.mtp.*` tensors (15 tensors, bfloat16, norms in the MLX +1 convention), stored in `mtp.safetensors` and referenced from `model.safetensors.index.json` — it is not a separate drafter model. Use it for speculative decoding (`--draft-kind mtp` in mlx-vlm) or ignore it; base inference is unaffected.

## Use with mlx-vlm

```bash
pip install mlx-vlm
```

```python
import mlx_vlm

model, processor = mlx_vlm.load("airagrp/Qwen3.8-27B-MLX-mxfp8-mixed")
response, _ = mlx_vlm.generate(
    model,
    processor,
    prompts="In one sentence, what is MLX?",
    max_tokens=64,
)
print(response)
```

```bash
mlx_vlm.generate --model airagrp/Qwen3.8-27B-MLX-mxfp8-mixed --prompt "In one sentence, what is MLX?" --max-tokens 64
```

## Use with MLX directly

Load with the standard MLX safetensors layout; weights use the MLX mxfp8 block-scale format (`group_size=32`, `bits=8`).

## Citations / license

Apache-2.0. Refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B) for architecture details, benchmarks, and usage guidelines.
