---
library_name: mlx
license: mit
pipeline_tag: image-text-to-text
base_model: ornith-ai/Ornith-1.5-35B-A3B
language: en
tags:
  - mlx
  - mlx-vlm
  - mxfp8
  - mixed-precision
  - qwen3_5
  - moe
  - multimodal
  - vision
  - video
  - text-generation
  - mtp
  - speculative-decoding
---
This repository contains [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) converted to MLX format with a mixed-precision quantization recipe, using [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) **0.6.17**.

## Quantization recipe

| Module | Precision |
|---|---|
| MoE `mlp.gate` router + `switch_mlp` experts (256×8) + shared expert (40 layers) | mxfp8 (group_size=32, bits=8) |
| Full attention `q_proj` / `k_proj` / `v_proj` / `o_proj` (10 layers) | mxfp8 (group_size=32, bits=8) |
| Linear (GDN) attention `in_proj_*` / `out_proj` (30 layers) | mxfp8 (group_size=32, bits=8) |
| Token embeddings (`embed_tokens`) | bfloat16 |
| Output head (`lm_head`) | bfloat16 |
| MTP head (full MoE decoder layer) | bfloat16 |
| Vision tower | bfloat16 |

- Effective size: ~39 GB (8.7 bits per weight), base model is ~67 GB in bfloat16.
- Quantized modules: mxfp8 (group_size=32, bits=8); bfloat16 modules are stored as-is. Per-module precision is detected from the presence of `.scales` tensors.

## MTP

The native MTP head (a full MoE decoder layer, 785 tensors) is **merged into this checkpoint** as `language_model.mtp.*` tensors (bfloat16, norms in the MLX +1 convention), stored in `mtp.safetensors` and referenced from `model.safetensors.index.json` — it is not a separate drafter model. Use it for speculative decoding (`--draft-kind mtp` in mlx-vlm) or ignore it; base inference is unaffected.

## Use with mlx-vlm

```bash
pip install mlx-vlm
```

```python
import mlx_vlm

model, processor = mlx_vlm.load("airagrp/Ornith-1.5-35B-A3B-mlx-mxfp8")
response, _ = mlx_vlm.generate(
    model,
    processor,
    prompts="In one sentence, what is MLX?",
    max_tokens=64,
)
print(response)
```

```bash
mlx_vlm.generate --model airagrp/Ornith-1.5-35B-A3B-mlx-mxfp8 --prompt "In one sentence, what is MLX?" --max-tokens 64
```

## Use with MLX directly

Load with the standard MLX safetensors layout; quantized weights use mxfp8 (group_size=32, bits=8).

## Citations / license

MIT. Refer to the [original model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for architecture details, benchmarks, and usage guidelines.
