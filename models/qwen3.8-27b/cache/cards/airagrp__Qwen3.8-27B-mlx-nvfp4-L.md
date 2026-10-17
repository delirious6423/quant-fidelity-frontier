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
| Full attention `q_proj` / `k_proj` / `v_proj` / `o_proj` (16 layers) | mxfp8 (group_size=32, bits=8) |
| Linear (GDN) attention `in_proj_*` / `out_proj` (48 layers) | mxfp8 (group_size=32, bits=8) |
| Token embeddings (`embed_tokens`) | bfloat16 |
| Output head (`lm_head`) | bfloat16 |
| MTP head | bfloat16 |
| Vision tower | bfloat16 |

- Effective size: ~24 GB (7.9 bits per weight), base model is ~54 GB in bfloat16.
- Quantized modules: mxfp8 (group_size=32, bits=8) / nvfp4 (group_size=16, bits=4); bfloat16 modules are stored as-is. Per-module precision is detected from the presence of `.scales` tensors.

## MTP

The native MTP head is **merged into this checkpoint** as `language_model.mtp.*` tensors (15 tensors, bfloat16, norms in the MLX +1 convention), stored in `mtp.safetensors` and referenced from `model.safetensors.index.json` — it is not a separate drafter model. Use it for speculative decoding (`--draft-kind mtp` in mlx-vlm) or ignore it; base inference is unaffected.

## Use with mlx-vlm

```bash
pip install mlx-vlm
```

```python
import mlx_vlm

model, processor = mlx_vlm.load("airagrp/Qwen3.8-27B-mlx-nvfp4-L")
response, _ = mlx_vlm.generate(
    model,
    processor,
    prompts="In one sentence, what is MLX?",
    max_tokens=64,
)
print(response)
```

```bash
mlx_vlm.generate --model airagrp/Qwen3.8-27B-mlx-nvfp4-L --prompt "In one sentence, what is MLX?" --max-tokens 64
```

## Use with MLX directly

Load with the standard MLX safetensors layout; quantized weights use mxfp8 (group_size=32, bits=8) / nvfp4 (group_size=16, bits=4).

## Citations / license

Apache-2.0. Refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B) for architecture details, benchmarks, and usage guidelines.

## Quality benchmarks

Perplexity and KL divergence on the WikiText-2 raw test split (297,053 tokens, 2048-token windows, fresh KV cache per window), measured against the bf16 reference (`Qwen3.8-27B`).

| Model | Size (GiB) | PPL (lower = better) | ΔPPL vs bf16 | KLD vs bf16 (nats/token, lower = better) |
|---|---|---|---|---|
| `Qwen3.8-27B` (bf16 reference) | 51.75 | 6.935 | — | — |
| [`Qwen3.8-27B-mlx-nvfp4-S`](https://huggingface.co/airagrp/Qwen3.8-27B-mlx-nvfp4-S) | 19.15 | 7.024 | +0.088 | 0.0573 |
| [`Qwen3.8-27B-mlx-nvfp4-M`](https://huggingface.co/airagrp/Qwen3.8-27B-mlx-nvfp4-M) | 21.57 | 6.983 | +0.048 | 0.0490 |
| [`Qwen3.8-27B-mlx-nvfp4-L`](https://huggingface.co/airagrp/Qwen3.8-27B-mlx-nvfp4-L) | 22.31 | 6.987 | +0.051 | 0.0407 |
| [`Qwen3.8-27B-mlx-mxfp8-S`](https://huggingface.co/airagrp/Qwen3.8-27B-mlx-mxfp8-S) | 26.62 | 6.922 | -0.014 | 0.0278 |
| [`Qwen3.8-27B-mlx-nvfp4-XL`](https://huggingface.co/airagrp/Qwen3.8-27B-mlx-nvfp4-XL) | 28.84 | 7.024 | +0.089 | 0.0382 |
| [`Qwen3.8-27B-mlx-mxfp8-M`](https://huggingface.co/airagrp/Qwen3.8-27B-mlx-mxfp8-M) | 29.78 | 6.893 | -0.042 | 0.0068 |
| [`Qwen3.8-27B-mlx-mxfp8-L`](https://huggingface.co/airagrp/Qwen3.8-27B-mlx-mxfp8-L) | 34.79 | 6.937 | +0.002 | 0.0046 |
| [`Qwen3.8-27B-mlx-mxfp8-XL`](https://huggingface.co/airagrp/Qwen3.8-27B-mlx-mxfp8-XL) | 36.31 | 6.937 | +0.002 | 0.0037 |

- **KLD** = mean per-token `D_KL(p_bf16 ‖ p_model)` over the full vocabulary — how much the quantized model's next-token distribution drifts from bf16.
- Data: `wikitext-2-raw-v1` test split, SHA-256 `c5b5caea5bd655cb…`; tokenizer: `Qwen3.8-27B`; mlx-vlm 0.6.17.
- Benchmarked 2026-09-03 with `benchmark_ppl_kld.py` (2048-token streaming windows, all positions except the first of each window scored).

### Plots

![Mean KLD vs quant size](benchmarks/kld-vs-size.png)

![PPL vs quant size](benchmarks/ppl-vs-size.png)
