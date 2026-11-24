---
base_model:
- incoai/Qwen3.8-Flash-Next-DFlash2
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: apache-2.0
library_name: compressed-tensors
pipeline_tag: text-generation
inference: false
quantized_by: nchapman
tags:
- dflash2
- speculative-decoding
- block-diffusion
- draft-model
- compressed-tensors
- w4a16
- gptq
- hadamard
- warp34
---

# Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ

A 4-bit quantized version of [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), a lightweight block-diffusion speculative drafter for `Qwen3.8-Flash-Next` (180B MoE). It proposes multiple upcoming tokens in parallel to dramatically boost decode speed.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (used alongside `Qwen3.8-Flash-Next` via `--draft-model`):

```bash
warp34 serve nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO --tp 2 \
  --draft-model nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ
```

## About this quant

- **Only `0.52 GiB` in VRAM** (down from `1.12 GiB` in BF16, `2.17×` smaller) — negligible overhead alongside the 180B MoE model.
- **`3.76×` less memory bandwidth per draft step** (`224 MiB` vs. `844 MiB`), keeping drafting overhead tiny.
- **Virtually identical acceptance rate to BF16:** Preserves `99.7%` of the 16-bit drafter's acceptance length (`2.752` vs. `2.760` tokens/step) and `99.9%` of its top-16 candidate recall.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Plugs straight into `--draft-model` with either `W2.6` or `W4` `Qwen3.8-Flash-Next`.

**Pairs with:** [Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO) or [Qwen3.8-Flash-Next-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
