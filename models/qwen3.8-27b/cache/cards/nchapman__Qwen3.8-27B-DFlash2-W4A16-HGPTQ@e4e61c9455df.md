---
base_model:
- incoai/Qwen3.8-27B-DFlash2
- Qwen/Qwen3.8-27B
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

# Qwen3.8-27B-DFlash2-W4A16-HGPTQ

A 4-bit quantized version of [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), a lightweight block-diffusion speculative drafter for `Qwen3.8-27B`. Instead of drafting one token at a time, it proposes multiple upcoming tokens in parallel to significantly speed up generation.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (used alongside `Qwen3.8-27B` via `--draft-model`):

```bash
warp34 serve nchapman/Qwen3.8-27B-W4A16-HGPTQ \
  --draft-model nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ
```

## About this quant

- **`2.74×` smaller in VRAM** (`1.31 GiB` vs. `3.58 GiB` in BF16), saving `2.27 GiB` alongside the 27B target model.
- **`3.76×` less memory bandwidth per draft step** (`843 MiB` vs. `3,175 MiB`), making parallel drafting noticeably faster on consumer GPUs.
- **Virtually identical acceptance rate to BF16:** Preserves `99.8%` of the 16-bit drafter's acceptance length (`2.535` vs. `2.540` tokens/step) and `100.0%` of its top-16 candidate recall.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Plugs straight into `--draft-model` with any `Qwen3.8-27B` checkpoint.

**Pairs with:** [Qwen3.8-27B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ) or [Qwen3.8-27B-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
