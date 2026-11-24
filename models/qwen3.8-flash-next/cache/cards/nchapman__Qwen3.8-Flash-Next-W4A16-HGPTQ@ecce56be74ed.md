---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- w4a16
- moe
- compressed-tensors
- vllm
- warp34
---

# Qwen3.8-Flash-Next-W4A16-HGPTQ

A 4-bit quantized version of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Qwen's flagship 180B-class Mixture-of-Experts multimodal model with hybrid attention, n-gram lookup embeddings, vision, and built-in Multi-Token Prediction (MTP) speculative decoding.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (2× 24 GB GPUs (`--tp 2` with tiered MoE) or an 80–96 GB GPU):

```bash
warp34 serve nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ --tp 2
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **`3.6×` smaller than BF16** (`92.84 GiB` total vs. `335.3 GiB` in BF16): `66.0 GiB` active model + `26.8 GiB` compressed 4-bit n-gram lookup table that `warp34` streams via `mmap`.
- **Full-model 4-bit calibration:** Quantizes all 512 routed experts per layer, shared experts, attention/GDN layers, vision tower, MTP drafter, and the massive `102 GB` n-gram table (down to `26.8 GiB`).
- **Near-BF16 fidelity:** Retains `97.4%` isolated vision top-1 (`99.3%` on UI/OCR guard) and `95.4%` MTP draft agreement with minimal divergence from the 335 GiB base model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Includes pre-computed hot-expert cache rankings (`static_hot_cache_rankings.json`) so `warp34` pins the hottest experts in VRAM and streams cold misses automatically.

**Running on 1× or 2× 24 GB GPUs and want more experts in VRAM?** Try [Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO) (`44.84 GiB` active model / `71.67 GiB` total).
**Want faster generation?** Pair with the 4-bit DFlash2 drafter: [Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ) (`0.52 GiB`).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
