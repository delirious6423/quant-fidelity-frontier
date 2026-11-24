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
- w2a16
- w4a16
- moe
- compressed-tensors
- vllm
- warp34
---

# Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO

A compact sub-3-bit quantized version of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), Qwen's flagship 180B-class Mixture-of-Experts multimodal model. Because only a small slice of experts activates per token, it delivers frontier-class coding and reasoning at fast decode speeds—now shrunk from `335 GiB` down to `71.67 GiB` total (`44.84 GiB` active model + `26.82 GiB` memory-mapped lookup table).

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (1× or 2× 24 GB RTX 30/40-series GPUs (`--tp 2` recommended on 2× RTX 3090 / 4090)):

```bash
warp34 serve nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO --tp 2
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO --trust-remote-code
```

## About this quant

- **Runs a 180B-class model on consumer GPUs:** `warp34` memory-maps the `26.82 GiB` n-gram table from host RAM/SSD, pins the hottest experts in VRAM, and streams cold misses on demand—fitting cleanly on `2× 24 GB` GPUs (`~21 GiB/GPU` with `128K` context) or `1× 24 GB` with RAM offloading.
- **Smart mixed 2-bit / 4-bit MoE precision (V2):** Keeps 100% of always-active layers (attention, shared experts, vision, and MTP) plus the 15 most sensitive routed-expert layers in 4-bit, while compressing the remaining 33 routed-expert layers with symmetric 2-bit quantization.
- **High fidelity for `4.68×` compression:** Retains `96.4%` isolated vision top-1 agreement (`97.0%` on UI/OCR guard) and `95.1%` isolated MTP draft agreement, with `37.7%` lower held-out divergence than V1.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Bundles pre-profiled hot-expert rankings (`static_hot_cache_rankings.json`) and strict `2 / 4 / 8`-bit layouts tuned specifically for `warp34`'s tiered MoE engine.

**Want even faster generation?** Pair with the 4-bit DFlash2 drafter: [Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ) (`0.52 GiB`) using `--draft-model`.
**Want maximum 4-bit fidelity (80 GB+ VRAM or tiered 2× 24 GB)?** See [Qwen3.8-Flash-Next-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ) (`92.84 GiB`).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
