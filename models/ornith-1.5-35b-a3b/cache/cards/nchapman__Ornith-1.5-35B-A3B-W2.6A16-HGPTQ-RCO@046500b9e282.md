---
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
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

# Ornith-1.5-35B-A3B-W2.6A16-HGPTQ-RCO

A compact sub-3-bit quantized version of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), Ornith AI's 35B-A3B Mixture-of-Experts self-improvement model (3B active parameters per token). It fits the full 256-expert multimodal model and built-in speculative draft head onto a single **16 GB GPU**.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (16 GB RTX 30/40-series GPUs):

```bash
warp34 serve nchapman/Ornith-1.5-35B-A3B-W2.6A16-HGPTQ-RCO
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Ornith-1.5-35B-A3B-W2.6A16-HGPTQ-RCO --trust-remote-code
```

## About this quant

- **Fits on a single 16 GB GPU** (`13.01 GiB` on disk vs. `67.0 GiB` in BF16, `5.15×` smaller), leaving ~`3 GiB` for KV cache, vision inputs, and speculative decoding.
- **Smart mixed 2-bit / 4-bit MoE precision (V2):** Keeps all always-active layers (attention, shared experts, vision, and MTP) plus the 12 most sensitive routed-expert layers in 4-bit, while compressing the remaining 28 routed-expert layers with symmetric 2-bit quantization.
- **Strong multimodal & speculative retention:** Preserves `96.7%` isolated vision top-1 agreement (`98.5%` on UI/OCR guard) and `91.2%` isolated MTP draft agreement.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Bundles pre-profiled hot-expert rankings (`static_hot_cache_rankings.json`) and strict `2 / 4 / 8`-bit layouts for fast single-GPU MoE serving.

**Have a 24 GB GPU?** See [Ornith-1.5-35B-A3B-W4A16-HGPTQ](https://huggingface.co/nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ) (`18.78 GiB`) for full 4-bit quality.

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
