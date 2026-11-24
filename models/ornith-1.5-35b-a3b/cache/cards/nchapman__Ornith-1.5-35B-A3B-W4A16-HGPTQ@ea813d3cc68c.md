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
- w4a16
- moe
- compressed-tensors
- vllm
- warp34
---

# Ornith-1.5-35B-A3B-W4A16-HGPTQ

A 4-bit quantized version of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), Ornith AI's 35B Mixture-of-Experts self-improvement model (3B active parameters per token), offering fast MoE decode speeds with vision and built-in Multi-Token Prediction (MTP) intact.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (24 GB RTX 30/40-series GPUs (e.g., RTX 3090 / 4090)):

```bash
warp34 serve nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Fits on a single 24 GB GPU** (`18.78 GiB` on disk vs. `67.0 GiB` in BF16, `3.6×` smaller), leaving room for KV cache, vision inputs, and speculative decoding.
- **Full MoE + Vision + MTP coverage:** All 256 routed experts, shared experts, attention/GDN layers, vision encoder, and the MTP drafter are calibrated and quantized to 4-bit (with an 8-bit output head).
- **Near-BF16 fidelity:** Retains `96.4%` isolated vision top-1 agreement (`97.0%` on UI/OCR guard) and `91.4%` isolated MTP draft agreement.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Bundles pre-profiled hot-expert rankings (`static_hot_cache_rankings.json`) and standard `4-bit` / `8-bit` layouts for fast single-GPU MoE serving.

**On a 16 GB GPU?** Try [Ornith-1.5-35B-A3B-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Ornith-1.5-35B-A3B-W2.6A16-HGPTQ-RCO) (`13.01 GiB`).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
