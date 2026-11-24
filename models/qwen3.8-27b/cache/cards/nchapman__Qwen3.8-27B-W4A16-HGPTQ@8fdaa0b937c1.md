---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- w4a16
- compressed-tensors
- vllm
- warp34
---

# Qwen3.8-27B-W4A16-HGPTQ

A 4-bit quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Qwen's flagship 27B dense multimodal model for coding, reasoning, instruction following, and vision tasks. It uses Qwen's hybrid attention architecture for fast long-context performance and includes built-in Multi-Token Prediction (MTP) for speculative decoding.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (24 GB RTX 30/40-series GPUs (e.g., RTX 3090 / 4090)):

```bash
warp34 serve nchapman/Qwen3.8-27B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.8-27B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Fits on a single 24 GB GPU** (`16.14 GiB` on disk vs. `51.75 GiB` in BF16, `3.2×` smaller), leaving room for `128K–256K` context in `warp34`, vision inputs, and speculative decoding.
- **Everything is calibrated, not just text.** Unlike 4-bit quants that strip or leave the vision tower and speculative draft head uncalibrated, this checkpoint calibrates the text backbone, vision encoder, and MTP drafter together (with an 8-bit output head).
- **Near-lossless quality:** Scores `85.99%` across IFEval (`81.9%`), MATH-500 (`82.8%`), and HumanEval+ (`93.3%`) vs. `86.90%` for the full 16-bit model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses standard `4-bit` / `8-bit` packed weights with Hadamard rotation so `warp34` runs full-speed CUDA kernels with zero extra flags.

**Have a smaller GPU?** Try [Qwen3.8-27B-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO) (`12.20 GiB`, fits on 16 GB).
**Want faster generation?** Pair with the 4-bit DFlash2 drafter: [Qwen3.8-27B-DFlash2-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
