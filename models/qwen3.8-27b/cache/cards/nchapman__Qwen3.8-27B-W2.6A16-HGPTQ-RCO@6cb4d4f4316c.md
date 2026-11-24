---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- w2a16
- w4a16
- compressed-tensors
- vllm
- warp34
---

# Qwen3.8-27B-W2.6A16-HGPTQ-RCO

A compact sub-3-bit quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Qwen's flagship 27B multimodal model for coding, reasoning, instruction following, and vision. It brings a full 27B-class model—with vision and built-in speculative decoding intact—down to a single **16 GB GPU**.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (16 GB RTX 30/40-series GPUs (e.g., RTX 4060 Ti 16 GB / 4070 Ti Super / 4080)):

```bash
warp34 serve nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO --trust-remote-code
```

## About this quant

- **Fits on a single 16 GB GPU** (`12.20 GiB` on disk vs. `51.8 GiB` in BF16, `4.2×` smaller), leaving room for `128K` context in `warp34`, vision inputs, and speculative decoding.
- **Smart mixed 2-bit / 4-bit precision (V2):** Keeps the most sensitive `36.75%` of backbone weights—plus the vision encoder and MTP speculative drafter—in 4-bit (with an 8-bit output head), while compressing the rest with symmetric 2-bit quantization.
- **Strong real-world retention for its size:** Scores `82.87%` across IFEval (`78.2%`), MATH-500 (`81.4%`), and HumanEval+ (`89.0%`) vs. `86.90%` in full 16-bit.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses strictly `2 / 4 / 8`-bit GPU-friendly formats (no slow 3-bit unpacking) so it runs fast out of the box.

**Have a 24 GB GPU?** Use [Qwen3.8-27B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ) (`16.14 GiB`, highest quality).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
