---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- llama.cpp
- quantized
---

# Qwen3.8-27B - Q4_0 (GGUF)

A GGUF Q4_0 quantization of Qwen/Qwen3.8-27B (https://huggingface.co/Qwen/Qwen3.8-27B), for use with llama.cpp (https://github.com/ggml-org/llama.cpp) and llama.cpp-compatible runtimes.

- Format: Q4_0 (legacy block quantization, ~4.5 bits/weight effective, including per-block scales)
- File size: ~15 GB (down from ~54 GB BF16)
- Base architecture: Qwen3_5MoeForConditionalGeneration (hybrid linear/full attention)

## Quality validation

Validated against the original BF16 weights via llama.cpp's llama-perplexity --kl-divergence tool, over 580 chunks (n_ctx=512):

| Metric | Value |
|---|---|
| KL divergence (quantized vs. BF16) | 0.0400 +/- 0.0004 |
| Top-1 token agreement (same top-p) | 91.4% +/- 0.07% |
| Perplexity, BF16 original | 6.9508 +/- 0.0449 |
| Perplexity, Q4_0 | 7.0892 +/- 0.0460 |
| Mean PPL(Q)/PPL(base) | 1.0199 (approx 2.0% relative increase) |

## Usage

Run with llama-server:

    llama-server -m Qwen3.8-27B-Q4_0.gguf -c 32768 --port 8080

Or with llama-cli for direct generation. Any llama.cpp-compatible frontend (llama-swap, LM Studio, Ollama via GGUF import, etc.) should load this directly.

## Original model card

For architecture details, benchmark numbers, and usage guidance for the base model, see Qwen/Qwen3.8-27B (https://huggingface.co/Qwen/Qwen3.8-27B).

## License

Apache 2.0, inherited from the base model.