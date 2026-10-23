---
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
library_name: llama-cpp
tags:
- qwen3.8
- qwen3.8-flash-next
- nvfp4
- moe
- quantized
- gguf
- llama-cpp
- multimodal
language:
- multilingual
---

# Qwen3.8-Flash-Next-NVFP4 - GGUF

NVFP4 GGUF quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), an experimental preview architecture underpinning Qwen4 with 125B total parameters (6B activated per token), plus 51B n-gram embedding.

This repository contains the **NVFP4 GGUF** conversion of the model, making it usable with [llama.cpp](https://github.com/ggerganov/llama.cpp) and other GGUF-compatible inference engines.

## Model Overview

Qwen3.8-Flash-Next is a multimodal causal language model with vision encoder built by the Qwen Team. It introduces architectural innovations including Hybrid Attention pairing Gated DeltaNet with Qwen Sparse Attention (QSA), Gated Residual connections, and N-gram Embedding for efficient parameter scaling with minimal active compute overhead.

| Property                      | Value                          |
| :---------------------------- | :----------------------------- |
| **Model Type**                | Causal Language Model with Vision Encoder |
| **Parameters**               | 125B total / 6B activated (+ 51B n-gram embedding, 4B MTP) |
| **Layers**                    | 48                             |
| **Hidden Dimension**          | 2560                           |
| **Hidden Layout**             | 12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE)) |
| **Gated DeltaNet**            | 48 Linear Attention Heads (V), 16 (QK), head dim 128 |
| **Qwen Sparse Attention**     | 24 Heads (Q), 2 (KV), head dim 256, RoPE dim 64 |
| **QSA Indexer**               | MQA (4 Q heads, 1 shared K head), head dim 128, budget 512 blocks (2048 tokens) |
| **Mixture of Experts**        | 512 experts, 10 routed + 1 shared, intermediate dim 640 |
| **Gated Residual**            | 4 branches, bottleneck rank 320 |
| **Context Length**            | 262,144 tokens natively (extensible up to 1,000,000 tokens) |
| **Vocabulary Size**           | 248,320 (padded)               |
| **N-gram Embedding**          | 20,000,000 (bigrams/trigrams at layer 2) |
| **Multi-Token Prediction**    | 1 layer, trained with multi-steps |

## GGUF Files

| File | Format | Size | Description |
| :--- | :--- | ---: | :--- |
| `qwen3.8-flash-next-nvfp4.gguf` | NVFP4 | 94G | NVFP4 quantized model |

> Note: N-gram embeddings are quantized using IQ4_NL

## Usage

### llama.cpp (CLI)

```bash
# Run text-only inference
./llama-cli \
  -m qwen3.8-flash-next-nvfp4.gguf \
  -p "Explain quantum computing in simple terms." \
  --temp 1.0 --top-k 20 --top-p 0.95
```

### llama-server (OpenAI-compatible API)

```bash
# Start OpenAI-compatible API server
./llama-server \
  -m qwen3.8-flash-next-nvfp4.gguf \
  --host [IP_ADDRESS] --port 8080
```

## Generation Parameters

Recommended parameters from the model's generation configuration:

| Parameter       | Value    |
| :-------------- | :------- |
| Temperature     | 1.0      |
| Top-K           | 20       |
| Top-P           | 0.95     |

### Recommended Sampling by Mode

| Mode | Temperature | Top-P | Top-K | Presence Penalty | Repetition Penalty |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Thinking (general) | 1.0 | 0.95 | 20 | 1.5 | 1.0 |
| Thinking (coding/WebDev) | 0.6 | 0.95 | 20 | 0.0 | 1.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 1.5 | 1.0 |

## Thinking Mode

The model operates in **thinking mode by default**, generating reasoning traces enclosed in `<think>\n...\n</think>\n\n` prior to the final response.

- **Enable (default):** Generates step-by-step thinking content before answering.
- **Disable:** Set `enable_thinking` to `false` in your chat template / inference configuration to output direct responses.
- **Preserve Thinking:** Retain thinking traces across multi-turn sessions by setting `preserve_thinking` to `true`, improving consistency in long-horizon agentic workflows.

## Key Features

- **Hybrid Attention with QSA:** Reworks Gated DeltaNet and Gated Attention into Gated DeltaNet and Qwen Sparse Attention (QSA) operating at the micro-block level, substantially cutting long-context latency.
- **Gated Residual:** Widened residual streams modulated via element-wise data-dependent read gates and per-branch scalar write gates for finer expressiveness and training stability.
- **N-gram Embedding:** Indexes short n-grams for parameter scaling that requires less compute and is friendly to memory-constrained accelerators.
- **Mixture of Experts:** 512 experts with 10 routed + 1 shared expert activated per token.
- **Ultra-Long Context:** 262,144 tokens natively supported and extensible up to 1,000,000 tokens.

## Acknowledgements

- Original model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Blog post: [Qwen3.8-Flash-Next](https://qwen.ai/blog?id=qwen3.8-flash-next)

## License

[Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
