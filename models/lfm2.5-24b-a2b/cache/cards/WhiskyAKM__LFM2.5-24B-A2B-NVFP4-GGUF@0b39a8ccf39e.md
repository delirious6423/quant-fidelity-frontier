---
pipeline_tag: text-generation
base_model: LiquidAI/LFM2.5-24B-A2B
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: llama-cpp
tags:
- liquid
- lfm2.5
- edge
- moe
- gguf
- nvfp4
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
---

# LFM2.5-24B-A2B NVFP4 - GGUF

NVFP4 GGUF conversions derived from the **LFM2.5-24B-A2B** model, providing highly efficient 4-bit floating-point quantization for reasoning and agentic tasks.

This repository provides **NVFP4** (NVIDIA 4-bit floating-point) GGUF files, making the model usable with [llama.cpp](https://github.com/ggerganov/llama.cpp) and other GGUF-compatible inference engines.

## Model Overview

**LFM2.5-24B-A2B** is a 24B-parameter reasoning model with only 2B active parameters, built for **fast, private agents on-device**. It is the largest and final model in the LFM2.5 family, delivering best-in-class agentic and tool-use performance with MoE speed.

- **Best-in-class agent**: Competitive with Qwen3.6-35B-A3B, a model with ~50% more total and active parameters, on agentic and tool-use tasks, scoring 48.09 on LiveClaw versus 38.19.
- **Reasoning at MoE speed**: Always thinks before it answers. With only 2B active parameters per token, each reasoning token is cheap.
- **Efficient inference**: Runs up to 1.7× faster than comparable MoEs on-device.

Find more information about LFM2.5-24B-A2B in our [blog post](https://www.liquid.ai/blog/lfm2-5-24b-a2b).

## Model Architecture

| Property                 | Value                                                              |
| :----------------------- | :----------------------------------------------------------------- |
| **Architecture**         | LFM2.5 (Mixture of Experts)                                        |
| **Total Parameters**     | 24.0B                                                              |
| **Active Parameters**    | ~2.45B                                                             |
| **Number of Layers**     | 40 (30 double-gated short convolution blocks + 10 GQA)             |
| **Experts**              | 64 per MoE block, top-4 routing                                    |
| **Training Budget**      | 36 trillion tokens                                                 |
| **Vocabulary Size**      | 128,000                                                            |
| **Context Length**       | 131,072 tokens                                                     |
| **Languages**            | English, Arabic, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Spanish, Vietnamese, Thai, Indonesian, Hindi, Russian, Polish |

## GGUF Files

| File                              | Format | Description                                      |
| :-------------------------------- | :----- | :---------------------------------------------- |
| `lfm2.5-24b-a2b-nvfp4.gguf`       | NVFP4  | NVFP4 quantized model - optimized for efficiency |

> **About NVFP4:** NVFP4 is NVIDIA's 4-bit floating-point format (E2M1). It offers significant memory savings while preserving quality close to bfloat16, and is accelerated on NVIDIA Blackwell GPUs via llama.cpp.

## Usage

### llama.cpp (CLI)

```bash
# Text generation / Reasoning inference
./llama-cli \
  -m lfm2.5-24b-a2b-nvfp4.gguf \
  -p "Solve the following problem step by step..." \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05
```

### llama-server (OpenAI-compatible API)

```bash
# API server
./llama-server \
  -m lfm2.5-24b-a2b-nvfp4.gguf \
  --host [IP_ADDRESS] --port 8080
```

## Generation Parameters

Recommended parameters for LFM2.5-24B-A2B:

| Parameter          | Value |
| :----------------- | :---- |
| Temperature        | 0.1   |
| Top-K              | 50    |
| Repetition Penalty | 1.05  |

## Key Features

- **Reasoning at MoE Speed:** Thinks before answering with low active parameter count per token.
- **Agent-Optimized:** Exceptional performance on tool-use and agentic tasks.
- **On-Device Ready:** Optimized for high throughput and memory efficiency.

## License

[lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE)
