---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
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
base_model:
- LiquidAI/LFM2.5-24B-A2B
pipeline_tag: text-generation
library_name: transformers
tags:
- text-generation-inference
- llama-cpp
- liquid
- lfm2.5
- edge
- moe
---

# **LiquidAI-LFM2.5-24B-A2B-GGUF**

> [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) is a highly efficient, 24-billion parameter Mixture of Experts (MoE) reasoning model developed by Liquid AI that activates only 2.45 billion parameters per token to deliver exceptional speed for private, on-device agentic workflows. Designed as a pure reasoning model that inherently "thinks" before answering, it features an expansive 131K context window and a 128K vocabulary that supports 16 languages. The model excels in complex tool-use and agentic tasks, outperforming larger competing models on benchmarks like LiveClaw and ToolSandbox thanks to specialized post-training that incorporates agentic reinforcement learning within popular agent harnesses. Furthermore, its highly optimized architecture enables industry-leading inference efficiency—achieving up to 184 tokens per second on edge devices like the Apple M5 Max and over 21,000 output tokens per second on high-end GPUs—making it a premier choice for deploying fast, capable, and resource-conscious local AI agents.

## Model Files

| File Name | Quant Type | File Size | File Link | Description |
|-----------|------------|-----------|-----------|-------------|
| LFM2.5-24B-A2B.BF16.gguf | BF16 | 48 GB | [Link](https://huggingface.co/prithivMLmods/LiquidAI-LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B.BF16.gguf) | Full BF16 weights. Highest quality, largest file size. |
| LFM2.5-24B-A2B.Q3_K_L.gguf | Q3_K_L | 12.4 GB | [Link](https://huggingface.co/prithivMLmods/LiquidAI-LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B.Q3_K_L.gguf) | Lower quality but usable, good for low RAM availability. |
| LFM2.5-24B-A2B.Q4_K_M.gguf | Q4_K_M | 14.5 GB | [Link](https://huggingface.co/prithivMLmods/LiquidAI-LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B.Q4_K_M.gguf) | Good quality, default size for most use cases, *recommended*. |
| LFM2.5-24B-A2B.Q5_K_M.gguf | Q5_K_M | 17 GB | [Link](https://huggingface.co/prithivMLmods/LiquidAI-LFM2.5-24B-A2B-GGUF/blob/main/LFM2.5-24B-A2B.Q5_K_M.gguf) | High quality, *recommended*. |

## llama.cpp

LLM inference in C/C++ — https://github.com/ggml-org/llama.cpp