---
base_model: Qwen/Qwen3.8-27B
library_name: vllm
tags: [qda, quantization, compressed-tensors, w3a16]
---

# QDA Qwen3.8-27B W3A16

A W3A16 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (3-bit weights), made with QDA.
It is in compressed-tensors format and runs directly in vLLM.

## Accuracy of this checkpoint (%)

| Benchmark | Score |
|---|---|
| Math (AIME 2025 + HMMT Feb 2026) | 85.9 |
| GPQA-Diamond | 79.1 |
| LiveCodeBench v6 | 84.7 |
| IFBench | 66.8 |
| **Average** | **79.1** |

Evaluated with vLLM 0.27.1 on GH200, static batching, model-card sampling
(`{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}`), seeds 17+j.

## Usage

```bash
vllm serve qda-lab/QDA-Qwen3.8-27B-W3A16
```
