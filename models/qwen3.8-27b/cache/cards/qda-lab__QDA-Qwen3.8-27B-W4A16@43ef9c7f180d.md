---
base_model: Qwen/Qwen3.8-27B
library_name: vllm
tags: [qda, quantization, compressed-tensors, w4a16]
---

# QDA Qwen3.8-27B W4A16

W4A16 checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) found by a QDA search
("QDA: Closed-Loop Quantization Design with LLM Agents", MLSys 2027 submission).
compressed-tensors format; loads in vLLM as is. These are the exact files the paper evaluated
(search `r-01097832`, hand-in `h5`).

## Accuracy of this checkpoint (%)

| Benchmark | Score |
|---|---|
| Math (AIME 2025 + HMMT Feb 2026) | 82.7 |
| GPQA-Diamond | 81.9 |
| LiveCodeBench v6 | 86.3 |
| IFBench | 68.2 |
| **Average** | **79.8** |

Evaluated with vLLM 0.27.1 on GH200, static batching, model-card sampling
(`{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}`), seeds 17+j.

## Usage

```bash
vllm serve qda-lab/QDA-Qwen3.8-27B-W4A16
```
