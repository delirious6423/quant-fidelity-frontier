---
base_model: Qwen/Qwen3.8-27B
library_name: vllm
tags: [qda, quantization, compressed-tensors, w4a8]
---

# QDA Qwen3.8-27B W4A8

A W4A8 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (4-bit weights and 8-bit (FP8) activations), made with QDA.
It is in compressed-tensors format and runs directly in vLLM.

**Hardware:** NVIDIA Hopper GPUs (H100/H200). vLLM's W4A8 kernels run only on Hopper; on other GPUs, use the W4A16 version.

## Accuracy of this checkpoint (%)

| Benchmark | Score |
|---|---|
| Math (AIME 2025 + HMMT Feb 2026) | 81.8 |
| GPQA-Diamond | 80.3 |
| LiveCodeBench v6 | 86.0 |
| IFBench | 68.3 |
| **Average** | **79.1** |

Evaluated with vLLM 0.27.1 on GH200, static batching, model-card sampling
(`{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}`), seeds 17+j.

## Usage

```bash
vllm serve qda-lab/QDA-Qwen3.8-27B-W4A8
```
