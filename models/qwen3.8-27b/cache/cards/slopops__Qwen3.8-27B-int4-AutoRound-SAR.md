---
language:
- en
- zh
- es
- fr
- ja
license: apache-2.0
library_name: transformers
tags:
- text-generation
- conversational
- quantization
- autoround
- qwen
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B # Update this to the exact original Hugging Face repo if different
---

# slopops/Qwen3.8-27B-int4-AutoRound-SAR

This repository contains an **int4 quantized** version of **Qwen3.8-27B** using **AutoRound** with specific SAR (Spark AutoRound with OpenCode-Instruct) settings. 

> **⚠️ IMPORTANT DISCLAIMER:** This repository contains a quantized version of the model. **No claims are made regarding the performance, accuracy, or benchmark results of this specific quantization.** The model details, architecture descriptions, and benchmark scores provided below are inherited directly from the **original base model's** documentation and represent the performance of the full-precision (BF16/FP16) base model, not this 4-bit quantized version.

## 1. AutoRound Quantization Details

This model was quantized using the [Spark AutoRound](https://github.com/whpthomas/spark-auto-round) algorithm. This is an optimally pre-configured int4 AutoRound quantization that achieves near fp8 quality with int4 speed.

The following specific hyperparameters were used during quantization:

```bash
--batch_size 8
--nsamples 512
--seqlen 2048
--group_size 64
--iters 1000
--dataset opencode-instruct
```

- **Precision:** 4-bit (int4)
- **Group Size:** 64 (A smaller group size often preserves more accuracy in AutoRound compared to standard 128 group sizes)
- **Calibration Dataset:** `opencode-instruct` (512 samples, 2048 sequence length)
- **Iterations:** 1000

## 2. Model Information (Base Model)

*The following information is inherited from the base Qwen3.8-27B model.*

### Introduction
Built on the architectural foundation of Qwen3.5, Qwen3.8 delivers substantial gains across coding, professional work, research, and long-horizon agentic tasks. Qwen3.8-27B brings these advances to a compact, deployment-friendly dense model: a native vision-language model that understands images and videos, with flexible thinking control, designed to carry complex, multi-step tasks through to completion with greater reliability.

### Model Overview

- Type: Causal Language Model with Vision Encoder
- Training Stage: Pre-training & Post-training
- Language Model
    - Number of Parameters: 27B
    - Hidden Dimension: 5120
    - Token Embedding: 248,320 (Padded)
    - Number of Layers: 64
    - Hidden Layout: 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))
    - Gated DeltaNet:
        - Number of Linear Attention Heads: 48 for V and 16 for QK
        - Head Dimension: 128
    - Gated Attention:
        - Number of Attention Heads: 24 for Q and 4 for KV
        - Head Dimension: 256
        - Rotary Position Embedding Dimension: 64
    - Feed Forward Network:
        - Intermediate Dimension: 17,408
    - LM Output: 248,320 (Padded)
    - MTP (Multi-Token Prediction): trained with multiple steps
- Context Length: 262,144 natively and extensible up to 1,000,000 tokens.

### Qwen3.8 Highlights

Qwen3.8-27B features the following enhancements:
- **Core Capabilities**: Comprehensive improvements across coding, professional work, research, and long-horizon agentic tasks.
- **Agent Execution**: Stronger autonomous planning and better handling of environment feedback, leading to more reliable end-to-end task completion.
- **Downstream Compatibility**: Broader support for popular harnesses and development tools, making it easier to integrate into your existing stack.
- **Flexible Thinking Control**: Thinking mode is on by default and can be disabled per request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from historical messages is retained via `preserve_thinking`.
- **Vision-Language Understanding**: Native support for image and video understanding, from STEM diagrams and documents to hour-scale videos.

## 3. How to Use

### vLLM Command

```bash
vllm serve slopops/Qwen3.8-27B-int4-AutoRound-SAR \
  --served-model-name Qwen3.8-27B-int4-AutoRound-SAR \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.55 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --optimization-level 3 \
  --performance-mode throughput \
  --load-format auto \
  --attention-backend flashinfer \
  --moe-backend auto \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8 \
  --chat-template chat_template.jinja \
  --default-chat-template-kwargs '{"preserve_thinking":true}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":-1,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
```

### Chat Template
This model utilizes the [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates). 

This is a universal drop-in Jinja template that fixes rendering errors, KV cache invalidation, token waste, empty think poisoning, and fatal agentic stalling across official Qwen chat templates.

It works across LM Studio, llama.cpp, vLLM, MLX, oMLX, KoboldCPP, and any engine that supports Hugging Face Jinja templates. You only need the single `chat_template.jinja` file at the root of this repository for all Qwen 3.5, 3.6, and 3.8 model sizes.

### Hardware & Deployment Notes (DGX Spark)
This model has been extensively tested and optimized for the **NVIDIA DGX Spark** (GB10 / sm_121 architecture). 

When deploying on DGX Spark hardware, it is highly recommended to use the experimental **`b12x` vLLM Docker image** (`vllm-node-b12x`) from the [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) repository. This custom build includes specialized high-performance B12X kernels specifically tuned for the Spark's unified memory architecture. 

*Tip for Spark users:* Change `--load-format auto` to `--load-format instanttensor` in the command above to significantly reduce model loading times and lower peak RAM usage during startup.

## 4. Limitations

While the base model is highly capable, applying a 4-bit quantization (especially on a 27B model) can introduce slight degradation in:
- Highly complex multi-step logical reasoning.
- Rare knowledge retrieval.
- Exact numerical calculations.

Users should be aware that quantized models may occasionally exhibit a slight drop in coherence or an increase in hallucinations compared to their full-precision counterparts.

## 5. License

This quantization is released under the same license as the original Qwen models. Please refer to the [original model's repository](https://huggingface.co/Qwen) for the exact licensing terms. Ensure you comply with the original model's acceptable use policy when deploying this quantized version.
