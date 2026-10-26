---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
language:
- en
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- qwen4_exp
- llama-cpp
- llama-cli
- rtx-pro-6000
- nvfp4
- blackwell
- SM120
- GB202
---

# **Qwen3.8-Flash-Next-NVFP4-GGUF**

> **Qwen3.8-Flash-Next** is an experimental open-weight preview from the Qwen team, exploring an architecture intended to underpin Qwen4. It is a 125B-parameter vision-language model with 6B active parameters per token, 51B n-gram embedding parameters, a 4B Multi-Token Prediction layer, and a 262K-token context window extendable to 1M via YaRN. Its architecture combines Gated DeltaNet, Qwen Sparse Attention, Gated Residual, and a 512-expert MoE design. Thinking is enabled by default, with configurable reasoning effort. It leads across several benchmarks, scoring 58.7 on DeepSWE 1.1, 62.5 on SWE-bench Pro, 81.0 on SWE-bench Multilingual, 73.5 on Toolathlon Verified, 91.7 on GPQA-Diamond, and 84.5 on AndroidWorld. However, DeepSeek-V4-Flash leads on NL2Repo-Bench, while Claude Opus 4.6 leads on HLE. It supports Transformers, vLLM, SGLang, and TokenSpeed and is released under the `qwen-community-1.0` license.

> [!NOTE]
This GGUF quantization is experimental.

## Model Files

| File Name | Quant Type | File Size | File Link | Description |
|-----------|------------|-----------|-----------|-------------|
| Qwen3.8-Flash-Next-NVFP4.gguf | NVFP4 | 186 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-NVFP4-GGUF/blob/main/Qwen3.8-Flash-Next-NVFP4.gguf) | NVFP4 quantization. Highly optimized for specific hardware accelerators, offering a strong balance of size and performance. |
| Qwen3.8-Flash-Next-NVFP4.mmproj.gguf | mmproj | 908 MB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-Flash-Next-NVFP4-GGUF/blob/main/Qwen3.8-Flash-Next-NVFP4.mmproj.gguf) | Multimodal projection file. Used for vision/language models. |

## llama.cpp

LLM inference in C/C++ — https://github.com/ggml-org/llama.cpp