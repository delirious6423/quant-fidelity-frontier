---

library_name: transformers
pipeline_tag: text-generation
tags:
- nvfp4
- fp4
- fp8
- vllm
- llm-compressor
- compressed-tensors
base_model: deepseek-ai/DeepSeek-V4-Flash-0731

---

# DeepSeek-V4-Flash-0731-NVFP4-FP8-BLOCK

## Model Overview

* **Model Architecture:** DeepseekV4ForCausalLM

  * **Input:** Text
  * **Output:** Text
* **Model Optimizations:**

  * **Routed expert weight quantization:** NVFP4
  * **Routed expert activation quantization:** NVFP4
  * **Attention/shared expert weight quantization:** FP8 block quantization
  * **Attention/shared expert activation quantization:** FP8
* **Release Date:** 2026-09-23
* **Version:** 1.0
* **Model Developers:** RedHatAI

This model is a mixed-precision quantized version of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

It was evaluated on GPQA to assess its quality in comparison to the base model.

### Model Optimizations

This model uses mixed-precision NVFP4 and FP8 quantization for efficient inference with vLLM.

The routed MoE expert linear layers use NVFP4 weights and activations with a group size of 16. Selected attention projections, the attention indexer `q_b_proj`, and shared expert layers use FP8 weights with 128x128 block quantization and dynamically quantized FP8 activations.

Layers outside these quantization targets retain their original precision.

The resulting checkpoint is exported using [compressed-tensors](https://github.com/vllm-project/compressed-tensors) and is ready for inference with vLLM.

The NVFP4 expert quantization was applied using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```bash
vllm serve RedHatAI/DeepSeek-V4-Flash-0731-NVFP4-FP8-BLOCK \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --served-model-name dsv4_nvfp4_fp8_gptq \
    --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4
```

## Creation

This model was created using [LLM Compressor](https://github.com/vllm-project/llm-compressor) with mixed-precision NVFP4 and FP8 quantization.

Routed MoE experts use NVFP4 weight and activation quantization, while selected attention, indexer, and shared expert linear operators use FP8 quantization. The checkpoint is exported in the compressed-tensors format for optimized deployment with vLLM.

## Evaluations

| Model                                               |       GPQA |
| --------------------------------------------------- | ---------: |
| deepseek-ai/DeepSeek-V4-Flash-0731                  |     84.34% |
| **RedHatAI/DeepSeek-V4-Flash-0731-NVFP4-FP8-BLOCK** | **86.15%** |

Note that GPQA is a non-deterministic eval on its own and quantization can sometimes act as a regularizer for individual evals.
