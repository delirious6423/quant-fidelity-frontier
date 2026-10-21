---
language:
- en
- zh
library_name: transformers
license: other
license_name: glm-5.3
pipeline_tag: text-generation
tags:
- mxfp4
- fp8
- vllm
- llm-compressor
- compressed-tensors
base_model: zai-org/GLM-5.3
---

# GLM-5.3-MXFP4xFP8

## Model Overview
- **Model Architecture:** GlmMoeDsaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** Mixed (MXFP4/FP8)
  - **Activation quantization:** FP8
- **Release Date:** 2026-10-18
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), ready for inference with vLLM.

This is a mixed-precision quantization where individual layers are assigned MXFP4 or FP8 precision to achieve an average of ~4.5 bits per parameter while preserving accuracy on sensitive layers.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) to mixed MXFP4 and FP8 precision, ready for inference with vLLM.

Typical MXFP4 quantization pairs MXFP4 weights with MXFP4 activations, a combination that is most performant on Blackwell GPUs. In this model, the majority of the linear layers (routed experts and dense MLPs) instead use MXFP4 weights with FP8 activations quantized dynamically on a per-group basis, making it particularly optimized for inference on Hopper GPUs as of vLLM v0.32. The remaining linear layers (attention, shared experts, and the multi-token prediction layer) are kept at FP8 precision with block-wise weight scaling and FP8 dynamic activations.

This optimization reduces the number of bits per parameter from 16 to approximately 4.5 on average, reducing the disk size and GPU memory requirements by approximately 72% (from ~1507 GB to ~415 GB).

Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```
vllm serve RedHatAI/GLM-5.3-MXFP4xFP8 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 5 \
  --tool-call-parser glm47 \
  --reasoning-parser glm47 \
  --enable-auto-tool-choice
```

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a mixed-precision recipe: MXFP4 weight quantization with per-group FP8 dynamic activation quantization for the expert and dense MLP layers, and FP8 block-wise quantization for the attention, shared expert, and multi-token prediction layers. The quantized model was exported in the [compressed-tensors](https://github.com/vllm-project/compressed-tensors) format.

## Evaluation

### Accuracy

In Progress
