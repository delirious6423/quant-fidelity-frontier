---
language:
  - en
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: onnxruntime-genai
tags:
  - amd
  - onnx
  - onnxruntime
  - onnxruntime-genai
  - multimodal
  - vision-language
  - qwen
  - moe
  - hip-ep
pipeline_tag: image-text-to-text
base_model:
  - Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next_gpuep_1.0_gpu_128

> **Source model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

## Introduction

This model was created using OGA Model Builder.

## Quantization Strategy

- Text decoder: RTN / Group 32 / FP16 activations / INT4 Weights (MoE experts: QMoE INT4, block size 32)
- Vision encoder: FP16
- Embedding model: FP16
- Engram: FP8

---

### License

This model is a quantized derivative of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and is distributed under the same license as the source model, the Qwen Community License Agreement 1.0. A copy of the license is included in the [LICENSE](LICENSE) file of this repository.

Modifications copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved. AMD has modified the original model files (quantization and conversion to ONNX) for optimized deployment on AMD hardware.

You may not use this file except in compliance with that license. See the [LICENSE](LICENSE) file for the specific language governing permissions and limitations.
