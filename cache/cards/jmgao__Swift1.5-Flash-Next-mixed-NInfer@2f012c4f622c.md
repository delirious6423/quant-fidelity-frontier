---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
- ukisai/Swift1.5-Qwen3.8-Flash-Next
- primitive-ai/Qwen3.8-Flash-Next-PLE-quant
base_model_relation: quantized
tags:
- ninfer
- qwen3.8
- flash-next
- nvfp4
- fp8
- int4
- mixed-precision
- blackwell
- rtx-pro-6000
- windows
- single-gpu
- multimodal
- speculative-decoding
---
# Swift 1.5 Qwen3.8-Flash-Next mixed quantization for NInfer

A quantization of ukisai/Swift1.5-Qwen3.8-Flash-Next using primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8's methodology to run on igorls's [NInfer workstation fork](https://github.com/igorls/ninfer) on a single RTX Pro 6000.

The quantized PLE tables in Swift1.5 are identical to the base model, so we take ours from primitive-ai/Qwen3.8-Flash-Next-PLE-quant.

This slightly differs from igorls/Qwen3.8-Flash-Next-mixed-NInfer in that the MTP expert banks are stored in NVFP4 directly instead of BF16 and quantized at runtime.