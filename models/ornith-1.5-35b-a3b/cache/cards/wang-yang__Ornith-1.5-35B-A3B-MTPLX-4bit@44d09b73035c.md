---
license: mit
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
language:
  - en
library_name: mlx
tags:
  - mlx
  - mtplx
  - mtp
  - multi-token-prediction
  - speculative-decoding
  - qwen35moe
  - moe
  - 4-bit
pipeline_tag: text-generation
---

# Ornith-1.5-35B-A3B-MTPLX

A **4-bit MLX quantization** of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), packaged for [MTPLX](https://github.com/youssofal/MTPLX) native multi-token speculative decoding on Apple Silicon.

Ornith-1.5-35B-A3B is a 35B-parameter mixture-of-experts model activating approximately 3B parameters per token. This release preserves its native one-layer MTP head and includes the vision tower.

## Quantization

- **Main model:** 4-bit affine, group size 64
- **MTP sidecar:** 4-bit affine, group size 64 (`mtp.safetensors`)
- **MTP norms:** BF16 (non-quantized 1-D tensors)
- **Vision tower:** BF16 (`model-vision.safetensors`)
- **Architecture:** `Qwen3_5MoeForConditionalGeneration`
- **MTP contract:** Qwen3-Next / delta norm encoding

The MTP sidecar contains 778 quantized matrix weights and 7 BF16 normalization tensors. It is discovered automatically from the model directory.

## Usage with MTPLX

```bash
mtplx start cli \
  --model <path-to-this-model>

# Optional: benchmark/tune the MTP draft depth
mtplx tune --model <path-to-this-model> --retune
```

The model can also be loaded by MLX-LM for autoregressive text generation, without MTPLX speculative decoding.

## Provenance and licensing

- **Base model:** [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — MIT
- **Runtime:** [MTPLX](https://github.com/youssofal/MTPLX) — Apache-2.0
- **Conversion:** MLX affine quantization, group size 64

This repository contains a quantized derivative of the base model. Please follow the upstream model license and usage conditions. MTPLX attribution is required when shipping a product or service built on MTPLX:

> Powered by MTPLX  
> https://github.com/youssofal/MTPLX
