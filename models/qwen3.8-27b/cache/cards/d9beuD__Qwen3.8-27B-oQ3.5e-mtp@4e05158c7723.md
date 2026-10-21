---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
tags:
- mlx
- oq
- quantized
---

> **⚙️ Chat template update (2026-09-14)** — The tool-calling chat template has been replaced with the community-fixed [froggeric v22.5](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) template, which fixes several tool-calling bugs (serialized JSON arguments, premature stops mid tool-call, `reasoning_effort=high`). <!-- froggeric-toolfix -->


# Qwen3.8-27B-oQ3.5e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.6.0.dev1) mixed-precision quantization.

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 3
- **Group size**: 64
- **Format**: MLX safetensors
