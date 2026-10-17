---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- mlx-vlm
- mxfp8
- abliterated
- vision-language
- qwen3.8
---

# Qwen3.8-27B Abliterated MLX MXFP8

Vision-enabled MLX conversion of `Qwen/Qwen3.8-27B`, pinned to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, with a measured refusal-direction edit.

- Language model: MXFP8, group size 32
- Vision tower: unchanged same-revision BF16 weights
- Edit: completion-projected global direction, strength 1.5, norm-preserving, 128 modules
- Runtime: `mlx-vlm` 0.6.3 or newer

On a frozen 12-case held-out suite, measured refusal markers changed from 12/12 in the exact
quantized parent to 0/12, while benign-sensitive refusal stayed 0/12 and utility stayed 6/6.
This is scoped evidence, not a claim that the model is universally “uncensored.”

The Hub's approximately 8.0B safetensors count reflects packed MXFP storage elements; the
underlying architecture remains the full 27B model.

```bash
python -m mlx_vlm.generate \
  --model Shiftedx/Qwen3.8-27B-Abliterated-MLX-MXFP8 \
  --image image.jpg --prompt "Describe this image."
```

This edit reduces refusal behavior and may increase harmful or incorrect outputs. Treat prompts,
images, and outputs as untrusted; do not submit secrets, and sandbox tools or generated code with
least-privilege access. This conversion adds no telemetry or remote execution.

The upstream Apache-2.0 license and model limitations continue to apply.
