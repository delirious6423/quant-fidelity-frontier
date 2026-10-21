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
- vision-language
- qwen3.8
---

# Qwen3.8-27B MLX MXFP8

Vision-enabled MLX conversion of `Qwen/Qwen3.8-27B`, pinned to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

- Language model: MXFP8, 8-bit, group size 32
- Vision tower: same-revision BF16 weights
- Runtime: `mlx-vlm` 0.6.3 or newer

The Hub's approximately 8.0B safetensors count reflects packed MXFP storage elements; the
underlying architecture remains the full 27B model.

## Use

```bash
python -m mlx_vlm.generate \
  --model Shiftedx/Qwen3.8-27B-MLX-MXFP8 \
  --image image.jpg \
  --prompt "Describe this image."
```

Text generation and three image-understanding smoke tests passed locally. Quantization can
still change behavior, so independently evaluate important use cases. Treat prompts, images,
and outputs as untrusted: do not submit secrets, and sandbox tools or generated code with
least-privilege access. This conversion adds no telemetry or remote execution.

The upstream Apache-2.0 license and model limitations continue to apply.
