---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- mtplx
- mxfp4
- native-mtp
- vision-language
- qwen3.8
---

# Qwen3.8-27B MLX MXFP4 + Native MTP

Vision-enabled MTPLX package of `Qwen/Qwen3.8-27B`, pinned to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

- Language model: MXFP4, 4-bit, group size 32
- Vision tower: same-revision BF16 weights
- Speculation: 15 native BF16 MTP tensors; recommended depth 3
- Runtime: MTPLX 2.0.2 or newer

The Hub's approximately 5.5B safetensors count reflects packed MXFP storage elements; the
underlying architecture remains the full 27B model.

## Serve

```bash
mtplx quickstart \
  --model Shiftedx/Qwen3.8-27B-MLX-MXFP4-MTP \
  --mtp --depth 3 --profile sustained
```

The MTP tensor gate, text generation, and three image requests through the local
OpenAI-compatible API passed. Quantization and speculative decoding can still change behavior,
so independently evaluate important use cases. Treat prompts, images, and outputs as untrusted:
do not submit secrets, and sandbox tools or generated code with least-privilege access. This
package adds no telemetry or remote execution.

The upstream Apache-2.0 license and model limitations continue to apply.
