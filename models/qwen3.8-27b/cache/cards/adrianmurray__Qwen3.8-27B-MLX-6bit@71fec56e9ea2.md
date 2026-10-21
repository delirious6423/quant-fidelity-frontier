---
language:
- en
- zh
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- quantized
- 6bit
- qwen
- reasoning
- coding
base_model: Qwen/Qwen3.8-27B
base_model_revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
---

# Qwen3.8-27B MLX 6-bit

This is a modified, 6-bit MLX quantization of Alibaba Cloud's official
`Qwen/Qwen3.8-27B` checkpoint. It is intended for local inference on Apple
Silicon and can be paired with the separately distributed native-MTP artifact
from the same source revision.

## Artifact identity

- Base model: `Qwen/Qwen3.8-27B`
- Base revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Quantization: affine 6-bit, group size 64
- Native context length: 262,144 tokens
- Weight shards: five MLX SafeTensors files
- Approximate on-disk size: 20 GB

The internal Transformers/MLX architecture identifiers retain the
`qwen3_5` name because Qwen3.8 uses the compatible architectural foundation.
Those implementation identifiers do not change the checkpoint identity.

## Reproduction

Using MLX LM 0.31.3 and MLX 0.32.0:

```bash
mlx_lm.convert \
  --hf-path /path/to/Qwen3.8-27B \
  --mlx-path /path/to/Qwen3.8-27B-MLX-6bit \
  --quantize --q-bits 6 --q-group-size 64 --q-mode affine
```

The source directory must correspond to the exact base revision above. Verify
the converted files against `SHA256SUMS` before publishing or loading them.

## Use with Qwen Prime Runtime

```bash
qwen-prime-runtime configure \
  --target /path/to/Qwen3.8-27B-MLX-6bit \
  --draft /path/to/Qwen3.8-27B-MTP-MLX-6bit
qwen-prime-runtime doctor
qwen-prime-runtime serve
```

## Performance

No throughput is guaranteed. On the development M4 Max, a warm 256-token
coding request measured approximately 26 server tokens/second with 53.9% draft
acceptance. A direct block-size sweep measured approximately 28 tokens/second
at block size four. Results vary with hardware, prompt length, cache state,
generation length, and acceptance ratio.

## License and attribution

The original Qwen3.8-27B model is Copyright 2026 Alibaba Cloud and is licensed
under Apache License 2.0. This directory includes the original license. The
weights and metadata in this repository were modified by conversion and
quantization for MLX; see `NOTICE` and `quantization_provenance.json`.

The Apache license does not grant trademark rights. This is an independent
derivative and is not endorsed by Alibaba Cloud or the Qwen team.
