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
- hybrid-q8-q4
- qwen
- reasoning
- coding
base_model: Qwen/Qwen3.8-27B
base_model_revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
---

# Qwen3.8-27B MLX Hybrid Q8/Q4

This is a modified MLX quantization of Alibaba Cloud's official
`Qwen/Qwen3.8-27B` checkpoint for local inference on Apple Silicon. It is the
recommended target for Qwen Prime and pairs with the separately published
[`adrianmurray/Qwen3.8-27B-MTP-MLX-6bit`](https://huggingface.co/adrianmurray/Qwen3.8-27B-MTP-MLX-6bit)
native-MTP draft.

## Artifact identity

- Base model: `Qwen/Qwen3.8-27B`
- Base revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Full-attention Q/K/V/O projections and LM head: affine 8-bit, group size 64
- MLP and linear-attention projections: affine 4-bit, group size 64
- Embeddings and layer norms: Float16
- Quantized modules: 65 at 8-bit and 432 at 4-bit
- Effective density: approximately 5.48 bits per weight
- Weight shards: four MLX SafeTensors files
- Approximate resident size: 17.2 GB

The internal Transformers/MLX architecture identifiers retain the `qwen3_5`
name because Qwen3.8 uses that compatible architectural foundation. Those
implementation identifiers do not change the checkpoint identity.

## Use with Qwen Prime

Download this target and the matching draft, then configure Qwen Prime Runtime:

```bash
qwen-prime-runtime configure \
  --target /path/to/Qwen3.8-27B-Hybrid-Q8Q4 \
  --draft /path/to/Qwen3.8-27B-MTP-MLX-6bit
qwen-prime-runtime doctor
qwen-prime-runtime serve
```

Qwen Prime for macOS is available from
[`adriancmurray/QwenPrime`](https://github.com/adriancmurray/QwenPrime/releases/latest).
Model weights remain outside the application and are never replaced by an app
update.

## Reproduction

Using MLX 0.32.0 and MLX LM 0.31.3, convert the exact base revision with a
module predicate that leaves embeddings and norms unquantized, assigns 8-bit
affine quantization to full-attention Q/K/V/O projections and `lm_head`, and
assigns 4-bit affine quantization to MLP and remaining linear projections. The
global fallback is 4-bit affine with group size 64. The complete resulting
module map is stored in `config.json`; machine-readable source and conversion
metadata are in `quantization_provenance.json`.

Verify every artifact against `SHA256SUMS` before loading or redistributing it.

## Verification and performance

On an Apple M4 Max, the downloaded Qwen Prime 1.1.1 application identified this
artifact as mixed Q8/Q4 and enabled the fused MTP and adaptive verification
paths. A deterministic Swift 6 smoke task generated 88 tokens at 33.85 server
tokens/second with 59.1% draft acceptance; the generated program compiled with
Swift 6 and passed assertions for ordering, duplicate removal, and empty input.
The public uniform 6-bit target passed the same functional check at 21.34
tokens/second in a direct runtime invocation.

These are single-machine measurements, not guaranteed minimums. Performance
depends on hardware, prompt length, cache state, generation length, and draft
acceptance. This smoke comparison is not a substitute for a broad code-quality
benchmark.

## License and attribution

The original Qwen3.8-27B model is Copyright 2026 Alibaba Cloud and is licensed
under the Apache License 2.0. This directory includes the original license. The
weights and metadata were modified by conversion and mixed-bit quantization for
MLX; see `NOTICE` and `quantization_provenance.json`.

The Apache license does not grant trademark rights. This independent derivative
is not affiliated with or endorsed by Alibaba Cloud or the Qwen team.
