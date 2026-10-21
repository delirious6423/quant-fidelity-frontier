---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx-node
pipeline_tag: image-text-to-text
tags:
  - mlx
  - mlx-node
  - quantized
  - mxfp4
  - mxfp8
  - mixed-precision
  - apple-silicon
  - qwen3.8
  - qwen3_5
  - vision-language
  - coding
  - unsloth
model_type: qwen3_5
quantized_by: mlx-node
---

# Qwen3.8-27B — Unsloth MXFP4 + MXFP8 (MLX)

[`Brooooooklyn/Qwen3.8-27B-MXFP4-mlx`](https://huggingface.co/Brooooooklyn/Qwen3.8-27B-MXFP4-mlx)
is an Apple Silicon MLX quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). The model has a
64-layer dense Qwen3.5-family language backbone with 48 linear-attention and 16
full-attention layers, a BF16 vision tower, and one preserved MTP layer.

This model is part of the
[Unsloth NVFP4 Tensor-Class Recipe for MLX — macOS + DGX collection](https://huggingface.co/collections/Brooooooklyn/unsloth-nvfp4-tensor-class-recipe-for-mlx-macos-dgx-6a5e3a893ae031d023e72ccf).

The source was the all-BF16 checkpoint at revision
[`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/commit/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).

## Update — re-converted 2026-09-01

The MXFP4 and MXFP8 weights are re-encoded. mlx-node now picks each MX block's
shared E8M0 exponent itself instead of rounding `log2(amax)` to nearest: MXFP4
tries both candidate exponents and keeps the lower squared error, and MXFP8 takes
the ceiling so a block can never saturate. Measured MXFP8 weight error across a
checkpoint drops from 6.91% to 2.66%.

Weight bytes changed; the tensor-class map, shard layout and tensor counts are unchanged.
Earlier revisions remain in this repository's history.

## Quantization recipe

This is a data-free, **weight-only** Apple translation of the
[Unsloth Qwen3.6 NVFP4 recipe](https://unsloth.ai/docs/models/qwen3.6#nvfp4),
applied to Qwen3.8's matching dense Qwen3.5-family tensor layout:

- tensors assigned NVFP4 by the recipe are stored as **MXFP4**, 4-bit with
  group size 32;
- tensors assigned FP8 by the recipe are stored as **MXFP8**, 8-bit with group
  size 32;
- every excluded tensor stays BF16.

Both MXFP classes use MLX weight-only quantized matmul with BF16/A16
activations. This ports the recipe's tensor-class selection; it does not claim
numerical or performance parity with NVIDIA NVFP4, calibrated W4A4/W8A8
execution, or FP8 KV-cache quantization.

No imatrix, calibration dataset, AWQ-style pre-scaling, activation
calibration, NVFP4 global scale, or FP8 KV-cache calibration was used.

| Tensor class | Stored format |
|---|---|
| Dense FFN `{gate,up,down}_proj`, layers 0–55 | MXFP4 4/32 |
| Dense FFN `{gate,up,down}_proj`, layers 56–63 | MXFP8 8/32 |
| Full-attention `{q,k,v,o}_proj` | MXFP8 8/32 |
| Linear-attention `in_proj_qkv`, `in_proj_z`, `out_proj` | MXFP8 8/32 |
| `lm_head` | MXFP8 8/32 |
| Embeddings; `in_proj_a/b`; GDN state, convolution, and norm tensors; all other norms | BF16 |
| Entire 15-tensor `mtp.*` subtree | BF16 |
| Vision tower and merger tensors | BF16 |

The allocation contains **168 MXFP4 modules** and **233 MXFP8 modules**. The
final eight dense FFNs intentionally use the higher class. The MTP subtree is
kept inline in the main shards; mlx-node detects it at load and enables native
MTP speculative decoding by default unless `enableMtp: false` is requested.

## Compatibility and usage

This checkpoint requires `@mlx-node/lm` and `@mlx-node/core` 0.0.10 or a newer
release that supports the same Qwen3.5-family MXFP and inline-MTP formats.

```bash
npm install @mlx-node/lm@^0.0.10 @mlx-node/core@^0.0.10
```

```typescript
import { loadSession } from '@mlx-node/lm';

const session = await loadSession('./Qwen3.8-27B-MXFP4-mlx');
const result = await session.send('Explain the purpose of a unit test in one sentence.');
console.log(result.text);
```

## Reproduction

Converter checkout:
[`mlx-node e281f0bb`](https://github.com/mlx-node/mlx-node/tree/e281f0bbc61eb71a244ad6681ffad324a1177780)
(package version 0.0.10). The invocation was:

```bash
mlx convert \
  --input /Users/brooklyn/.mlx-node/models/qwen3.8-27b \
  --output /Users/brooklyn/.mlx-node/models/qwen3.8-27b-unsloth-mxfp4-mlx \
  --model-type qwen3_5 \
  --dtype bfloat16 \
  --quantize \
  --q-recipe unsloth \
  --q-mxfp
```

No `--imatrix-path` was supplied. The converter therefore applied the fixed
tensor-class map without AWQ pre-scaling.

## Validation

The five-shard SafeTensors index contains 1,600 entries and reports
`metadata.total_size = 23,277,610,464` bytes. Header and index validation
confirmed exact shard closure, valid physical offsets, 168 MXFP4 groups, 233
MXFP8 groups, 401 scale sidecars, no bias sidecars, and identical
`quantization` and `quantization_config` blocks.

All 333 vision/merger tensors and all 15 `mtp.*` tensors remain BF16 without
quantization sidecars. No imatrix or calibration artifact is present. The
tokenizer, chat template, generation config, and preprocessor assets are
byte-identical to the pinned source snapshot.

A one-token mlx-node smoke test loaded the checkpoint and produced `OK`. A
second explicit native-MTP smoke detected `hasMtpWeights() = true`, completed a
four-token deterministic generation with `enableMtp: true`, and reported an MTP
decode cycle. These plumbing checks do not validate model quality, long-context
behavior, vision quality, or benchmark performance.

## License and attribution

The source model card declares the Apache-2.0 license. Model capability and
training credit belong to the Qwen Team. The tensor-class recipe is credited
to Unsloth. This repository converts the pinned BF16 source weights into the
mixed MXFP4/MXFP8 MLX representation described above.
