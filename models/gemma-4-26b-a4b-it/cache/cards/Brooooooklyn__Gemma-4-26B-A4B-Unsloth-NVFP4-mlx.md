---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: mlx-node
pipeline_tag: image-text-to-text
tags:
  - mlx
  - mlx-node
  - gemma4
  - nvfp4
  - fp8
  - quantized
---

# Gemma 4 26B-A4B IT — Unsloth NVFP4/E4M3 recipe for MLX

This is a weight-only MLX conversion of
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
using the tensor-class policy from
[`unsloth/gemma-4-26B-A4B-it-NVFP4`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-NVFP4):

| Tensor class | Count | Storage |
| --- | ---: | --- |
| Dense and routed-expert MLP `gate/up/down` projections | 180 | NVFP4, 4-bit, group 16 |
| Language-model attention `q/k/v/o` projections | 115 | E4M3 FP8 plus one BF16 scale per output |
| Routers, embeddings/head, norms, vision tower, and unmatched tensors | — | BF16 |

There is no Qwen-style “final eight FFNs use the high class” exception for
Gemma 4. All Gemma language-model dense and expert FFNs stay NVFP4.

## Update — re-converted 2026-09-01

Annihilated dense MLP blocks: **1.7422% -> 0.0000%**; worst dense tensor 11.6023% -> 0.0000%.

The imatrix / AWQ pre-scaling is removed. Under a float-scaled format it drove
block scales to the zero code and annihilated whole blocks, so mlx-node now
refuses `--imatrix-path` for these class maps. The reproduce command above is
updated accordingly; the previous one fails on current mlx-node.

Weight bytes changed; the tensor-class map, shard layout and tensor counts are unchanged.
Earlier revisions remain in this repository's history.

## Provenance

- Base weights: `google/gemma-4-26B-A4B-it` revision
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- Tensor-class reference: `unsloth/gemma-4-26B-A4B-it-NVFP4` revision
  `20df0542b1a86ce19f495ac2eca2c7c12bce82f9`
- Converter: current `mlx-node` source with Gemma fixed-map support

No imatrix or calibration dataset is used. The FP4 tensor-class maps are
data-free, and AWQ pre-scaling is refused under a float-scaled format:
dividing a weight column by an AWQ scale drives the block amax down until
`amax / 6` falls under E4M3's smallest normal value and the shared block scale
rounds to the zero code, which decodes the whole block to zero.

## Runtime semantics

This artifact preserves the upstream low/high weight classes, but activations
remain 16-bit. NVFP4 uses MLX weight-only quantized matmul. The raw E4M3
attention weights are reconstructed to BF16 once when the model loads, then use
ordinary A16 matmul. This is not upstream W4A4/W8A8 execution and does not claim
global-scale, FP8 KV-cache, numerical, or performance parity.

The tokenizer, chat template, processor, and vision tower are retained, but
only text generation was smoke-tested for this upload. At the time of
conversion, Gemma's fixed Unsloth map and rank-2 plain-FP8 loader require the
mlx-node source version that introduced this support; it is newer than the
plain published `v0.0.8` behavior.

## Reproduce

```bash
yarn mlx convert \
  --input /path/to/gemma-4-26b-a4b-it \
  --output ./Gemma-4-26B-A4B-Unsloth-NVFP4-mlx \
  --model-type gemma4 \
  --dtype bfloat16 \
  --quantize \
  --q-recipe unsloth \
  --q-mode nvfp4 \
  --q-bits 4
```

## Validation

- 25,805,936,206 source parameters converted.
- 1,338 serialized tensors across four SafeTensors shards.
- Exact index/shard closure.
- Exact class map: 180 NVFP4 4/16 FFNs and 115 E4M3 FP8 attention overrides.
- E4M3 storage verified as raw U8 `[N,K]` weights plus BF16 `[N,1]` scales.
- Quantization metadata is identical in `quantization` and
  `quantization_config`.
- Text smoke test: loaded in 24.4 seconds and generated one deterministic token,
  `OK`.

This model belongs to the
[`Unsloth NVFP4 Tensor-Class Recipe for MLX macOS/DGX`](https://huggingface.co/collections/Brooooooklyn/unsloth-nvfp4-tensor-class-recipe-for-mlx-macos-dgx-6a5e3a893ae031d023e72ccf)
collection.
