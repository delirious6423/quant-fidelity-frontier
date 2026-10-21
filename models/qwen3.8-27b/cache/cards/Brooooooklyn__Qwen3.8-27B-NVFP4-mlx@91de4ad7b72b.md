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
  - nvfp4
  - fp8
  - e4m3
  - mixed-precision
  - nvidia
  - cuda
  - dgx-spark
  - linux
  - aarch64
  - qwen3.8
  - qwen3_5
  - vision-language
  - coding
  - unsloth
model_type: qwen3_5
quantized_by: mlx-node
---

# Qwen3.8-27B — Unsloth NVFP4 + E4M3 FP8 weights (MLX, DGX)

[`Brooooooklyn/Qwen3.8-27B-NVFP4-mlx`](https://huggingface.co/Brooooooklyn/Qwen3.8-27B-NVFP4-mlx)
is an MLX mixed-weight-format quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), prepared for
experimental NVIDIA CUDA inference on Linux aarch64. The model has a 64-layer
dense Qwen3.5-family language backbone with 48 linear-attention and 16
full-attention layers, a BF16 vision tower, and one preserved MTP layer.

This model is part of the
[Unsloth NVFP4 Tensor-Class Recipe for MLX — macOS + DGX collection](https://huggingface.co/collections/Brooooooklyn/unsloth-nvfp4-tensor-class-recipe-for-mlx-macos-dgx-6a5e3a893ae031d023e72ccf).

The source was the all-BF16 checkpoint at revision
[`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/commit/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).

## Update — re-converted 2026-09-01

Every dense SwiGLU FFN block scale is lifted by an exact power of two, folded
into the feeding norm. NVFP4 stores a block scale as `amax / 6` in E4M3, and on
real weights those land in E4M3's subnormal band, carrying three mantissa bits
instead of four -- or rounding to the zero code, which decodes all 16 weights in
the block to zero. The fold is exact, so the layer's function is unchanged.

Annihilated dense MLP blocks: **0.0049% -> 0.0000%**; worst dense tensor 0.0770% -> 0.0000%.

Weight bytes changed; the tensor-class map, shard layout and tensor counts are unchanged.
Earlier revisions remain in this repository's history.

## Quantization recipe

This is a data-free, **weight-only** MLX storage port of the
[Unsloth Qwen3.6 NVFP4 recipe](https://unsloth.ai/docs/models/qwen3.6#nvfp4),
applied to Qwen3.8's matching dense Qwen3.5-family tensor layout:

- tensors assigned NVFP4 by the recipe are stored as **NVFP4**, 4-bit with
  group size 16;
- tensors assigned FP8 by the recipe are stored as raw **E4M3 FP8** weight
  bytes with one BF16 dequantization scale per output channel;
- every excluded tensor stays BF16.

The NVFP4 class uses MLX weight-only quantized matmul with BF16/A16
activations. For the FP8 class, mlx-node reconstructs each weight to BF16 once
at load and then uses ordinary A16 matmul. The serialized `fp8_e4m3` form is
`Uint8 [N, K]` weight plus BF16 `[N, 1]` scale; it is not MLX `mxfp8` and it is
not native W8A8 execution.

No imatrix, calibration dataset, AWQ-style pre-scaling, activation
calibration, NVFP4 global scale, or FP8 KV-cache calibration was used. This
artifact preserves the recipe's tensor-class boundaries and weight storage
formats under mlx-node's A16 runtime; it does not claim numerical or
performance parity with Unsloth's calibrated W4A4/W8A8 execution.

| Tensor class | Stored format |
|---|---|
| Dense FFN `{gate,up,down}_proj`, layers 0–55 | NVFP4 4/16 |
| Dense FFN `{gate,up,down}_proj`, layers 56–63 | E4M3 FP8 weight + per-output BF16 scale |
| Full-attention `{q,k,v,o}_proj` | E4M3 FP8 weight + per-output BF16 scale |
| Linear-attention `in_proj_qkv`, `in_proj_z`, `out_proj` | E4M3 FP8 weight + per-output BF16 scale |
| `lm_head` | E4M3 FP8 weight + per-output BF16 scale |
| Embeddings; `in_proj_a/b`; GDN state, convolution, and norm tensors; all other norms | BF16 |
| Entire 15-tensor `mtp.*` subtree | BF16 |
| Vision tower and merger tensors | BF16 |

The allocation contains **168 NVFP4 modules** and **233 E4M3 FP8 modules**.
The top-level config is `nvfp4`, 4-bit, group size 16; the 233 higher-class
modules are explicit `fp8_e4m3` overrides. The final eight dense FFNs
intentionally use the higher class.

## Target and usage

The canonical native target is `aarch64-unknown-linux-gnu`: Linux aarch64 with
glibc and NVIDIA CUDA 13.0. mlx-node currently validates this experimental
inference-only path on NVIDIA GB10 / DGX Spark (`sm_121`). It is not a generic
CUDA or x86_64 artifact, and there is no published CUDA prebuilt native npm
binary. Build the pinned checkout on the DGX host:

```bash
git clone https://github.com/mlx-node/mlx-node.git
cd mlx-node
git checkout e281f0bbc61eb71a244ad6681ffad324a1177780
git submodule update --init --recursive
yarn install
yarn build
```

The current CUDA preview uses eager execution and a flat KV cache:

```bash
MLX_QWEN35_FORCE_EAGER=1 \
MLX_QWEN35_PAGED_OVERRIDE=0 \
  yarn oxnode your-script.ts
```

For example:

```typescript
import { loadSession } from '@mlx-node/lm';

const session = await loadSession('./Qwen3.8-27B-NVFP4-mlx');
const result = await session.send('Explain the purpose of a unit test in one sentence.');
console.log(result.text);
```

The full BF16 MTP subtree is preserved inline and is supported by mlx-node on
the macOS fallback path. CUDA speculative decoding is not supported in the DGX
preview, so preservation of these tensors does not claim DGX MTP validation.

## Reproduction

Converter checkout:
[`mlx-node e281f0bb`](https://github.com/mlx-node/mlx-node/tree/e281f0bbc61eb71a244ad6681ffad324a1177780)
(package version 0.0.10). The invocation was:

```bash
mlx convert \
  --input /Users/brooklyn/.mlx-node/models/qwen3.8-27b \
  --output /Users/brooklyn/.mlx-node/models/qwen3.8-27b-unsloth-nvfp4-mlx \
  --model-type qwen3_5 \
  --dtype bfloat16 \
  --quantize \
  --q-recipe unsloth \
  --q-mode nvfp4
```

No `--imatrix-path` was supplied. The converter therefore applied the fixed
tensor-class map without AWQ pre-scaling.

## Validation

The five-shard SafeTensors index contains 1,600 entries and reports
`metadata.total_size = 23,417,338,336` bytes. Header and index validation
confirmed exact shard closure, valid physical offsets, 168 inherited NVFP4
groups, 233 complete `fp8_e4m3` groups, 401 scale sidecars, no bias sidecars,
and identical `quantization` and `quantization_config` blocks.

All 333 vision/merger tensors and all 15 `mtp.*` tensors remain BF16 without
quantization sidecars. No imatrix or calibration artifact is present. The
tokenizer, chat template, generation config, and preprocessor assets are
byte-identical to the pinned source snapshot.

A macOS A16-fallback smoke test detected `hasMtpWeights() = true` and completed
a short deterministic generation with native MTP explicitly enabled. This
plumbing check does not validate model quality, long-context behavior, vision
quality, DGX/CUDA execution, or benchmark performance.

## License and attribution

The source model card declares the Apache-2.0 license. Model capability and
training credit belong to the Qwen Team. The tensor-class recipe is credited
to Unsloth. This repository converts the pinned BF16 source weights into the
mixed NVFP4/plain-E4M3 MLX representation described above.
