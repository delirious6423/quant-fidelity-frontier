---
license: mit
library_name: gguf
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
  - gguf
  - deepseek-v4
  - dspark
  - dflash
  - speculative-decoding
  - draft-model
  - auxiliary-model
---

# DeepSeek V4 Flash 0731 compact DSpark drafters

This repository contains the compact DSpark auxiliary for DeepSeek V4 Flash
0731. It is not a standalone language model and does not include target-model
weights.

## Artifact

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf` | 6,976,490,112 | `2c8065906a7b4d841e10f15b350ed14410fe715df3973e9622dccd278ed28e8e` |
| `DeepSeek-V4-Flash-0731-DSpark-Drafter-IQ2_XXS-MXFP4-Q8_0-dflash.gguf` | 7,302,985,888 | `e4b2b63356e11823c81eb2d3dc9739ae3c2935b970747c9b3be101548a3c17ec` |

The routed experts use Q2_K. Dense projections use Q8_0. The 81-tensor GGUF
uses the standardized llama.cpp `general.architecture=dflash` schema.

The Q2_K artifact is the smallest option. The mixed artifact uses IQ2_XXS for
the gate and up routed-expert tensors and preserves the down routed-expert
tensors in native MXFP4. Its remaining quantized tensors use Q8_0. The mixed
recipe follows the small-drafter design from antirez/ds4 issue 642 and keeps
more expert fidelity while remaining suitable for 128 GiB Apple Silicon Macs.
The target, drafter, KV cache, runtime buffers, and other loaded models must
still fit together.

Unsloth publishes a full-MXFP4 Q8_0 drafter and a BF16 drafter. This repository
provides the smaller mixed expert recipe that Unsloth does not publish.

## Download

```sh
hf download dev7a/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF \
  DeepSeek-V4-Flash-0731-DSpark-Drafter-IQ2_XXS-MXFP4-Q8_0-dflash.gguf
```

## Provenance and verification

The source is `deepseek-ai/DeepSeek-V4-Flash-0731` revision
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`. The repository contains the
pinned source manifest, the Q2_K converter, standardized `dflash` rewriters,
verification scripts, tests, build manifests, and checksums.

Both releases passed these schema checks:

- 81 canonical `dflash` tensors with 32-byte alignment.
- Byte-identical tensor payloads before and after the schema rewrite.
- Byte-identical repeated builds.
- Complete target-tokenizer metadata copied through the official llama.cpp
  DeepSeek V4 DSpark converter at commit
  `15586e2d7165570fb3aa7c26e0d442e289ef69de`, including
  `tokenizer.ggml.mask_token_id=128799`.
- A real llama.cpp `draft-dspark` decode generated 120 draft tokens and
  accepted 38, with no invalid `-1` token.
- ns4 build `e470f38280d9ff4f020c242233998562a88463ee` loaded the corrected Q2
  file and completed an exact 32-token decode, drafting 47 and accepting 12.

The Q2_K build also passed its numeric limits: maximum Q2_K block-relative
error `0.49072265625`, below `0.5`, and maximum Q8_0 block-relative error
`0.00396728515625`, below `0.01`.

The mixed artifact was converted with `antirez/ds4` pull request 645 at commit
`d147270d7422ba0eb55985c8d4075b9dfaa84bf1`. It uses the public routed-MoE
importance matrix from `antirez/deepseek-v4-gguf`. The build manifest pins its
revision, checksum, and the deterministic MTP alias transformation.

Reproduce the artifact on Linux AArch64 with Python 3.14:

```sh
uv venv --python 3.14.6 .venv
uv pip install --python .venv/bin/python \
  --require-hashes --only-binary=:all: \
  -r requirements-linux-aarch64-py314.lock
.venv/bin/python scripts/download_sources.py --destination sources
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce.py \
  --sources sources --recipe q2_k-q8_0 \
  --manifest-dir manifest --repeat-check
.venv/bin/python scripts/dflash.py \
  --sources sources --recipe q2_k-q8_0 \
  --legacy-input DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0.gguf \
  --target-tokenizer-gguf tokenizer.gguf \
  --manifest-dir manifest --repeat-check

# The mixed legacy GGUF is produced with the pinned ds4 converter and imatrix.
.venv/bin/python scripts/rewrite_mixed_dflash.py \
  DeepSeek-V4-Flash-0731-DSpark-Drafter-IQ2_XXS-MXFP4-Q8_0.gguf \
  --target-tokenizer-gguf tokenizer.gguf \
  --manifest-dir manifest
```

## Compatibility

Use this file only with a compatible DeepSeek V4 Flash 0731 target. The runtime
must support the standardized llama.cpp `dflash` GGUF schema and the selected
Q2_K or IQ2_XXS/MXFP4 expert recipe.
This is a community conversion, not an official DeepSeek release.

The source-derived weights remain under DeepSeek's MIT license. Conversion code
and third-party notices are provided in `LICENSE.code` and
`THIRD_PARTY_NOTICES.md`.
