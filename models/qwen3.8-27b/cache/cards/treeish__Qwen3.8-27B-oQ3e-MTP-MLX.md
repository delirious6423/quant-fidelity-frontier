---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model:
  - Qwen/Qwen3.8-27B
tags:
  - mlx
  - safetensors
  - qwen3_5
  - oq
  - oqe
  - imatrix
  - quantized
  - mtp
  - 3-bit
  - treeish
  - sprig
---

# Qwen3.8-27B-oQ3e-MTP-MLX

This is the Qwen3.8-27B model package used by Sprig in Treeish. It combines an oQ3e mixed-precision MLX quant with its embedded Multi-Token Prediction head and the chat template used for Sprig's coding-agent workflow.

The package is ready to download and load as-is. Treeish pins an exact repository commit rather than following `main`.

## Model

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Architecture: 27B dense parameters
- Quantisation: oQ3e imatrix-enhanced mixed precision
- Default quantisation: 3-bit affine, group size 64
- Per-tensor overrides: 4, 5 and 6-bit, with group size 64
- Format: MLX safetensors
- MTP: 1 embedded layer under `language_model.mtp.*`
- Context length: 262,144 tokens

The package contains 2,209 indexed tensors, including 333 vision-tower tensors and 29 embedded MTP tensors. Its 3 weight shards contain 13,785,495,056 bytes of tensor data.

## Provenance

The quantised weights, model configuration and oQ calibration report are byte-identical to [mlx-works/Qwen3.8-27B-oQ3e-mtp](https://huggingface.co/mlx-works/Qwen3.8-27B-oQ3e-mtp) at commit `6a3f3c74a14d2fcba565ad8d5b11aff52b522f42`. The source declares oMLX 0.5.7 as the converter and Qwen/Qwen3.8-27B as the base model. Its imatrix report records 128 samples of 512 tokens from `oqe_code_multilingual`, with 503 of 504 entries applied. It does not identify the exact base-model commit used for conversion, so this is a curated, byte-pinned distribution rather than a byte-reproducible conversion recipe.

The chat template is Froggeric's v22.5 template. It is byte-identical to `chat_template.jinja` in [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) at commit `855bffc49448e299789730ff92c9b8d834d6cc14`.

No model, tokenizer or configuration tensors were changed for this release. The release replaces the chat template and adds the licence, provenance and file manifest. `tokenizer_config.json` is taken from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

## Runtime compatibility

This package is built for Treeish's pinned MLX Swift runtime. A different runtime must support the per-tensor quantisation overrides in `config.json` and the embedded Qwen MTP layout.

Treeish uses this model from 36 GB of unified memory and recommends 48 GB. Headroom depends on context length, cache settings and other running applications.

## Treeish validation

Treeish's release benchmark has not yet been run on this package. `RELEASE_MANIFEST.json` records the release status and is updated when it has.

## Limitations

Quantisation trades some model quality for memory use and local generation speed. Applications should validate the model against their own prompts, tool format and runtime.

The model package contains no custom executable code. File sizes, SHA-256 digests and source revisions are recorded in `RELEASE_MANIFEST.json`.

## Licence

Qwen3.8-27B is licensed under Apache 2.0. The full licence text is included in `LICENSE`. The Froggeric template repository also declares Apache 2.0 and is attributed above.
