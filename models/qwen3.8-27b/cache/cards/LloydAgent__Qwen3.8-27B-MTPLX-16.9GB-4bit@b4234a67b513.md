---
language:
- en
library_name: mlx
tags:
- qwen
- qwen3
- mtplx
- mlx
- apple-silicon
- 4-bit
- quantized
- text-generation
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
license: apache-2.0
---

# Qwen3.8-27B-MTPLX-16.9GB-4bit

A custom MTPLX-forged version of `Qwen/Qwen3.8-27B`, optimized for Apple Silicon with a compact **16.924 GB** complete artifact size (**15.762 GiB**) while retaining native MTPLX multi-token prediction support.

This build is intended for users who want a capable 27B-class Qwen model that leaves headroom for a practical context window and lightweight applications on machines with constrained unified memory.

## Key details

| Field | Value |
| --- | --- |
| Public model name | `Qwen3.8-27B-MTPLX-16.9GB-4bit` |
| Base model | `Qwen/Qwen3.8-27B` |
| Base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Complete artifact size | **16.924 GB** / **15.762 GiB** |
| Quantization | **4-bit affine**, group size 64 |
| Measured body precision | 4.501 bits/weight |
| MTPLX support | Native MTP sidecar, FP16, 15 tensors |
| Runtime compatibility | `verified-native` in MTPLX 2.9.0 |
| Max MTP depth in artifact | 3 |
| Recommended tested depth | Depth 2 on Apple M2 Pro |
| Vision tower | Preserved (`model-vision.safetensors`) |

## Built with MTPLX

This model was forged and optimized using MTPLX.

Powered by MTPLX — https://github.com/youssofal/MTPLX

MTPLX runs local LLMs on Apple Silicon with native multi-token prediction where supported by the model architecture. The MTP sidecar in this artifact is preserved as native FP16 weights rather than being requantized.

## Forge recipe

```json
{
  "body_bits": 4,
  "body_dtype": "auto",
  "body_group_size": 64,
  "body_mode": "affine",
  "mtp_policy": "keep_bf16"
}
```

`--dtype auto` on Apple M-series hardware maps non-quantized tensors, including the native MTP sidecar floats, to FP16.

## Verification summary

`mtplx inspect --require-mtp` reports this artifact as:

- support level: `verified-native`
- runtime compatibility: `native`
- unverified model: `false`
- MTP tensor count: 15 / 15 expected
- quantization: 4-bit affine, group size 64

## Usage

Install MTPLX from the project instructions, then load this model by Hugging Face repo id or local path in MTPLX.

```bash
mtplx inspect --require-mtp <repo-or-local-path>
mtplx start --model <repo-or-local-path>
```

For best results on similar Apple Silicon hardware, start with MTP depth 2 and tune locally:

```bash
mtplx tune --model <repo-or-local-path> --retune
```

## Notes and limitations

- This is an MLX/MTPLX-oriented artifact for Apple Silicon.
- Runtime memory depends on context length, KV cache settings, server mode, and other resident applications; the listed 16.924 GB is the complete on-disk artifact size, not a universal maximum runtime allocation.
- Model weights remain governed by the upstream base model license and terms, plus any licenses attached to the tooling used to create this artifact.

## Attribution

- Base model: `Qwen/Qwen3.8-27B`
- MTPLX: https://github.com/youssofal/MTPLX
