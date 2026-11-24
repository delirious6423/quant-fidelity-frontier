---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - incoai/Qwen3.8-27B-DFlash2
pipeline_tag: image-text-to-text
tags:
  - tenstorrent
  - blackhole
  - quietbox2
  - qwen3.8
  - dflash2
  - regen
---

# Qwen3.8-27B + DFlash2 + Vision Pre-Quantized Weights for Blackhole QuietBox 2 (`qwen-tt`)

Pre-quantized `.ttq` + `vocab.gguf` weights for serving **Qwen3.8-27B** (text, image, video, tool calling, DFlash2 speculative decoding, 262K context) on a 4-chip Tenstorrent Blackhole QuietBox 2 via [`moconnor/qwen3.8-27b-qb2-regen`](https://huggingface.co/moconnor/qwen3.8-27b-qb2-regen) and [`tenstorrent/regen-qwen3.8-27b-qb2`](https://github.com/tenstorrent/regen-qwen3.8-27b-qb2).

## Provenance & SHA-256

| File | Format | Size | Source Revision | SHA-256 |
| --- | --- | ---: | --- | --- |
| `vocab.gguf` | GGUF tokenizer | 11 MiB | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `e0e0e2e46d19d2044dcf49d51e16aeeec487b6fd82969bef0e57b638536943c2` |
| `bfp4-bfp8ah.ttq` | RTN BFP4 MLP/GDN + BFP8 Attn/Head | 18 GiB | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `7959ff0109a9625c55a57cb2f4ea9fc30b18d656e7e410d7f565aec10b999f8d` |
| `dflash2-bfp8.ttq` | BFP8 5-layer drafter + selector | 2.1 GiB | `incoai/Qwen3.8-27B-DFlash2@dedf8df68adfb1afeaf7b7480c0a0243108177b4` | `c77aa0951daed6f799d76fed2e5b2a0690b6d9d85a43b1cfafd6fb47f195e915` |
| `vision-bfp8.ttq` | BFP8 27-block ViT + 2×2 merger | 473 MiB | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `8703d25b3e28a88eb65810d37e466c29c03e061990c4677716c6e3f4f0469ba0` |

## Usage

Serve directly with [`tt-model`](https://github.com/tenstorrent/tt-model-manager):

```bash
tt-model serve moconnor/qwen3.8-27b-qb2-regen
```

## License

Weights are derived from `Qwen/Qwen3.8-27B` and `incoai/Qwen3.8-27B-DFlash2`, both licensed under **Apache-2.0**.
