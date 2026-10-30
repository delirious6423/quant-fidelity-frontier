---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- deepseek41
- text-generation
- speculative-decoding
- vision
---

# DeepSeek V4.1 Flash 1017 — GGUF

GGUFs of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) @ `1bae0e10` for [Razor](https://github.com/flaviodelgrosso/razor).

- **`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`** (366 GB): the model. IQ2_XXS routed gate/up, Q2_K routed down; Q8_0 attention, shared experts and head; native FP8 Engram tables. SHA-256 `5296e85f…62ce93a`; imatrix SHA-256 `75fadebe…cce052f` (in `quantize.imatrix.sha256`).
- **`DeepSeek-V4.1-Flash-1017-DSpark.gguf`** (8.6 GB): lossless DSpark drafter for speculative decoding, same revision. SHA-256 `3347e357…2ed6b05`.
- **`DeepSeek-V4.1-Flash-Vision.gguf`** (971 MB): vision encoder (`deepseek4-vision`), built from DeepSeek-V4.1-Flash; not used by Razor yet. SHA-256 `cc283f03…0e9e15`.

## Use

```sh
razor download deepseek41 deepseek41-dspark
razor serve --model ~/.razor/models/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --placement ssd --expert-cache-gib 32 \
  --speculation opportunistic --dspark ~/.razor/models/DeepSeek-V4.1-Flash-1017-DSpark.gguf
```

SSD expert streaming fits 128 GiB Macs; fully resident needs ~192 GB.

MIT, as upstream.
