---
base_model:
  - Qwen/Qwen3.8-27B
library_name: llama.cpp
tags:
  - gguf
  - qwen3_5
  - mtp
  - speculative-decoding
  - vision-language
---

# Qwen3.8-27B GGUF MTP (BF16-origin Q8_0)

This repository contains a llama.cpp-compatible Q8_0 conversion of
`Qwen/Qwen3.8-27B`, exported directly from the official BF16 checkpoint.
The native multi-token prediction (MTP) head is supplied as a separate draft
GGUF for `draft-mtp` speculative decoding.

## Files

- `Qwen3.8-27B-BF16-origin-Q8_0.gguf` — text/vision language model
- `mtp-Qwen3.8-27B-BF16-origin-Q8_0.gguf` — native MTP draft model
- `mmproj-Qwen3.8-27B-BF16-origin-Q8_0.gguf` — vision projector

## llama.cpp / llama-app

```bash
llama download -hf trevon/Qwen3.8-27B-GGUF-MTP \
  -hff Qwen3.8-27B-BF16-origin-Q8_0.gguf --mtp
```

For manual loading:

```bash
llama serve \
  --model Qwen3.8-27B-BF16-origin-Q8_0.gguf \
  --mmproj mmproj-Qwen3.8-27B-BF16-origin-Q8_0.gguf \
  --spec-type draft-mtp \
  --spec-draft-model mtp-Qwen3.8-27B-BF16-origin-Q8_0.gguf \
  --spec-draft-n-max 3
```

## Conversion

Converted with current upstream llama.cpp from the official Qwen BF16 source
revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` using `Q8_0` output,
`--no-mtp` for the target, and `--mtp` for the separate native draft model.

The upstream Qwen model license and acceptable-use terms apply.
