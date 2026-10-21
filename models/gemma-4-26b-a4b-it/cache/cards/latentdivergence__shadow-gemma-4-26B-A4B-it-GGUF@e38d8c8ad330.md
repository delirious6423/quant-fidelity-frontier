---
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags:
- gguf
- gemma4
- mixture-of-experts
- shadow-studio
---

# Shadow Studio mirror — Gemma 4 26B A4B IT (GGUF)

Unmodified GGUF copy of **Gemma 4 26B A4B instruction-tuned** (mixture-of-experts, ~4B active) weights, mirrored as the stable download source for the [Shadow Studio](https://latentdivergence.com) local AI workspace (Best tier).

- `google_gemma-4-26B-A4B-it-Q4_K_M.gguf` — Q4_K_M quantization (~17 GB)

Files are byte-identical re-hosts of the quantizations published by [bartowski](https://huggingface.co/bartowski) (thanks!). No further modification was made. Shadow Studio verifies each download against a pinned SHA-256.

## License

Gemma is provided under and subject to the **Gemma Terms of Use** found at https://ai.google.dev/gemma/terms. Use of these weights is also subject to Google's Prohibited Use Policy: https://ai.google.dev/gemma/prohibited_use_policy

By downloading these files you agree to those terms. This repository is deliberately **not gated** so that Shadow Studio's built-in downloader (which carries no auth token) can fetch models directly; the app surfaces the Gemma Terms notice in its Model Manager before download.