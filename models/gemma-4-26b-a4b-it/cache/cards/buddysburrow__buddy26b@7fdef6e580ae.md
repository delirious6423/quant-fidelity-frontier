---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
---

# Buddy26B

The brain of **Buddy: Your AI Friend** (iOS), by Buddy's Burrow. It runs entirely on the phone.

Built on Google's **Gemma 4 26B-A4B** (instruction-tuned), using Unsloth's UD-IQ2_M quantization
([unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)), split so the
experts can be streamed from flash:

| File | What |
|---|---|
| `buddy26b.json` | Map of the split (which expert lives where) |
| `buddy26b.resident.gguf` | Always-used weights (5-bit), 1.78 GB |
| `buddy26b.experts` | Experts, read from flash as needed, 8.27 GB |
| `buddy26b-eyes.gguf` | Vision projector (ggml-org's Gemma 4 26B-A4B mmproj, Q8_0), 0.81 GB |

Gemma 4 is released by Google under the Apache License 2.0 (see google/gemma-4-26B-A4B-it); this split keeps that license.
