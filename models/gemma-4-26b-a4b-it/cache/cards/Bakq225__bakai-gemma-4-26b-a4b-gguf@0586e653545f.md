---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags: [gguf, llama.cpp, gemma4, on-device, android]
---

# Gemma 4 26B-A4B for 8 GB phones (BakAI)

Mixed low-bit GGUF of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it),
used by the BakAI Android app, which downloads it on first launch.

| tensors | type |
|---|---|
| routed experts gate/up | IQ2_XXS (2.06 bpw) |
| routed experts down | IQ4_NL (llama.cpp needs 256-divisible columns; Gemma 4's down projection has 704) |
| attention, token embedding | Q4_K |
| dense MLP | Q4_K / Q5_0 |

Average 3.05 bpw, 9.63 GB, split into 7 parts with `llama-gguf-split` (load the first part;
llama.cpp finds the rest). Importance matrix: bartowski's for the same model.
Quality against the original has not been measured yet.

License: Apache 2.0, same as the base model ([Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license)).
