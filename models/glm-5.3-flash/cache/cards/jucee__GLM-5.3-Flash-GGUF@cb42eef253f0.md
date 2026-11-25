---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- gguf
- jucee
- apple-silicon
- metal
- vision
---

# GLM-5.3 Flash for Jucee

GLM-5.3 Flash GGUF files for running on Apple Silicon Macs with [Jucee](https://github.com/bitnautlabs/jucee). Jucee downloads and runs them automatically.

| File | Size on disk | Runs on | Source |
|---|---|---|---|
| `GLM-5.3-Flash-Q4_K-AttnQ8.gguf` | 173 GiB | one 256 GB Mac, or two Macs together | Jucee recipe (`recipes/glm53-flash-q4-q8attn.sh`) |
| `GLM-5.3-Flash-Q2.gguf` | 90 GiB | one 128 GB Mac | Mirror of `antirez/glm-5.3-flash-gguf` |
| `GLM-5.3-Flash-Vision-Encoder.gguf` | 1.1 GiB | Vision sidecar | Mirror of `antirez/glm-5.3-flash-gguf` |

## `GLM-5.3-Flash-Q4_K-AttnQ8.gguf`

The upstream GLM-5.3 Flash Q4 keeps its KDA attention projections and output head in BF16 (11.4 GiB read per decoded token — more than all routed experts combined). This file stores those 137 tensors as Q8_0 and is 4.54 GiB smaller. Because they are BF16 in the official checkpoint too, the result matches a rebuild from source.

Measured on a Mac Studio with the same Jucee engine build for both files (2026-10-08):

| | Upstream Q4 | `Q4_K-AttnQ8` |
|---|---|---|
| Decode, 16k context | 30.0 tok/s | 34.8 tok/s |
| Decode, 64k context | 29.5 tok/s | 34.0 tok/s |
| Prefill, 16k context | 519 tok/s | 583 tok/s |

Quality scored against Z.AI's FP8 model (lower NLL is better): 100 cases at 4096 context, `0.30056` for upstream Q4 vs `0.29965` for this file; 8 long code cases at 8192 context, `0.50915` vs `0.51445` (within per-case spread).

Built from `GLM-5.3-Flash-Q4_K.gguf` in [antirez/glm-5.3-flash-gguf](https://huggingface.co/antirez/glm-5.3-flash-gguf) using [`recipes/glm53-flash-q4-q8attn.sh`](https://github.com/bitnautlabs/jucee/blob/main/recipes/glm53-flash-q4-q8attn.sh).

## Speculative Decoding

Jucee drafts GLM-5.3 Flash with DFlash 2 from [jucee/GLM-5.3-Flash-DFlash2](https://huggingface.co/jucee/GLM-5.3-Flash-DFlash2) (mirrored from [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2), Apache 2.0) and converts it to Q8_0 locally on your Mac.

## License & Attribution

Base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.AI, licensed under the **MIT License**.
Original GGUF conversion by [antirez/glm-5.3-flash-gguf](https://huggingface.co/antirez/glm-5.3-flash-gguf).
Jucee and its recipes are Apache 2.0. Jucee is a trademark of Bitnaut, LLC.
