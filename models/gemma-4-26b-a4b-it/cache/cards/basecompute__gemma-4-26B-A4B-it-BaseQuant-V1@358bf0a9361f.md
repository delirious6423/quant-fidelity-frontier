---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
- gguf
- basert
- llama.cpp
- quantized
---

# gemma-4-26B-A4B-it-BaseQuant-V1

BaseQuant V1 build of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), as a `.base` bundle for baseRT and a GGUF for llama.cpp.

| Size | baseRT bundle | GGUF (llama.cpp) |
|---|---|---|
| Medium | `gemma-4-26B-A4B-it-BaseQuant-V1-Medium.base` (18.40 GB) | `gemma-4-26B-A4B-it-BaseQuant-V1-Medium.gguf` (16.93 GB) |

The `.base` bundle also carries the vision tower and the MTP head for speculative decoding. It needs **baseRT TODO-VERSION** or later.

## Usage

```bash
# baseRT
hf download basecompute/gemma-4-26B-A4B-it-BaseQuant-V1 gemma-4-26B-A4B-it-BaseQuant-V1-Medium.base --local-dir .
basert serve ./gemma-4-26B-A4B-it-BaseQuant-V1-Medium.base

# SuperFluid
curl -fsSL https://superfluid.sh/install.sh | sh
superfluid serve basecompute/gemma-4-26B-A4B-it-BaseQuant-V1 --runtime basert --pull-file gemma-4-26B-A4B-it-BaseQuant-V1-Medium.base

# llama.cpp
llama-server -hf basecompute/gemma-4-26B-A4B-it-BaseQuant-V1 -hff gemma-4-26B-A4B-it-BaseQuant-V1-Medium.gguf -c 32768 -ngl 99
```

## Accuracy

KL divergence of the next-token distribution from the BF16 model, on held-out text: 58 prompts in ten categories, 32,466 scored tokens. None of it was used to build these files. Lower is closer to BF16. The `.base` bundle is scored in baseRT, and the GGUFs in llama.cpp. unsloth's `UD-Q4_K_XL`, the same size, is shown for reference.

![Held-out KL divergence by category](heldout_kl.png)

| Category | Tokens | BaseQuant `.base` | BaseQuant GGUF | unsloth `UD-Q4_K_XL` |
|---|---:|---:|---:|---:|
| Agentic coding | 1,925 | 0.00527 | 0.00501 | 0.00536 |
| Chat | 3,291 | 0.00753 | 0.00787 | 0.00642 |
| Code | 2,897 | 0.10475 | 0.11525 | 0.10739 |
| Long context | 2,048 | 0.31866 | 0.31207 | 0.33115 |
| Math | 4,261 | 0.00251 | 0.00247 | 0.00297 |
| Multilingual | 4,253 | 0.00240 | 0.00269 | 0.00286 |
| Prose | 3,976 | 0.28921 | 0.29907 | 0.30696 |
| Research | 2,584 | 0.00457 | 0.00439 | 0.00553 |
| Science | 4,369 | 0.00275 | 0.00303 | 0.00367 |
| Tool use | 2,862 | 0.00575 | 0.00502 | 0.00613 |
| **All** | 32,466 | 0.06783 | 0.06957 | 0.07127 |
