---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- basert
- quantized
---

# Qwen3.8-Flash-Next-BaseQuant-V1

BaseQuant V1 build of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), as a `.base` bundle for baseRT.

| Size | baseRT bundle | on the GPU |
|---|---|---|
| Medium | `Qwen3.8-Flash-Next-BaseQuant-V1-Medium.base` (140.32 GB) | 81.43 GB of resident weights |

The bundle holds the language model only (no vision tower, no MTP head). The file is larger than its GPU footprint because the per-layer n-gram embedding table (28.8 GB) is gathered from host memory rather than kept resident. There is no GGUF of this build. It needs a baseRT with the `qwen4_exp` encoder: **not in a release yet** — [baseRT-internal PR #911](https://github.com/basecompute/baseRT-internal/pull/911) on top of 0.5.3 — and 96 GB or more of unified memory (built and measured on an M3 Ultra).

## Usage

```bash
hf download basecompute/Qwen3.8-Flash-Next-BaseQuant-V1 Qwen3.8-Flash-Next-BaseQuant-V1-Medium.base --local-dir .
basert serve ./Qwen3.8-Flash-Next-BaseQuant-V1-Medium.base
```

## Accuracy

KL divergence of the next-token distribution from the BF16 model, on held-out text: 58 prompts in ten categories, 31,519 scored tokens. None of it was used to build this file. Lower is closer to BF16. The `.base` bundle is scored in baseRT; unsloth's `UD-Q4_K_XL` GGUF, the same GPU footprint (81.8 GB), is scored in llama.cpp and shown for reference. Over the 31,519 tokens the BaseQuant bundle changes the greedy next token 0 times against BF16; the `UD-Q4_K_XL` build changes it once.

![Held-out KL divergence by category](heldout_kl.png)

| Category | Tokens | BaseQuant `.base` | unsloth `UD-Q4_K_XL` |
|---|---:|---:|---:|
| Agentic coding | 2,474 | 0.01160 | 0.01183 |
| Chat | 2,740 | 0.00664 | 0.00615 |
| Code | 2,883 | 0.01244 | 0.00864 |
| Long context | 2,048 | 0.01034 | 0.00964 |
| Math | 4,608 | 0.00367 | 0.00388 |
| Multilingual | 4,264 | 0.00229 | 0.00322 |
| Prose | 3,994 | 0.00973 | 0.00953 |
| Research | 1,801 | 0.01982 | 0.02816 |
| Science | 4,239 | 0.00437 | 0.00538 |
| Tool use | 2,468 | 0.00870 | 0.01126 |
| **All** | 31,519 | 0.00778 | 0.00830 |

## Throughput

baseRT on an M3 Ultra, PR #911, the pipeline's tune bench (median of 3): prefill 1025 tok/s at 512 tokens and 1142 tok/s at 4096; decode 60.6 tok/s at 512 tokens of context, 59.0 at 4096, 57.1 at 16384.
