---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - ornith
  - qwen3_5_moe
---

# Ornith-1.5-35B-A3B GGUF

Dynamic GGUF of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MIT). Text only. The vision tower was not exported.

## File

`Ornith-1.5-35B-A3B-DYN-DIQ4XS-GU2XXS-Q8_0.gguf` — 14,197,860,128 bytes.

SHA-256: `0a90d392e524cd4f0331f41222ef88f78c490245c4a540150f943d5ca1aa125a`

| tensor | type |
| --- | --- |
| base (attention and the rest) | Q8_0 |
| `output.weight` | Q6_K |
| `token_embd.weight` | Q8_0 |
| `ffn_down_exps` | IQ4_XS |
| `ffn_gate_exps`, `ffn_up_exps` | IQ2_XXS |
| shared experts | IQ4_XS |
| MTP block `blk.40` experts | IQ4_XS |

How it was built: convert to BF16, quantize a plain IQ4_NL working copy, collect an importance matrix (48 chunks × 512 context), then quantize from BF16 with that matrix and the overrides above.

## Speed

Two RTX 3060 12 GB, context reservation 90,112, 81,928 prompt tokens, batch 2048 / ubatch 512, KV q4_0 / q4_0, Flash Attention, DFlash `n_max` 8 (the published drafter, left BF16, on the display card):

- prefill **1261.9 tok/s**
- decode **109.7 tok/s**

That decode used a repeated short function, so the drafter accepted every token. So, not a **real** speed check, just validation.

## Sampling

Model card general settings: temperature 0.6, top_p 0.95, top_k 20. Reasoning on.
