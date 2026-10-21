---
license: mit
library_name: sglang
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- deepseek-v4
- fp8
- quantized
- dspark
language:
- en
- zh
---

# DeepSeek-V4-Flash-FP8

FP8 (block-wise `e4m3` + `f32` scale, 128×128 block) repackaging of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

## What this is

The source checkpoint mixes three storage formats:
- **MXFP4** (FP4 packed in int8 + E8M0 scale) — routed experts only
- **MXFP8** (F8_E4M3 + E8M0 scale) — attention, shared experts, MTP
- **BF16 / F32** — norms, embeddings, gates

This repo unifies everything into a single block-wise FP8 layout so the model can run through one FP8 kernel path.

## Conversion

| Tensor | Source | Target |
|---|---|---|
| `ffn.experts.*.{w1,w2,w3}` | int8 (FP4) + E8M0 scale | F8_E4M3 + F32 scale |
| `attn.*`, `shared_experts.*`, `mtp.*.e_proj` | F8_E4M3 + E8M0 scale | F8_E4M3 + F32 scale (weight unchanged) |
| `attn.wo_a.weight` | F8_E4M3 + E8M0 scale | BF16 (scale absorbed) |
| `attn.wo_a.scale` | E8M0 | dropped |
| norms, embed, gate, etc. | BF16 / F32 | unchanged |