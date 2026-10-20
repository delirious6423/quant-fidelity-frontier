---
license: other
license_name: lfm1.0
base_model:
- LiquidAI/LFM2.5-24B-A2B
tags:
- gguf
- lfm2
- moe
- text-generation
---
# LFM2.5-24B-A2B — antirez-style Q4 / Q3 / Q3+ GGUF

GGUF for mainstream llama.cpp (LFM2 MoE support, llama.cpp ≥ b7164).

## Recipe (antirez style)

| Tensor group | Q4 file | Q3 file (Q3\_K down) | Q3+ file (Q4\_K down) |
| --- | --- | --- | --- |
| Routed expert gate / up (76 tensors) | IQ4\_XS | IQ3\_XXS | IQ3\_XXS |
| Routed expert down (38 tensors) | Q4\_K | Q3\_K | Q4\_K |
| Shared experts, attention, shortconv projections, output | Q8\_0 | Q8\_0 | Q8\_0 |
| Token embeddings | F16 | F16 | F16 |
| Routers, norms, conv | F32 | F32 | F32 |

Calibrated with bartowski's imatrix (487 chunks, full expert coverage).
Everything except the routed experts is kept at Q8\_0 or better, so only the
64-expert FFN carries the quantization noise. KL-checked against the `Q8_0`
reference on short prompts (2 chunks @ ctx 60). Note: this model's experts
are more quantization-sensitive than Ling-3.0-tiny's — Q4 lands at the KL
level G9v3's Q2 showed, and Q3 roughly doubles that:

| File | KL divergence | Same top token | PPL (quant vs Q8\_0) |
| --- | --- | --- | --- |
| Q4 | 0.22 / 0.34 | 83% / 74% | 31.0 / 18.1 vs 15.5 |
| Q3 | 0.56 / 0.44 | 66% / 67% | 36.1 / 15.7 vs 15.5 |
| Q3+ (Q4\_K down) | 0.47 / 0.46 | 72% / 66% | 40.0 / 21.0 vs 15.5 |

## Files

| File | Size | BPW |
| --- | ---: | ---: |
| LFM2.5-24B-A2B-Q3-IQ3_XXS-Q3_K.gguf | 9.77 GiB | 3.50 |
| LFM2.5-24B-A2B-Q3-IQ3_XXS-Q4_K.gguf | 10.71 GiB | 3.84 |
| LFM2.5-24B-A2B-Q4-IQ4_XS-Q4_K.gguf | 12.84 GiB | 4.60 |

## Model

LFM2.5-24B-A2B is a 64-expert MoE with ~2 B active parameters out of 24 B
total (4 experts per token). Hybrid architecture: depthwise shortconv blocks
mixed with GQA attention, 38 MoE layers, hidden size 2048. Context: 131 072
tokens.

Original model: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).
Imatrix and BF16 source: [bartowski/LiquidAI_LFM2.5-24B-A2B-GGUF](https://huggingface.co/bartowski/LiquidAI_LFM2.5-24B-A2B-GGUF).
License: LFM Open License v1.0.
