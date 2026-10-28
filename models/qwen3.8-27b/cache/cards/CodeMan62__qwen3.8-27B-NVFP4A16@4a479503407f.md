---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- nvfp4
- fp4
- quantized
- 4-bit
- compressed-tensors
- vllm
- qwen3.8
- hollowquant
language:
- en
---

# Qwen3.8-27B NVFP4A16

Weight-only NVFP4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [hollowQuant](https://github.com/HollowFound/hollowQuant) (data-free RTN with a
±1-codepoint block-scale search, `scale_search="neighbors"`). The artifact is a
compressed-tensors **`nvfp4-pack-quantized`** checkpoint (W4A16): E2M1 codes packed two per
byte, one E4M3 scale per 16 weights, one fp32 scale per tensor — 4.50 bits per weight.

## What was quantized

- All `mlp.gate/up/down_proj` (64 layers)
- Full-attention `q/k/v/o_proj` (16 layers; q/k/v share a global scale)
- Linear-attention (Gated DeltaNet) `linear_attn.in_proj_qkv` + `in_proj_z` (shared scale),
  `in_proj_a` + `in_proj_b` (shared scale), `out_proj` (48 layers)
- Kept dense: token embeddings, `lm_head`, vision tower, MTP head, norms

496 modules, 24.35B params quantized; 48.7 GB of bf16 linears → 13.7 GB (~20.5 GB checkpoint).

## Quality

Full-vocabulary KL vs bf16 on self-generated chat text (2,919 tokens, transformers 5.8):

| mean KL | p99 KL | top-1 match |
|---|---|---|
| 0.0238 | 0.2223 | 93.87% |

Downstream (bf16 reference on vLLM 0.31.0, greedy, full runs): GSM8K 90.45%, MMLU 83.46%.
Quantized scores: run `benchmarks/eval_nvfp4_local.py` from the hollowQuant repo on your GPU.

## Usage (vLLM ≥ 0.17.0)

```bash
vllm serve <this-repo> --max-model-len 8192
```

Loads as `Qwen3_5ForConditionalGeneration` with compressed-tensors NVFP4; all quantized
modules (including the Gated DeltaNet projections) run on the FP4 Marlin kernel with 16-bit
activations. Fits one 32 GB GPU (e.g. RTX 5090) with headroom for KV cache.
