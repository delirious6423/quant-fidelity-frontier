---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
license: apache-2.0
tags:
- qwen3.8
- gptq
- int8
- w8a8
- gs128
- quantized
- vllm
- amd
- mi100
- gfx908
- rocm
---

# Qwen3.8-27B GPTQ INT8 W8A8 GS128

**This target model is designed to run with the matching [DFlash2 draft model](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128) for speculative decoding.**

GPTQ 8-bit quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced and validated on **4x AMD Instinct MI100 (gfx908 / CDNA1)** over XGMI. This is
the reference checkpoint of an **int8-native serving stack** tuned specifically for
gfx908, where int8 matrix rate is 2x every dtype except fp16 (equal) at half the memory
bandwidth.

## Quantization details

| Parameter | Value |
|---|---|
| Format | GPTQ |
| Bits | 8 |
| Group size | **128** |
| Symmetric | true |
| desc_act | false |
| true_sequential | true |
| lm_head | false (kept fp16 — tied weights feed the logits GEMM) |
| Vision encoder / MTP module | untouched BF16 (carried in checkpoint) |
| Quantizer | GPTQModel 7.3.4, RTN-free (Hessian-compensated) |
| Calibration | 512 samples, mixed evol-codealpaca-v1 (code) + C4 (general text), binned uniformly 256–2048 tokens |

**Why group_size 128:** the accompanying runtime dispatches AITER true int8×int8
compute (W8A8) for every decode and prefill GEMM shape, which requires one weight
scale per 128-wide K block.
Quality was verified by a teacher-forced KL-divergence sweep over 6 precision
configurations before baking — this config measured **KLD 0.011 vs the gs32 reference**
(gate 0.02) and showed activation quantization adds only +0.0032 KLD on top.

## Serving (vLLM, gfx908 fork)

Use with the companion fork stack:

- **vLLM**: [curvedinf/int8-vllm](https://github.com/curvedinf/int8-vllm) — branch
  `main` (AITER W8A8 INT8 GEMMs everywhere, int8 embedding gather,
  int8 per-token-head KV, vLLM custom all-reduce)
- **AITER**: [curvedinf/int8-aiter](https://github.com/curvedinf/int8-aiter) —
  branch `main` (int8 unified-attention kernels + gfx908 tuning)

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
VLLM_GFX908_INT8_LM_HEAD=1 \
VLLM_GFX908_ACT_QUANT=round \
VLLM_DISABLED_KERNELS=TritonW8A16LinearKernel \
vllm serve <this-model-dir> \
  --tensor-parallel-size 4 \
  --max-num-seqs 8 \
  --dtype half \
  --max-model-len 65536 \
  --kv-cache-dtype int8_per_token_head \
  --mamba-ssm-cache-dtype float32 \
  --speculative-config '{
    "method": "dflash",
    "model": "curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128",
    "num_speculative_tokens": 15,
    "kv_cache_dtype": "int8_per_token_head"
  }'
```

This is one fixed TP4/C8 contract: both GS128 models, AITER W8A8 INT8 GEMMs
at every M, AITER unified attention, vLLM custom all-reduce, fp32 Mamba/GDN
state, and INT8 target/draft KV.
`--dtype half` is the model's residual/native dtype; GEMM inputs are dynamically
quantized to INT8 by the AITER W8A8 path.

The top-level KV flag configures this target; the nested field configures the
DFlash2 draft. Both use INT8 per-token-head KV. Do not substitute
`TRITON_ATTN`, W8A16, RCCL or AITER CAR all-reduce, fp16 KV, no speculation,
another TP size, or another concurrency in the intended recipe.

## Model architecture

Qwen3.8-27B is a hybrid dense multimodal model (model_type `qwen3_5`): 64 layers —
48 GDN linear-attention + 16 full-attention (repeating 3:1), hidden 5120, 27B
parameters, vocab 248,320, context 262,144. This checkpoint serves the language model
(`--language-model-only`); vision weights are carried for completeness.

## Reproduction

Quantization script and full experiment ledger (KLD sweep, kernel microbenchmarks,
A/B measurements) live in the vLLM fork:
[`~/models/quantize_qwen38_27b_gptq8.py`](https://github.com/curvedinf/int8-vllm)
and `logs/c8_optimization/experiments.md`.
