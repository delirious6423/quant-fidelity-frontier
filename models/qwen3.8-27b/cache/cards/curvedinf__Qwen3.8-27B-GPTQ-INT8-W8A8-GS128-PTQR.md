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
- ptqr
- quantized
- vllm
- amd
- mi100
- gfx908
- rocm
---

# Qwen3.8-27B GPTQ INT8 W8A8 GS128 PTQR

**This target model is designed to run with the matching [DFlash2 draft model](https://huggingface.co/curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR) for speculative decoding.**

PTQR-retrained quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced and validated on **4x AMD Instinct MI100 (gfx908 / CDNA1)** over XGMI. It uses the
same GPTQ INT8 W8A8 GS128 checkpoint format and the same serving methodology as
[my original GPTQ checkpoint](https://huggingface.co/curvedinf/Qwen3.8-27B-GPTQ-INT8-W8A8-GS128)
— This checkpoint is a drop-in replacement with lower quantization error.

## What is PTQR?

PTQR (post-training quantization with rounding-dither retraining) fine-tunes the model
weights directly under the deployed quantization: a student whose forward pass replicates
the exact INT8 serving kernels (the G128 weight grid, dynamic per-token activation
quantization, INT8 KV cache) is distilled against a frozen BF16 teacher, with geometrically
annealed dither on each rounding decision so that early training explores neighboring grid
points and late training is bit-identical to serving. The retrained weights are re-quantized
and exported through the same GPTQ GS128 wrapper, so the checkpoint stays format- and
config-identical to the one-shot original.

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
| Quantizer | PTQR retraining on the GPTQModel 7.3.4 GS128 grid (weights-only SGD, group scales frozen) |
| Retraining | quantization-aware distillation vs a frozen BF16 teacher; no calibration pass |

**Why group_size 128:** the accompanying runtime dispatches AITER true int8×int8
compute (W8A8) for every decode and prefill GEMM shape, which requires one weight
scale per 128-wide K block.
Quality was verified at the serving gate (teacher-forced KL-divergence probe, greedy
agreement, speculative-decoding acceptance): this checkpoint measures **KLD 0.0069 vs
0.0110** for the original GPTQ checkpoint (gate 0.02), **42/52 vs 38/52** greedy
agreement, and acceptance **3.88/13** vs the BF16 baseline 3.67.

## Serving (vLLM, gfx908 fork)

Use with the companion fork stack:

- **vLLM**: [curvedinf/int8-vllm](https://github.com/curvedinf/int8-vllm) — branch
  `main` (AITER W8A8 INT8 GEMMs everywhere, int8 embedding gather,
  INT8 KV, vLLM custom all-reduce)
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
  --dtype bfloat16 \
  --max-model-len 65536 \
  --kv-cache-dtype int8_block_g128 \
  --mamba-ssm-cache-dtype float32 \
  --speculative-config '{
    "method": "dflash",
    "model": "curvedinf/Qwen3.8-27B-DFlash2-GPTQ-INT8-W8A8-GS128-PTQR",
    "num_speculative_tokens": 13,
    "kv_cache_dtype": "int8_block_g128"
  }'
```

This is one fixed TP4/C8 contract: both GS128 models, AITER W8A8 INT8 GEMMs
at every M, AITER unified attention, vLLM custom all-reduce, fp32 Mamba/GDN
state, and INT8 target/draft KV.
`--dtype bfloat16` is the model's residual/native dtype; GEMM inputs are dynamically
quantized to INT8 by the AITER W8A8 path.
The top-level KV flag configures this target; the nested field configures the
DFlash2 draft. Both use INT8 block KV with 128-wide groups. Do not substitute
`TRITON_ATTN`, W8A16, RCCL or AITER CAR all-reduce, fp16 KV, no speculation,
another TP size, or another concurrency in the intended recipe.

## Model architecture

Qwen3.8-27B is a hybrid dense multimodal model (model_type `qwen3_5`): 64 layers —
48 GDN linear-attention + 16 full-attention (repeating 3:1), hidden 5120, 27B
parameters, vocab 248,320, context 262,144. This checkpoint serves the language model
(`--language-model-only`); vision weights are carried for completeness.

## Reproduction

The PTQR trainer, the export vehicle, and the full experiment ledger (KLD gates,
acceptance batteries, A/B measurements) live in the vLLM fork:
[curvedinf/int8-vllm](https://github.com/curvedinf/int8-vllm) — see
`docs/recipes/README.md` and `docs/recipes/surface_experiments_ledger.jsonl`.
