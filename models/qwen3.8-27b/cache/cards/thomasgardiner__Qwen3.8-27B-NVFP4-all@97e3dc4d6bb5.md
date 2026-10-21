---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
  - RadixArk/Qwen3.8-27B-NVFP4
  - Qwen/Qwen3.8-27B
inference: false
tags:
  - qwen3.8
  - nvfp4
  - modelopt
  - sglang
---

# Qwen3.8-27B-NVFP4-all

**+13 to +17% decode throughput on an RTX 5090** over RadixArk/Qwen3.8-27B-NVFP4 at the same GSM8K and MATH-500 accuracy. That is this checkpoint versus the RadixArk export, both served with FlashInfer `b12x` and the bf16 DFlash2 draft. The full shipping stack against stock SGLang is +25 to +33% on fixed prompts; see [thomgardiner/sglang-sm120-nvfp4](https://github.com/thomgardiner/sglang-sm120-nvfp4).

## Model Overview

- **Model Architecture:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), hybrid Gated DeltaNet + full attention, 64 layers
- **Input / Output:** text / text
- **Base checkpoint:** [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- **Release Date:** 2026-09-06
- **License:** Apache-2.0

The 208 linear projections that RadixArk/Qwen3.8-27B-NVFP4 stores in FP8 are NVFP4 here: `linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj` in the 48 Gated DeltaNet layers and `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` in the 16 attention layers. MLP and `lm_head` tensors are byte-identical to the base checkpoint. 20.6 GB → 18 GB.

## Model Optimizations

Weights are quantized with a symmetric per-block scheme, block size 16, E4M3 block scales and one FP32 global scale per tensor (`amax / (448 × 6)`). Activations use a static per-tensor scale. Activation scale = the base checkpoint's FP8 static activation scale ÷ 6. No new calibration; the base export was calibrated by Model Optimizer on 1,024 samples of cnn_dailymail at 512 tokens. Tensors that SGLang fuses at load time (`in_proj_qkv` + `in_proj_z`, `q_proj` + `k_proj` + `v_proj`) share one global scale per group.

Weights were quantized from the bf16 release of Qwen3.8-27B, not from the FP8 tensors. `in_proj_a`, `in_proj_b`, convolutions, norms, embeddings, and the `mtp` head remain bf16 as in the base checkpoint.

## Deployment

```bash
python3 -m sglang.launch_server \
  --model-path thomasgardiner/Qwen3.8-27B-NVFP4-all \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
  --speculative-num-draft-tokens 8 \
  --attention-backend flashinfer \
  --trust-remote-code \
  --max-running-requests 1 \
  --cuda-graph-max-bs-decode 1 \
  --mem-fraction-static 0.91
```

Tested with `lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe` (FlashInfer 0.6.17, CUDA 13.0). On SM120, apply the `b12x` backend patch from [thomgardiner/sglang-sm120-nvfp4](https://github.com/thomgardiner/sglang-sm120-nvfp4) so NVFP4 GEMMs run on FlashInfer's SM120 kernel. Without it the GEMMs stay on CUTLASS; see the cutlass row in that repo's serving table.

## Creation

```bash
python3 convert.py <Qwen/Qwen3.8-27B bf16 dir> <RadixArk/Qwen3.8-27B-NVFP4 dir> <output dir>
```

`convert.py` in this repository. Its `check` mode re-quantizes two of the base checkpoint's NVFP4 MLP tensors from bf16 and compares them to the stored tensors: block scales match 100%, packed weights 99.4 to 99.6%, reconstruction error identical to five digits.

## Evaluation

One RTX 5090 per target, SGLang with the `b12x` patch, DFlash2 bf16 draft, greedy, thinking disabled, 8 concurrent requests. lm-evaluation-harness 0.4, `--apply_chat_template --num_fewshot 0`, `max_gen_toks` 4096 (2048 for IFEval and HumanEval).

| Benchmark | Metric | RadixArk/Qwen3.8-27B-NVFP4 | Qwen3.8-27B-NVFP4-all | Recovery |
| --- | --- | ---: | ---: | ---: |
| GSM8K (1319) | exact match, flexible-extract | 85.90 ± 0.96 | 85.82 ± 0.96 | 99.9% |
| MATH-500 | math_verify | 86.00 ± 1.55 | 84.80 ± 1.61 | 98.6% |
| GPQA Diamond (198) | CoT zero-shot, exact match | 68.18 ± 3.32 | 67.68 ± 3.33 | 99.3% |
| IFEval (541) | prompt-level strict | 79.30 ± 1.74 | 80.96 ± 1.69 | 102.1% |
| IFEval (541) | prompt-level loose | 83.55 ± 1.60 | 84.47 ± 1.56 | 101.1% |
| HumanEval (164) | pass@1, executed | 93.29 | 92.68 | 99.3% |
| **Average** | | | | **99.8%** |

Every difference is inside one standard error. HumanEval is graded by running the fenced code the model returned against the task's `check()`; the harness's own `humaneval_instruct` filter expects a pre-opened code fence and scores a chat model at 0. With thinking enabled, GSM8K on the first 500 items was 83.0 ± 1.7 (RadixArk) and 85.2 ± 1.6 (this checkpoint). Result files and the grader: [receipts/lmeval](https://github.com/thomgardiner/sglang-sm120-nvfp4/tree/main/receipts/lmeval), [bench/he_grade.py](https://github.com/thomgardiner/sglang-sm120-nvfp4/blob/main/bench/he_grade.py).

### Throughput

Both columns use FlashInfer `b12x` and the bf16 DFlash2 draft. The change is this checkpoint only.

Named datasets, greedy, thinking on, concurrency 1:

| Workload | RadixArk/Qwen3.8-27B-NVFP4 | Qwen3.8-27B-NVFP4-all | Change |
| --- | ---: | ---: | ---: |
| MT-Bench, first turn, 80 prompts | 190.5 tok/s | 215.1 tok/s | +12.9% |
| GSM8K, 100 prompts | 264.0 tok/s | 309.2 tok/s | +17.1% |
| MATH-500, 100 prompts | 261.6 tok/s | 303.2 tok/s | +15.9% |

Acceptance length: 3.80 → 3.72 (MT-Bench), 5.07 → 5.06 (GSM8K), 5.14 → 5.05 (MATH-500).

Fixed prompts, greedy, thinking off, streamed verify steps, n=2:

| prompt | RadixArk + b12x | this checkpoint + b12x | Change |
| --- | ---: | ---: | ---: |
| prose | 143.8 tok/s | 156.3 tok/s | +8.7% |
| code | 297.7 tok/s | 346.4 tok/s | +16.4% |
| math | 321.1 tok/s | 353.1 tok/s | +9.9% |

Decode step 18.8 → 16.3 ms on all three. Interleaved n=6 of the shipping stack (this checkpoint + b12x + bf16 draft versus stock CUTLASS + RadixArk) is 156.6 / 346.4 / 353.2 tok/s, +25 / +33 / +26%.

Code/math recipe on this checkpoint: FP8 DFlash2 draft and 12 draft tokens, n=2: 157 / 396 / 439 tok/s. Prose stays at the shipping number. See [thomgardiner/sglang-sm120-nvfp4](https://github.com/thomgardiner/sglang-sm120-nvfp4).

One `bench_serving` run on random-token prompts (ISL 8192 / OSL 1024, concurrency 1, seed 7) read 196.2 → 263.9 tok/s (+34.5%). Acceptance on that shape moves between runs. Do not quote it as the typical gain.

Not measured: batch sizes above 1, HumanEval on this checkpoint, other engines.

## Artifact identity

| File | Size | sha256 (first 16) |
| --- | ---: | --- |
| model-00001-of-00003.safetensors | 8.7 GB | 1b43868facbccec8 |
| model-00002-of-00003.safetensors | 8.5 GB | f570545a63dc7a7b |
| model-00003-of-00003.safetensors | 0.8 GB | 50d11a5812c1330f |

Base: RadixArk/Qwen3.8-27B-NVFP4, `conversion-manifest.json` carried over unchanged.
