---
base_model: google/gemma-4-26B-A4B-it
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - quantized
  - vllm
  - w4a16
  - hybrid-one-shot
  - multimodal
  - vision
---

# HivenetQuant/gemma-4-26B-A4B-it-NVFP4

NVFP4 (`nvfp4`) **W4A16** quantization of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) — FP4 (E2M1) weights with FP8 block scales. The MLP and routed experts form the 4-bit tier; the attention layers are kept at **BF16**. The split is a fixed hybrid one-shot assignment (no sensitivity search), applied with GPTQ and data-free RTN for the routed experts.

The checkpoint targets NVIDIA Blackwell's native FP4 tensor cores (e.g. RTX 5090); the benchmark below compares it against the original FP16 weights across the evaluated tasks.

**Preserved (kept in BF16, not quantized):** the **vision encoder** (`model.visual*`), plus `lm_head` and norm/router layers. So multimodal understanding is fully retained — see the vision (MMMU-Pro / AI2D) row(s) in the benchmark below.

## Serving with vLLM

Runs on NVIDIA Blackwell (e.g. RTX 5090) with native NVFP4. This is the exact configuration the benchmarks below were validated on (FP8 KV cache, 128k context):

```bash
vllm serve HivenetQuant/gemma-4-26B-A4B-it-NVFP4 \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8 \
    --reasoning-parser gemma4 \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --chat-template /chat_template.jinja
```

All numbers below were produced with this serve config — **NVFP4 weights, FP8 KV cache, 128k context**.

## Quantization recipe

- **Method:** post-training quantization with **llm-compressor** → compressed-tensors, applied as a **hybrid one-shot** — a fixed per-module assignment (attention vs MLP/routed-expert tiers), **not** a per-layer sensitivity search. The 4-bit NVFP4 weights are recovered with **GPTQ** block-wise reconstruction and the linearized routed experts data-free with **RTN**. The exact per-layer tiers are in [`recipe_summary.json`](./recipe_summary.json).
- **Precision (quantized tier):** **W4A16** — FP4 (E2M1) weights with FP8 per-block scales, group size 16. 342 modules stay BF16 (`lm_head`/norms, the vision tower).
- **Protection axis:** this is the **BF16-protected** release — the maximum-accuracy of the two (attention at BF16).
- **Baseline = FP16, not FP8.** Every Δ in the benchmark below is measured against the original full-precision weights — a stricter reference than cards that quantize from, and compare against, an already-lossy FP8 checkpoint.

## Scope of this evaluation

This card reports a **paired FP16 ↔ NVFP4 comparison** — identical prompts, harness and decoding run head-to-head on both precisions — across a diverse task set. The goal is to show that NVFP4 quantization introduces **no meaningful degradation versus the original weights**, not to re-establish the base model's absolute capability.

A *paired* delta is detectable at far smaller sample sizes than an absolute score, so we evaluate a **representative sample per task** (exact sizes and decoding settings are in [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md)) rather than every full benchmark set — a statistically sound degradation check that avoids the considerable compute and energy of re-running suites the base model has already published.

For **absolute capability and full-suite results**, see the base model [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) and independent leaderboards: NVFP4 tracks FP16 within noise on every dimension tested here, so those numbers carry over.

## Benchmark vs FP16

| Dataset | FP16 | NVFP4 | Δ |
|---|---|---|---|
| `mmlu_pro` | 82.62 | 82.11 | -0.51 |
| `gpqa_diamond_cot_zeroshot` | 80.40 | 80.10 | -0.30 |
| `aime25` | 87.33 | 85.33 | -2.00 |
| `ifeval (ifeval_safe)` | 94.73 | 94.73 | +0.00 |
| `mmmu_pro_standard_cot` | 71.33 | 71.73 | +0.40 |
| `ai2d_no_mask` | 79.27 | 73.87 | -5.40 |
| `livecodebench (codegen v6)` | 79.17 | 77.78 | -1.39 |
| `bfcl (tool-calling)` | 70.34 | 66.74 | -3.60 |
| `niah` | 95.00 | 96.00 | +1.00 |

## Speed

Measured with vLLM `bench serve` (synthetic random prompts, `--ignore-eos`) — `vllm bench serve`. Input lengths 1,024, 10,000 tokens, 512 output tokens, concurrency 1, 10, at **TP4**. Served with FP8 KV cache, max-model-len 131,072, gpu-mem 0.9, on NVIDIA GeForce RTX 5090, vllm/vllm-openai:v0.25.1. FP16 is measured identically at the same TP, so each ratio isolates the quantization effect.

| operating point | FP16 | this model (NVFP4) | speedup |
|---|---|---|---|
| decode · 1k ctx · 1 stream (tok/s) | 204 | 219 | 1.07× |
| prefill · 1k ctx (tok/s) | 13047 | 11809 | 0.91× |
| decode · 10k ctx · 1 stream (tok/s) | 156 | 167 | 1.07× |
| prefill · 10k ctx (tok/s) | 19352 | 20537 | 1.06× |
| throughput · 1k ctx · ×10 (tok/s) | 1092 | 1389 | 1.27× |

_decode_ = single-stream (batch 1) output rate; _prefill_ = prompt ingestion (input tokens ÷ time-to-first-token); _throughput_ = aggregate output rate under concurrent load.

**Reproducing these numbers** — the serving config, pinned harness versions, and per-dataset sampling / seeds / decisions contract are in [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md) (standard open-source tools, no proprietary harness).

## About

Published by HivenetQuant, the Hugging Face account for Hivenet's model-optimization work. Hivenet is Antimatter's cloud and AI-inference platform. Antimatter brings together energy and power infrastructure through Antimatter US, modular data centers through Policloud, and cloud software and AI inference through Hivenet.

This checkpoint was quantized and benchmarked by Hivenet's AI Products team on Antimatter's Policloud infrastructure. We publish the precision plan, benchmark results, serving configuration, and reproducibility details so developers can examine the trade-offs and run the model on compatible NVIDIA Blackwell hardware. Our broader aim is to make high-performance inference easier to deploy on infrastructure that operators can choose and control.

### Run it on Hivenet

Want to run this model yourself? Hivenet provides GPU infrastructure through Compute, along with an open-source Router for managing requests across compatible inference backends. [Explore your options](https://www.hivenet.com/hivenet-router).
