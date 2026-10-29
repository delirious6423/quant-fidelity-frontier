---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- nvfp4
- quantized
- vllm
- w4a4
- auto-mixed-precision
- multimodal
- vision
- mtp
- speculative-decoding
- qwen3_8
---

# HivenetQuant/Qwen3.8-27B-NVFP4

NVFP4 (`nvfp4`) **W4A4** quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) — FP4 (E2M1) weights and activations with FP8 block scales. The MLP is the 4-bit tier; the attention and recurrent (DeltaNet) layers the search protects are kept at **FP8**. The per-layer split is chosen automatically by a weight-sensitivity search and applied with GPTQ.

The checkpoint targets NVIDIA Blackwell's native FP4 tensor cores (e.g. RTX 5090); the benchmark below compares it against the original FP16 weights across the evaluated tasks.

**Preserved (kept in BF16, not quantized):** the **vision encoder** (`model.visual*`) and the **MTP speculative-decoding head** (`mtp.*`), plus `lm_head` and norm/router layers. So multimodal understanding and MTP speculative decoding are fully retained — see the vision (MMMU-Pro / AI2D) and MTP accept-rate row(s) in the benchmark below.

## Serving with vLLM

Runs on NVIDIA Blackwell (e.g. RTX 5090) with native NVFP4. This is the exact configuration the benchmarks below were validated on (FP8 KV cache, 256k context):

```bash
vllm serve HivenetQuant/Qwen3.8-27B-NVFP4 \
    --tensor-parallel-size 2 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --kv-cache-dtype fp8 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

**With MTP speculative decoding** (the multi-token-prediction head is preserved) add one flag for faster decoding:

```bash
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

All numbers below were produced with this serve config — **NVFP4 weights, FP8 KV cache, 256k context**.

## Quantization recipe

- **Method:** post-training quantization with **llm-compressor** → compressed-tensors. The per-layer **NVFP4-vs-FP8** split is chosen automatically by a weight-sensitivity search (ModelOpt supernet probe + Kneedle elbow), then the NVFP4 weights are recovered with **GPTQ** block-wise reconstruction. The exact per-layer tiers are in [`recipe_summary.json`](./recipe_summary.json).
- **Precision (quantized tier):** **W4A4** — FP4 (E2M1) weights and activations with FP8 per-block scales, group size 16. **150** attention/recurrent layers are kept at FP8 (the protection tier); 207 modules stay BF16 (`lm_head`/norms, the vision tower, the MTP head).
- **Protection axis:** this is the **FP8-protected** release — the smaller, faster of the two (attention/recurrent at FP8).
- **Baseline = FP16, not FP8.** Every Δ in the benchmark below is measured against the original full-precision weights — a stricter reference than cards that quantize from, and compare against, an already-lossy FP8 checkpoint.

## Scope of this evaluation

This card reports a **paired FP16 ↔ NVFP4 comparison** — identical prompts, harness and decoding run head-to-head on both precisions — across a diverse task set. The goal is to show that NVFP4 quantization introduces **no meaningful degradation versus the original weights**, not to re-establish the base model's absolute capability.

A *paired* delta is detectable at far smaller sample sizes than an absolute score, so we evaluate a **representative sample per task** (exact sizes and decoding settings are in [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md)) rather than every full benchmark set — a statistically sound degradation check that avoids the considerable compute and energy of re-running suites the base model has already published.

For **absolute capability and full-suite results**, see the base model [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and independent leaderboards: NVFP4 tracks FP16 within noise on every dimension tested here, so those numbers carry over.

## Benchmark vs FP16

| Dataset | FP16 | NVFP4 | Δ |
|---|---|---|---|
| `mmlu_pro` | 77.50 | 77.26 | -0.24 |
| `gpqa_diamond_cot_zeroshot` | 88.79 | 88.08 | -0.71 |
| `aime25` | 98.00 | 96.00 | -2.00 |
| `ifeval (ifeval_safe)` | 90.57 | 90.94 | +0.37 |
| `mmmu_pro_standard_cot` | 75.07 | 73.73 | -1.33 |
| `ai2d_no_mask` | 85.47 | 84.27 | -1.20 |
| `livecodebench (codegen v6)` | 87.85 | 86.11 | -1.74 |
| `bfcl (tool-calling)` | 80.14 | 79.18 | -0.96 |
| `niah` | 100.00 | 100.00 | +0.00 |
| `mtp` | 62.32 | 61.36 | -0.97 |

## Speed

Measured with vLLM `bench serve` (synthetic random prompts, `--ignore-eos`) — `vllm bench serve`. Input lengths 1,024, 10,000 tokens, 512 output tokens, concurrency 1, 10, at **TP4**. Served with FP8 KV cache, max-model-len 262,144, gpu-mem 0.9, on NVIDIA GeForce RTX 5090, vllm/vllm-openai:v0.26.0. FP16 is measured identically at the same TP, so each ratio isolates the quantization effect.

| operating point | FP16 | this model (NVFP4) | speedup |
|---|---|---|---|
| decode · 1k ctx · 1 stream (tok/s) | 71 | 102 | 1.44× |
| prefill · 1k ctx (tok/s) | 5553 | 7039 | 1.27× |
| decode · 10k ctx · 1 stream (tok/s) | 59 | 85 | 1.44× |
| prefill · 10k ctx (tok/s) | 6544 | 8928 | 1.36× |
| throughput · 1k ctx · ×10 (tok/s) | 469 | 675 | 1.44× |

_decode_ = single-stream (batch 1) output rate; _prefill_ = prompt ingestion (input tokens ÷ time-to-first-token); _throughput_ = aggregate output rate under concurrent load.

With **MTP speculative decoding** enabled (the `--speculative-config` flag above), single-stream decode is substantially faster still — the preserved MTP head accepts a substantial fraction of drafted tokens (the measured accept rate is the `mtp` row in the benchmark).

**Reproducing these numbers** — the serving config, pinned harness versions, and per-dataset sampling / seeds / decisions contract are in [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md) (standard open-source tools, no proprietary harness).

## About

Published by HivenetQuant, the Hugging Face account for Hivenet's model-optimization work. Hivenet is Antimatter's cloud and AI-inference platform. Antimatter brings together energy and power infrastructure through Antimatter US, modular data centers through Policloud, and cloud software and AI inference through Hivenet.

This checkpoint was quantized and benchmarked by Hivenet's AI Products team on Antimatter's Policloud infrastructure. We publish the precision plan, benchmark results, serving configuration, and reproducibility details so developers can examine the trade-offs and run the model on compatible NVIDIA Blackwell hardware. Our broader aim is to make high-performance inference easier to deploy on infrastructure that operators can choose and control.

### Run it on Hivenet

Want to run this model yourself? Hivenet provides GPU infrastructure through Compute, along with an open-source Router for managing requests across compatible inference backends. [Explore your options](https://www.hivenet.com/hivenet-router).