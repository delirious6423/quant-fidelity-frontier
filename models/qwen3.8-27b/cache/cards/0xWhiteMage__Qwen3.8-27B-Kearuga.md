---
library_name: transformers
license: apache-2.0
language:
  - en
  - zh
  - de
  - fr
  - ja
  - ko
  - es
  - ru
pipeline_tag: text-generation
tags:
  - kearuga
  - qwen3
  - qwen3.5
  - speculative-decoding
  - dflash2
  - gptq
  - nvfp4
  - fp8
  - hybrid-quantization
  - gb10
  - dgx-spark
  - grace-blackwell
  - sm121
base_model: Qwen/Qwen3.8-27B
model_type: qwen3_5
---

# 🧙‍♂️ Qwen3.8 27B Kearuga

<p align="center">
  <img src="https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga/resolve/main/assets/header.png" width="100%"><br><a href="https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2"><img src="https://img.shields.io/badge/%F0%9F%A4%97_DFlash_2-Kearuga_Drafter-orange.svg" alt="Kearuga DFlash 2 Drafter"></a> <a href="https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2"><img src="https://img.shields.io/badge/%F0%9F%A4%97_DFlash_2-Stock_Drafter_(baseline)-yellow.svg" alt="Stock DFlash 2 Drafter (baseline)"></a> <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a> <a href="https://x.com/0xWhiteMage" target="_blank"><img src="https://img.shields.io/badge/X-@0xWhiteMage-000000?logo=x&logoColor=white" alt="Follow on X"></a> <a href="https://ko-fi.com/0xwhitemage" target="_blank"><img src="https://img.shields.io/badge/Ko--fi-Donate-FF5E5B?logo=ko-fi&logoColor=white" alt="Donate on Ko-fi"></a>
</p>

---

## ⚡ Overview

**Qwen3.8 27B Kearuga** is a custom **tiered sensitivity hybrid-quantized** checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), specifically engineered for high-throughput and interactive speculative-decoding serving on a single **NVIDIA DGX Spark (GB10 / Grace-Blackwell, 128 GB Unified Memory)**.

By pairing **GPTQ with Four-Over-Six (4o6) group scales**, **NVFP4 AWQ**, **FP8 E4M3**, and **protected BF16**, Kearuga:
* **Cuts weight footprint by 52%** (from 51.8 GiB unquantized base down to **24.85 GB** across 3 shards + MTP draft head).
* **Preserves output fidelity to within a 0.0165 mean KL divergence** against the unquantized BF16 base model, maintaining **40/40 (100%) top-1 argmax agreement**.
* **Unlocks fast speculative serving with SGLang**: with the [Kearuga DFlash 2 drafter v1.0](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) (K=12, NVFP4, 64K draft head) the serving repo's `bench/scale.py` probe measures **C1 44.7 · C2 78–94 · C4 131 tok/s aggregate** (T=0, thinking off, 512 forced tokens, aggregate incl. TTFT 0.23–0.34 s; identical with the GPU clock lock on or off), and the paired 50-prompt battery puts it **+14.5 % C1 / +10.8 % C4 over the stock `z-lab` drafter** (C1 net-decode medians: code 57.6, math 62.2, tool calls 114.6, prose 21.6 tok/s). Serving stack: [GitHub v0.6.6](https://github.com/0xWhiteMage/Qwen3.8-27B-Kearuga-SGLang-DGX-Spark-DFlash2); drafter release **v1.0**. Without speculative decoding the same target measures 10.30 tok/s C1 / 39.56 tok/s C4 on that battery (serving stack v0.6.7) — a 3.46× / 2.75× gain from the drafter.
* **Runs on official SGLang Docker images** — the Kearuga-drafter profile adds a 5-file read-only Python overlay (hash-verified at launch, no Docker build); the stock profile needs no overlay.

---

## 📊 Kearuga vs. Base Qwen3.8-27B (BF16)

The table below presents fidelity and quality metrics measured on an isolated NVIDIA DGX Spark (GB10 / SM121, 128 GB Unified Memory) running SGLang (KV cache in BF16, thinking disabled, temperature 0). They do not depend on the drafter — the target verifies every drafted token:

| Metric | Base Model: `Qwen/Qwen3.8-27B` (Native BF16) | **Qwen3.8 27B Kearuga** (Hybrid Quantized) | Delta / Operational Significance |
|---|:---:|:---:|:---|
| **Weight Footprint** | 51.8 GiB | **24.85 GB** | **−52.0%** (leaves ≈ 90 GB of the 121 GB unified memory for KV cache, drafter and runtime) |
| **Fidelity-40 Mean KL Divergence** | 0.0000 | **0.0165** | **−50.6% KL reduction** vs legacy NVFP4 baseline (0.0334) |
| **Fidelity-40 Mean Jensen-Shannon (JS)** | 0.0000 | **0.0034** | Virtually identical distribution profile |
| **Fidelity-40 Top-1 Agreement** | 40 / 40 | **40 / 40 (100%)** | Zero argmax token divergence across 40 evaluation prompts |
| **Fidelity-40 Exact 32-Token Continuation** | 40 / 40 | **20 / 40 (50.0%)** | 50% byte-identical; remaining 50% diverge only at near-tie logits |
| **Held-Out Full-Vocab KL (72,082 tokens)** | 0.0000 | **0.0208** | Verified on held-out text sequences |
| **Held-Out Top-1 Agreement** | 100.0% | **95.0%** | Exceptional semantic adherence across broad corpora |
| **Quality-200 Objective Score** | — | **157 / 180** | GSM8K: 66/80 · HumanEval: 39/40 · IFEval: 34/40 · Agentic: 18/20 — 155/180 served with the Kearuga drafter (near-tie band 154–159) |
| **Decode throughput (Kearuga drafter, `bench/scale.py`)** | — | **C1 44.7 · C2 78–94 · C4 131 tok/s aggregate** | T=0, thinking off, 512 forced tokens, aggregate incl. TTFT (0.23 / 0.28 / 0.33 s); measured 2026-09-17, identical with the GPU clock lock on or off (the decode loop is memory-bandwidth-bound) |
| **Decode throughput without speculation (same target, no drafter, serving-repo 50-prompt battery)** | — | **C1 10.30 · C4 39.56 tok/s aggregate** (flat ≈ 10.3 tok/s per stream) | Thinking off, T=0, single tenant, 2026-09-23; the production configuration (Kearuga drafter K=12, 64K head, FP8 draft KV) measures 35.66 / 108.83 on the same battery → the drafter is worth **3.46× C1 / 2.75× C4** (serving stack v0.6.7) |
| **Serving footprint at --mem-fraction-static 0.85** | — | **~100.6 GB static** (24.9 GB weights + 56.9 GB BF16 target KV + 8.9 GB FP8 drafter KV for 932,355 tokens + 5.8 GB GDN state + drafter + graphs) | Measured at the v0.6.6 production boot (2026-09-22); 4 seats × 262K window sharing a 932K-token pool (818,294 tokens with BF16 drafter KV, v0.6.5 and earlier) |

**Kearuga drafter (K=12, NVFP4, 64K draft head).** In a paired bake-off on the same image, target, flags and clock — 50-prompt battery (10 prompts × 5 domains) × 2 runs, temperature 0, thinking off; "net decode tok/s" = (completion tokens − 1)/(last − first token time) per request, TTFT excluded — the [Kearuga DFlash 2 drafter](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) with the draft-head overlay reached **35.33 tok/s C1 (+14.5 %)** and **108.90 tok/s C4 (+10.8 %)** aggregate vs the stock drafter (30.85 / 98.31), better on all 10 domain cells. C1 net-decode medians: code 46.27 → 57.62, math 51.21 → 62.18, tool calls 80.82 → 114.63, prose 19.45 → 21.64, instruction-following 19.87 → 21.80. The target still verifies every drafted token, so output is unchanged (Fidelity-40 KL 0.0165, top-1 40/40). Full tables on the [drafter model card](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2).

---

## 🛡️ Tiered Sensitivity Quantization Architecture

Uniform quantization (applying INT4 or NVFP4 indiscriminately across all layers) degrades attention sink states, vocabulary distribution tails, and speculative draft features. 

Inspired by empirical mixed-precision sensitivity research (EXL3 tiered sensitivity), **Kearuga partitions the model's 2,009 tensors into four distinct sensitivity tiers**:

| Tier | Module Scope | Precision | Count | Architectural Purpose |
|---|---|:---:|:---:|---|
| **Tier 1 (Protect)** | `embed_tokens`, `lm_head`, all RMSNorms, visual tower (27 blocks / 333 weights), and MTP head tensors | Native BF16 | 348 | Protects vocabulary tail, multi-modal reasoning, and MTP draft survival |
| **Tier 2 (Medium)** | Self-Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`), GDN Recurrence (`in_proj`) (Layers 2–61), and Boundary MLPs (Layers 0, 1, 62, 63) | FP8 E4M3 (Block / Static scales) | 205 (195 + 10) | Keeps DFlash 2 tap layers `[5, 19, 33, 47, 61]` pristine; stabilizes recurrences |
| **Tier 3 (Core)** | MLP `gate_proj` & `up_proj` (Layers 2–61, 60 layers × 2) | W4A16 NVFP4 (GPTQ-4o6) | 120 | GPTQ with Four-Over-Six group scaling eliminates 21.7% of weight-state KL divergence |
| **Tier 4 (Down)** | MLP `down_proj` (Layers 2–61, 60 layers × 1) | NVFP4 AWQ (Pre-quantized) | 60 | Preserves outlier scales via calibrated channel activation protection |


### 1. Four-Over-Six (4o6) Group Scaling
Standard GPTQ assigns a single static exponent code per group block (amax → code 6). Kearuga's **4o6 algorithm** evaluates dynamic range block-by-block and selects the optimal scale:
* **Code 6**: Standard headroom for high-variance weights.
* **Code 4**: Higher numerical precision for weights with tight local distributions.

In our calibration, **44.7% of blocks selected Code 4**, yielding a **16% reduction in Hessian-weighted MSE** (0.529 → 0.445) and cutting held-out KL divergence by **21.7%** without increasing tensor bit-width or memory footprint.

### 2. Fused-Shared Global Scale Contract
SGLang's high-performance Blackwell kernels fuse `gate_proj` and `up_proj` into a single GEMM operation, indexing a unified `weight_scale_2.max()`. Quantizing gate and up independently introduces scale divergence that breaks dequantization. Kearuga enforces an exact **fused-shared global scale** across every layer:

```text
scale_shared = max(amax(gate), amax(up))
```

This ensures bit-exact kernel compatibility and eliminates dequantization distortion.

---

## 🚀 Serving & Deployment Guide

Kearuga is served directly via official SGLang Docker container images (`sha256:616a3e97…` or later) with native DFlash 2 speculative acceleration. The recommended drafter is [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) **v1.0** (Kearuga-distilled, NVFP4, K=12):

```bash
docker run --gpus all \
  --shm-size 32g \
  --ulimit memlock=-1:-1 \
  --cap-add IPC_LOCK \
  -p 8888:8888 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe \
  python3 -m sglang.launch_server \
    --model-path 0xWhiteMage/Qwen3.8-27B-Kearuga \
    --served-model-name "Qwen3.8 27B Kearuga" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path 0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2 \
    --speculative-draft-model-quantization modelopt_fp4 \
    --speculative-num-draft-tokens 12 \
    --speculative-draft-window-size 2048 \
    --kv-cache-dtype bf16 \
    --mem-fraction-static 0.85 \
    --max-total-tokens 1048576 \
    --context-length 262144 \
    --max-running-requests 4 \
    --host 0.0.0.0 \
    --port 8888
```

The 64K draft-head overlay adds a further ≈ 6 % on top of plain SGLang (35.33 vs 33.30 tok/s C1 aggregate) — it is documented on the [drafter card](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) and is handled automatically by the serving repo's launcher `./start-dflash2.sh`.

**Alternative: stock drafter (K=10)** — the serving repo's `DRAFTER_PROFILE=stock` (the original v0.5.0 recipe) with [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2): use the command above with `--speculative-draft-model-path z-lab/Qwen3.8-27B-DFlash2`, no `--speculative-draft-model-quantization`, and `--speculative-num-draft-tokens 10`.

### Key Serving Parameters

| Parameter | Recommended Setting | Rationale |
|---|:---:|---|
| `--speculative-num-draft-tokens` | `12` with the Kearuga drafter | K=8 loses code/math acceptance 7–12 %; K=14/16 give nothing at C1 and lose 2–6 % at C4. `10` with the stock drafter. |
| `--kv-cache-dtype` | `bf16` | Preserves full logit fidelity (KL: 0.0170 → 0.0165, exact continuations: 20/40 vs 19/40) with negligible compute cost. |
| `--context-length` | `262144` | Full native context supported without YaRN interpolation. |
| `--mem-fraction-static` | `0.85` + `--max-total-tokens 1048576` | The validated production pair: at 0.85 the pool comes out at 932,355 tokens with FP8 drafter KV (serving stack v0.6.6; 818,294 with BF16 drafter KV) — the cap is a ceiling, not the allocation — leaving ≈ 15 GB GPU-side headroom for CUDA graphs and the OS. |

---

## ⏱️ Saturated Priority Scheduling

In multi-agent production pipelines, background batch traffic can cause interactive request freezes. Kearuga's serving stack enables native priority preemption:

```json
{
  "model": "Qwen3.8 27B Kearuga",
  "priority": 100,
  "messages": [
    {"role": "user", "content": "Analyze this mission-critical trace..."}
  ]
}
```

| Server Load State | Default Priority TTFT | Interactive Priority (`"priority": 100`) | Latency Reduction |
|---|:---:|:---:|:---:|
| **DFlash 2 (All 4 Seats Saturated)** | ~43.15 s | **~2.63 s** | **93.9% faster** |

---

## 🧪 Comprehensive Verification & Quality Gates

Every promoted build of Kearuga must satisfy a strict **15-gate fail-closed verification harness** prior to promotion:

1. **Static Tensor Census**: 2,009 tensors intact across 3 shards; exactly 15 MTP tensors and 333 visual tensors confirmed byte-identical to base BF16.
2. **Serving Contract Audit**: Strict verification that fused `gate_proj` + `up_proj` `weight_scale_2` are identical across all layers.
3. **Semantic Canaries**: 100% pass on exact arithmetic (`19 × 23 = 437`) and logic traps (`9.9 vs 9.11`).
4. **Long-Context NIAH**: 3/3 Needle-In-A-Haystack retrieval at 64K depth (25%, 50%, 90%).
5. **Quality-200 Benchmark**: 157/180 verified score on GSM8K, HumanEval, IFEval, and agentic coding (155/180 served with the Kearuga drafter — near-tie band 154–159).
6. **Multi-Boot Stability**: 3 consecutive clean cold boots with zero loader warnings or device-side asserts.

---

## 📦 Model Files & Shards

| File Name | File Size | Description & Scope |
|---|:---:|---|
| `config.json` | 8.2 KB | Architectural model hyperparameters & quantization configuration |
| `generation_config.json` | 240 B | Sampling defaults, EOS tokens, and generation bounds |
| `hf_quant_config.json` | 14.1 KB | Detailed ModelOpt mixed-precision tier mappings & scale rules |
| `model.safetensors.index.json` | 185 KB | Complete tensor index mapping all 2,009 tensors to shards |
| `model-00001-of-00003.safetensors` | 9.96 GB | Shard 1 (layers 0–21 weights & attention projections) |
| `model-00002-of-00003.safetensors` | 9.99 GB | Shard 2 (layers 22–43 weights & attention projections) |
| `model-00003-of-00003.safetensors` | 4.94 GB | Shard 3 (layers 44–63 weights & lm_head) |
| `model-mtp.safetensors` | 0.85 GB | Native Multi-Token Prediction (MTP) draft head (BF16) |
| `tokenizer.json` | 11.4 MB | Full BPE vocabulary (151,646 tokens) |
| `tokenizer_config.json` | 7.3 KB | Chat templates, special tokens, and tokenizer definitions |


---

## 🔗 Related Resources

* **Kearuga DFlash 2 Drafter v1.0**: [`0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2`](https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga-DFlash2) — Kearuga-distilled, Kearuga-calibrated NVFP4 drafter + 64K draft head (served by serving stack v0.6.6)
* **Stock DFlash 2 Drafter (baseline / fallback)**: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
* **Base Model (BF16)**: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
* **Creator / Maintainer**: [The White Mage](https://benjaminang.tv) · [@0xWhiteMage on X](https://x.com/0xWhiteMage)

---

## 📜 Citation

If you use Kearuga or build upon its tiered sensitivity architecture in your work, please cite:

```bibtex
@misc{kearuga-2026,
  title={Kearuga: Hybrid GPTQ-4o6 + FP8 Quantization of Qwen3.8-27B for Speculative-Decoding Serve on NVIDIA DGX Spark},
  author={0xWhiteMage},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/0xWhiteMage/Qwen3.8-27B-Kearuga}
}
```

## 📄 License
This checkpoint is released under the **Apache 2.0 License**. The underlying base model `Qwen/Qwen3.8-27B` is subject to its original license terms from Alibaba Cloud.
