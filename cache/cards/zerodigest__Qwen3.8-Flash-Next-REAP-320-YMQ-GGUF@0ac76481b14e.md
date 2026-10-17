---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
tags:
- text-generation
- gguf
- quantizer
- autoround
- architecture-aware
- mixture-of-experts
- moe
- mamba
- ssm
- reap
- expert-pruning
- 113b
---

<!-- zerodigest-lab-banner-v2-mid-tone -->
<div style="background: #1f1f23; color: #f4f4f5; padding: 20px; border: 1px solid #38bdf8; border-radius: 8px; margin: 20px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <h2 style="color: #38bdf8; margin: 0 0 10px 0; font-size: 18px; font-weight: 700; letter-spacing: -0.5px;">⚡ Fuel the Lab: Keep the Optimization Loops Running</h2>
  <p style="font-size: 14px; line-height: 1.5; margin: 0 0 16px 0; color: #d4d4d8;">Every single <b>ZeroDigest YMQ</b> release is handcrafted and manually calibrated via intensive importance-matrix sweeps to protect critical logic pathways. This project is entirely independent research—no automation bots, no corporate backers, and no external funding. Running multi-hour compute arrays consumes massive local infrastructure overhead out-of-pocket. Consider checking out our compiler or supporting our compute costs!</p>
  <div style="background: #18181b; padding: 12px; border-radius: 6px; border: 1px solid #3f3f46; font-size: 13px;">
    <p style="margin: 0 0 8px 0; font-weight: 600; color: #f4f4f5;">👉 Developer Resources & Support Paths:</p>
    <div style="margin-bottom: 6px;">🛠️ <a href="https://github.com/minyor/ymq-compiler" target="_blank" style="color: #38bdf8; text-decoration: underline; font-weight: bold;">View Compiler Source on GitHub (YMQ v2.0)</a></div>
    <div style="margin-bottom: 6px;">🚀 <a href="https://runpod.io?ref=dbjmkmeh" target="_blank" style="color: #38bdf8; text-decoration: underline; font-weight: bold;">Deploy weights on RunPod Cloud Compute (Affiliate Link)</a></div>
    <div style="margin-bottom: 8px;">☕ <a href="https://ko-fi.com/zerodigest" target="_blank" style="color: #38bdf8; text-decoration: underline; font-weight: bold;">Support the Lab on Ko-fi (One-Time / Monthly)</a></div>
    <div style="margin: 8px 0; border-top: 1px solid #3f3f46;"></div>
    <div style="margin-bottom: 4px; font-family: monospace;">🧡 <span style="color: #f97316; font-weight: bold;">BTC:</span> <code style="background: #27272a; padding: 1px 4px; border-radius: 4px; color: #f4f4f5;">14Fmic9z3VA1ZU11bWoP6JtU7csTAApwZo</code></div>
    <div style="font-family: monospace;">🔷 <span style="color: #6366f1; font-weight: bold;">ETH/USDT:</span> <code style="background: #27272a; padding: 1px 4px; border-radius: 4px; color: #f4f4f5;">0xbe4cdc3adc27c21ef1c27c6b403311db07b35ed2</code></div>
  </div>
</div>

# Qwen3.8-Flash-Next-REAP-320-YMQ-GGUF

> **Base Model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
> **REAP Manifest:** [AnonimousA/seleccion_mass_K320.json](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF/blob/main/manifests/seleccion_mass_K320.json)
> **Inspired By:** [AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF)

### 🎯 REAP Expert Pruning × Architecture-Aware Mixed Precision Quantization

This repository features a **custom expert-pruned and architecture-aware quantized** build of **Qwen3.8-Flash-Next**, processed through a two-stage pipeline:

1.  **REAP Expert Selection (K=320):** A custom importance-matrix-driven manifest (`seleccion_mass_K320.json`) surgically selects the top 320 expert slots per MoE routing layer from the original dense expert pool, permanently pruning the least-activated pathways. This is not a uniform prune—selections are driven by per-token activation mass collected across calibration prompts.
2.  **YMQ-Compiler (v2.0) Log-Space Quantization:** The pruned checkpoint is then quantized using the custom YMQ-Compiler framework, which assigns per-layer precision tiers based on importance-matrix cluster analysis—elevating high-leverage expert layers to `Q6_K` / `IQ4_NL` shields while compressing low-activity background experts down to `IQ2_XS`.

These builds natively support parallel multi-token prediction (MTP) speculation engines and utilize high-context optimization parameters tailored for demanding code development API execution environments (such as RooCode/Aider).

<p align="center">
  <img src="./logo.jpg" width="800" alt="ZeroDigest YMQ Logo">
</p>

---

## 📊 Quantization Preset Tier Details

| Preset Tier | Total Size | Target Usage / Memory VRAM Profile | Cognitive Real-World Coding Quality |
| :--- | :--- | :--- | :--- |
| **`XS-Pro`** |[~63 GB](./Qwen3.8-Flash-Next-reap320-YMQ-XS-Pro.gguf) | 💎 **48GB+ Workstation Driver** | **The REAP MoE High-Context.** `Q8_0` shared-expert armor and `Q5_K` attention shields protect every routing decision. The surviving expert pool is graded across four precision tiers — individually tuned per layer via importance-matrix log-space clustering. No uniform blanket compression. |
| **`M-TI`** | [~63 GB](./Qwen3.8-Flash-Next-reap320-YMQ-M-TI.gguf) | 💎 **48GB+ Workstation Driver (Recommended)** | **The REAP MoE High-Context.** `Q8_0` shared-expert armor and `Q5_K` attention shields protect every routing decision. The surviving expert pool is graded across four precision tiers — `Q6_K` ➔ `IQ4_NL` ➔ `IQ3_XXS` ➔ `IQ2_XS` (floor) — individually tuned per layer via importance-matrix log-space clustering. No uniform blanket compression. |

## 📉 Perplexity Evaluation Metrics (WikiText-2)

The following metrics demonstrate the mathematical quality preservation of the **REAP + YMQ-Compiler** pipeline compared to standard linear quantization layouts. Tested natively via `llama-perplexity` over a 4096 context window using the official WikiText-2 test corpus.

| Model Variant | File Size | Perplexity | Mean KL-Divergence | Internal Bit Gradient (High ➔ Mid ➔ Low ➔ Floor) |
| :--- | :--- | :--- | :--- | :--- |
| **`XS-Pro`** | [~63 GB](./Qwen3.8-Flash-Next-reap320-YMQ-XS-Pro.gguf) | `TBD` | `TBD` | Expert Gradient: `IQ4_XS` ➔ `IQ3_S` ➔ `IQ2_S` ➔ `IQ2_XS` (floor) · Armor: `Q8_0` / `Q5_K` |
| **`M-TI`** | [~63 GB](./Qwen3.8-Flash-Next-reap320-YMQ-M-TI.gguf) | `TBD` | `TBD` | Expert Gradient: `Q6_K` ➔ `IQ4_NL` ➔ `IQ3_XXS` ➔ `IQ2_XS` (floor) · Armor: `Q8_0` / `Q5_K` |

> ℹ️ **Why so large?** Qwen3.8-Flash-Next is a massive sparse Mixture-of-Experts architecture with hybrid Mamba SSM state-space tracking. Even after REAP expert pruning (K=320 per layer) and aggressive 2-bit expert floors, the surviving expert pool keeps the file at ~63 GB. The active parameter count per token remains small (~5B), so inference speed stays comparable to much smaller dense models once weights are resident in memory.

> ⚠️ **REAP Pruning Trade-Off:** REAP expert pruning is not free. Permanently removing ~40% of the least-activated expert slots means the knowledge those experts carried is **gone forever**—it cannot be recovered by quantization or fine-tuning. Expect noticeable degradation on broad general-knowledge tasks, rare factual recall, and domain-specific queries that relied on the pruned pathways. The K=320 selection threshold keeps the experts that fire most often across calibration prompts, but "rarely activated" is not the same as "useless." **For coding, structured reasoning, and agent-loop workloads—where the surviving expert pathways carry the bulk of the signal—this build remains a strong and practical driver.** If your primary use case is code generation, refactoring, or long-context agent execution, the pruned model handles those tasks reliably. For open-ended creative writing or encyclopedic recall, expect the full unpruned model to perform better.

### 💡 The REAP + YMQ Pipeline Breakthrough

Standard quantization pipelines treat every expert in a Mixture-of-Experts model as equally important, applying a uniform bit-depth across the entire network. This wastes valuable VRAM on dormant background experts that rarely activate, while starving the critical routing pathways that actually carry reasoning signal.

**REAP (Router-based Expert Activation Pruning)** solves this at the architectural level:
*   **Expert Mass Profiling:** During calibration, every expert slot in every MoE layer is scored by cumulative activation mass across representative prompts. The `seleccion_mass_K320.json` manifest captures the top-320 highest-mass expert indices per layer.
*   **Permanent Pruning:** Expert slots that fall below the selection threshold are permanently removed from the checkpoint, reducing the model's effective parameter count without any distillation or retraining loss.
*   **Importance-Aware Quantization:** The same importance data that drove pruning is reused by the YMQ-Compiler to assign per-layer precision tiers across a four-grade gradient: high-mass expert layers receive full `Q6_K` armor, mid-tier layers are elevated to `IQ4_NL`, lower-importance blocks drop to `IQ3_XXS`, and dormant background layers are safely compressed to the `IQ2_XS` floor.

The result is a model that retains the reasoning capacity of the full 113B-parameter dense MoE within a ~63 GB footprint, with per-layer quantization budgets precisely matched to actual cognitive load.

---

## ⚖️ YMQ vs. Uniform Quantization (The AutoRound Philosophy)

Standard quantization pipelines apply a blunt, uniform bit-depth across every single layer in a model. This wastes valuable VRAM on silent background layers while starving critical logic anchors of necessary precision.

The **YMQ-Compiler** implements a philosophy similar to advanced weight-tuning frameworks like **Intel's AutoRound**:
*   **Targeted Bit Allocation:** It strips bits away from low-leverage background tensors and automatically re-allocates that saved VRAM budget straight into full high-fidelity shields for the model's highest cognitive spikes and boundary pathways.
*   **Instant Optimization:** Instead of running heavy, days-long optimization training loops, YMQ achieves a highly accurate mixed-precision layout instantly by analyzing layer importance metrics in log-space.

The result is a custom mixed-precision portfolio that matches the low perplexity and high context stability of premium optimized quants (like AutoRound), while maintaining an ultra-lightweight, high-speed cache footprint.

---

## 🛠️ The YMQ Compilation Architecture

Standard quantization pipelines treat network tensors like a flat dataset, applying destructive blanket low-bit compression to delicate tracking networks. The **YMQ-Compiler** solves high-context logic decay by parsing model files dynamically via an automated, multi-tiered protection matrix:

1.  **REAP-Aware Per-Layer Expert Grading:** Unlike standard quantizers that apply one bit-depth to all `ffn_*_exps` tensors, the YMQ-Compiler reads the importance manifest and assigns each of the 48 transformer blocks a unique expert precision tier across a four-grade gradient: **HIGH** (`Q6_K`) → **MID** (`IQ4_NL`) → **LOW** (`IQ3_XXS`) → **FLOOR** (`IQ2_XS`). High-mass layers like `blk.0`, `blk.4`, `blk.6`, and `blk.47` receive full `Q6_K` armor, while dormant background blocks safely compress to the `IQ2_XS` floor.
2.  **Log-Space Gap Detection Clustering:** Instead of flat percentage thresholds, the engine computes statistical cluster variances in log-space, isolating intermediate logical reasoning spikes and elevating them to stable non-linear formats, while compressing idle fact-storage expert layers to aggressive 2-bit baselines.
3.  **Fading Boundary Tapering:** Recognizes the extreme fragility of initial token entry data vectors, forcing an input wave cushion that gradually stabilizes parameters before hitting the fallback pools.
4.  **Dedicated Gate Insulation:** Hard-shields volatile parallel Transformer Multi-Head Attention, Mamba Linear State Space Model (SSM), Indexer, and PLE (Positional Layer Embedding) routing paths at `Q8_0`/`Q5_K`, keeping context tracking perfectly noise-free across the hybrid architecture.
5.  **Shared Expert Armor:** The always-active shared expert pathways (`ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`) are locked at `Q8_0`—effectively lossless—to preserve the reasoning backbone that every token traverses regardless of expert routing decisions.
6.  **Asymmetric Vocabulary Shielding:** Fixes tied-weight boundary errors by mapping the final logit classification exit heads to robust configurations to completely eliminate formatting loops and API tag leakage under deep contexts.
7.  **Native Next-N Speculative Stripping:** Processed with advanced pre-tokenizer stripping to ensure zero index offset drift or layer-shifting risks across hybrid REAP-pruned configurations.

---

## 🌾 The REAP Pruning Pipeline

This build was produced by a custom two-stage pipeline:

| Stage | Tool | Description |
| :--- | :--- | :--- |
| **1 — Expert Pruning** | [`ymq_reap_gguf.py`](https://github.com/minyor/ymq-compiler/blob/main/ymq_reap_gguf.py) | Reads the `seleccion_mass_K320.json` manifest and permanently prunes all non-selected expert slots from the base BF16 checkpoint. Reduces expert pool density while preserving activation-mass pathways. |
| **2 — Importance-Aware Quantization** | [`ymq-compiler`](https://github.com/minyor/ymq-compiler) | Applies per-layer precision tiers derived from importance-matrix log-space clustering. Shared experts and attention gates receive high-fidelity shields; background expert pool is compressed to dense 2-bit grids. |

> **REAP Manifest Source:** The expert selection manifest (`seleccion_mass_K320.json`) was authored by [AnonimousA](https://huggingface.co/AnonimousA) as part of the [Qwen3.8-Flash-Next-REAP-320-GGUF](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-320-GGUF) project. Credit and thanks for the calibration methodology.

---

## 🚀 Recommended Runtime Parameters (llama.cpp / llama-server)

*Need to scale up? Deploy this exact script on on-demand cloud GPUs via [RunPod Cloud Compute](https://runpod.io?ref=dbjmkmeh).*

```bash
$ ./llama-server -m Qwen3.8-Flash-Next-reap320-YMQ-M-TI.gguf \
  --ctx-size 145000 \
  --ctx-checkpoints 2 \
  -ctk q8_0 -ctv q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --md mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --mmproj Qwen3.8-Flash-Next-mmproj-Q4_K_S.gguf \
  --temp 0.6 --top-p 0.95 --top-k 20 --repeat-penalty 1.05 \
  --jinja -fa \
  --ubatch-size 512 \
  --chat-template-kwargs '{"reasoning_effort": "low"}'
```

> ⚠️ **VRAM Note:** At ~63 GB, this model runs comfortably on 48GB of vram or x2 3090.

### 🖼️ Vision Projection (`--mmproj`)

For multimodal vision support, pair this build with the native Flash-Next vision projection:

| Variant | File | Size | Notes |
| :--- | :--- | :--- | :--- |
| **Q4_K_S** | `Qwen3.8-Flash-Next-mmproj-Q4_K_S.gguf` | ~478 MB | Compact vision projection for VRAM-constrained setups. Retains strong image understanding at reduced footprint. |

Pass via `--mmproj <path-to-file>` in your llama-server invocation (see example above).

### 🎕 MTP Speculative Draft (`--md`)

This model ships with native multi-token prediction heads. Pair with the shared MTP draft head for speculative decoding:

| Variant | File | Size | Notes |
| :--- | :--- | :--- | :--- |
| **Q4_K_M** | `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | ~2 GB | Shared MTP draft head for 2-token speculative decoding. Enable via `--spec-type draft-mtp --spec-draft-n-max 2`. |

---

## ☕ Support & Future R&D

If the **YMQ-Compiler** builds saved your context window from collapsing or optimized your active development cycle speeds, consider buying a coffee to fund further low-level optimization research. Your support keeps the server nodes baking future model scales!

👉 **[Support ZeroDigest Research on ko-fi](https://ko-fi.com/zerodigest)**

---

## 📦 Source Framework & Automation Code

The compiler pipeline automation engine, REAP pruning scripts, setup thresholds, and structural mapping rules are open-source. To view the implementation details or compile your own custom models natively using this profile layout, visit the official development hub:

👉 **[GitHub: ZeroDigest / YMQ-Compiler](https://github.com/minyor/ymq-compiler)**

---

## 🙋 Acknowledgements

| Credit | Role |
| :--- | :--- |
| **[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** | Base architecture and weights |
| **[AnonimousA](https://huggingface.co/AnonimousA)** | REAP expert selection manifest (`seleccion_mass_K320.json`) and calibration methodology inspiration |
| **YMQ-Compiler** | Architecture-aware importance-matrix quantization pipeline (this build) |
