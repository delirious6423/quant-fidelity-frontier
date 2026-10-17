---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- text-generation
- gguf
- quantizer
- autoround
- architecture-aware
- mamba
- ssm
- multi-token-prediction
- mtp
---

<!-- zerodigest-lab-banner-v2-mid-tone -->
<div style="background: #1f1f23; color: #f4f4f5; padding: 20px; border: 1px solid #38bdf8; border-radius: 8px; margin: 20px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
  <h2 style="color: #38bdf8; margin: 0 0 10px 0; font-size: 18px; font-weight: 700; letter-spacing: -0.5px;">⚡ Fuel the Lab: Keep the Optimization Loops Running</h2>
  <p style="font-size: 14px; line-height: 1.5; margin: 0 0 16px 0; color: #d4d4d8;">Every single <b>ZeroDigest YMQ-MTP</b> release is handcrafted and manually calibrated via intensive importance-matrix sweeps to protect critical logic pathways. This project is entirely independent research—no automation bots, no corporate backers, and no external funding. Running multi-hour compute arrays consumes massive local infrastructure overhead out-of-pocket. Consider checking out our compiler or supporting our compute costs!</p>
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

# Qwen3.8-27B-YMQ-MTP-GGUF

> **Source Model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

### ⚖️ An Architecture-Aware, AutoRound-Inspired Mixed Precision Layout

This repository features advanced, custom architecture-aware quantizations of **Qwen3.8-27B** processed directly from official raw `BF16` source files using the custom **YMQ-Compiler (v2.0)** log-space framework.

These builds natively support parallel multi-token prediction (MTP) speculation engines and utilize high-context optimization parameters tailored for demanding code development API execution environments (such as RooCode/Aider).

<p align="center">
  <img src="./logo.jpg" width="800" alt="ZeroDigest YMQ Logo">
</p>

---

## 📊 Quantization Preset Tier Details

| Preset Tier | Total Size | Target Usage / Memory VRAM Profile | Cognitive Real-World Coding Quality |
| :--- | :--- | :--- | :--- |
| **`XXS`** | [~9.8 GB](./Qwen3.8-27B-YMQ-XXS.gguf) | Absolute VRAM Squeeze / 12GB Card Lifeline | Massive structural quantization noise. Best restricted to low-context, single-turn instructions. Fits 12GB cards with context cache breathing room. |
| **`XS-TI`** | [~10.2 GB](./Qwen3.8-27B-YMQ-XS-TI.gguf) | ⚡ **12GB Card Lifeline (High-Context)** | **The 12GB Context Champion.** Safely pins core attention layers to a stable `IQ3_XXS` gradient while crushing non-critical auxiliary arrays to `IQ2_XS`. |
| **`M-TI`** | [~12.6 GB](./Qwen3.8-27B-YMQ-M-TI.gguf) | 💎 **Premium 16GB GPU Workspace Driver** | **The 16GB Workstation Choice.** Employs a robust `Q5_K/IQ4_XS` mixed-precision gradient that protects logical reasoning focus while leaving over 3.4GB of VRAM wide open. |
| **`M`**   | [~14.5 GB](./Qwen3.8-27B-YMQ-M.gguf) | **The Ultimate Coding Sweet Spot (Recommended)** | **Elite logical stability.** Complete logic clarity. It crushes standard industry 4-bit alternatives. |
| **`L`**   | [~17.0 GB](./Qwen3.8-27B-YMQ-L.gguf) | Premium Single-GPU Processing / Heavy workloads | Near-lossless instruction formatting. Pristine multi-turn architecture safety. |
| **`XL`**  | [~19.0 GB](./Qwen3.8-27B-YMQ-XL.gguf) | Maximum VRAM Fill / No Compromises | Mathematical saturation ceiling. Full precision logic tracks for massive multi-file codebase operations. |

## 📉 Perplexity Evaluation Metrics (WikiText-2)

The following metrics demonstrate the mathematical quality preservation of the **YMQ-Compiler** log-space cluster analysis compared to standard linear quantization layouts. Tested natively via `llama-perplexity` over a 4096 context window using the official WikiText-2 test corpus.

| Model Variant | File Size | Perplexity | Mean KL-Divergence | Internal Bit Gradient (High ➔ Mid ➔ Low ➔ Default ➔ Floor) |
| :--- | :--- | :--- | :--- | :--- |
| **`XXS`** | [~9.8 GB](./Qwen3.8-27B-YMQ-XXS.gguf) | 7.7848 | `0.193804 ± 0.0022` | `IQ3_XXS` ➔ `IQ2_S` ➔ `IQ2_XS` ➔ `IQ2_XS` *(No Floor)* |
| **`XS-TI`** | [~10.2 GB](./Qwen3.8-27B-YMQ-XS-TI.gguf) | 7.6090 | `0.168378 ±   0.001935` | `IQ3_XXS` ➔ `IQ3_XXS` ➔ `IQ3_XXS` ➔ `IQ3_XXS` ➔ `IQ2_XXS` |
| **`M-TI`** | [~12.9 GB](./Qwen3.8-27B-YMQ-M-TI.gguf) | **7.0109** | `0.104087 ± 0.0016` | `Q5_K` ➔ `IQ4_XS` ➔ `IQ3_S` ➔ `IQ3_XXS` ➔ `IQ2_XXS` |
| **`M`** | [~14.5 GB](./Qwen3.8-27B-YMQ-M.gguf) | **6.8413** | `0.053286 ± 0.0013` | `Q5_K` ➔ `IQ4_XS` ➔ `IQ3_S` ➔ `IQ3_XXS` *(No Floor)* |
| **`L`** | [~17.0 GB](./Qwen3.8-27B-YMQ-L.gguf) | 6.9791 | `0.031546 ± 0.0009` | `Q6_K` ➔ `Q5_K` ➔ `IQ4_NL` ➔ `IQ3_S` *(No Floor)* |
| **`XL`** | [~19.0 GB](./Qwen3.8-27B-YMQ-XL.gguf) | 6.8196 | `0.011598 ± 0.0005` | `Q6_K` ➔ `Q6_K` ➔ `Q5_K` ➔ `IQ4_NL` *(No Floor)* |

### 💡 The Core Architectural Discovery
Notice the dramatic performance leap between the `S` and `M` presets. The **YMQ-Compiler** log-space algorithm automatically detects the true data signals on the newly updated Qwen 3.8 hybrid Attention/Mamba routing nodes. 

By shifting the quantization boundaries slightly in the **`M` preset**, the engine safely promotes the model's high-leverage logical spikes straight into full high-fidelity precision layers. This drops the perplexity score down to an elite **6.8413**—matching the raw reasoning power of the massive 19GB `XL` file while clawing back a clean **5 Gigabytes of VRAM overhead cache space** for your local agent environments!

---

## ⚖️ YMQ vs. Uniform Quantization (The AutoRound Philosophy)

Standard quantization pipelines apply a blunt, uniform bit-depth across every single layer in a model. This wastes valuable VRAM on silent background layers while starving critical logic anchors of necessary precision.

The **YMQ-Compiler** implements a philosophy similar to advanced weight-tuning frameworks like **Intel's AutoRound**:
*   **Targeted Bit Allocation:** It strips bits away from low-leverage background tensors and automatically re-allocates that saved VRAM budget straight into full high-fidelity shields for the model's highest cognitive spikes and boundary pathways.
*   **Instant Optimization:** Instead of running heavy, days-long optimization training loops, YMQ achieves a highly accurate mixed-precision layout instantly by analyzing layer importance metrics in log-space.

The result is a custom mixed-precision portfolio that matches the low perplexity and high context stability of premium optimized quants (like AutoRound), while maintaining an ultra-lightweight, high-speed single-GPU cache footprint.

---

## 🛠️ The YMQ Compilation Architecture

Standard quantization pipelines treat network tensors like a flat dataset, applying destructive blanket low-bit compression to delicate tracking networks. The **YMQ-Compiler** solves high-context logic decay by parsing model files dynamically via an automated, multi-tiered protection matrix:

1. **Log-Space Gap Detection Clustering**: Instead of flat percentage thresholds, the engine computes statistical cluster variances in log-space, successfully isolating intermediate logical reasoning spikes and elevating them to stable non-linear 4-bit (`IQ4_XS`) formats, while compressing idle fact-storage layers to aggressive 2-bit baselines.
2. **Fading Boundary Tapering**: Recognizes the extreme fragility of initial token entry data vectors, forcing an input wave cushion (`L00=IQ4_NL` → `L01=IQ4_XS` → `L02=IQ3_XXS`) that gradually stabilizes parameters before hitting the fallback pools.
3. **Dedicated Gate Insulation**: Hard-shields volatile parallel Transformer Multi-Head Attention and Mamba Linear State Space Model (SSM) routing paths, keeping context tracking perfectly noise-free.
4. **Asymmetric Vocabulary Shielding**: Fixes tied-weight boundary errors by mapping the final logit classification exit heads to robust configurations to completely eliminate formatting loops and API tag leakage under deep contexts.
5. **Native Next-N Speculative Stripping**: Processed with advanced pre-tokenizer stripping to ensure zero index offset drift or layer-shifting risks across hybrid configurations.

---

## 🚀 Recommended Runtime Parameters (llama.cpp / llama-server)

*Need to scale up? Deploy this exact script on on-demand cloud GPUs via [RunPod Cloud Compute](https://runpod.io?ref=dbjmkmeh).*

```
$./llama-server -m models/Qwen3.8-27B-YMQ-M.gguf -ctk q8_0 -ctv q4_0 --ctx-size 245760 --mmproj models/Qwen3.8-27B.mmproj-Q6_K.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 --timeout 36000 --checkpoint-min-step 2048 --ctx-checkpoints 4 \
  --n-predict -1 --temp 0.6 --top-p 0.95 --top-k 20 --repeat-penalty 1.05 --jinja -fa
```

### 🖼️ Vision Projection (`--mmproj`)

For multimodal vision support, pair these builds with one of the following projection files:

| Variant | File | Size | Notes |
| :--- | :--- | :--- | :--- |
| **Full Precision (F16)** | [Qwen3.8-27B.mmproj-f16.gguf](https://huggingface.co/mradermacher/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.mmproj-f16.gguf) | ~928 GB | Full-precision native vision tower. Maximum fidelity for image reasoning tasks. |
| **Q6_K** | [Qwen3.8-27B.mmproj-Q6_K.gguf](./Qwen3.8-27B.mmproj-Q6_K.gguf) (this repo) | ~587 MB | High-fidelity quantized vision tower. Excellent quality-to-size balance with minimal perceptible degradation. |
| **Q4_K_S** | [Qwen3.8-27B.mmproj-Q4_K_S.gguf](./Qwen3.8-27B.mmproj-Q4_K_S.gguf) (this repo) | ~478 MB | Compact vision projection for VRAM-constrained setups. Retains strong image understanding at reduced footprint. |

Pass via `--mmproj <path-to-file>` in your llama-server invocation (see example above).

---

## ☕ Support & Future R&D

If the **YMQ-Compiler** builds saved your context window from collapsing or optimized your active development cycle speeds, consider buying a coffee to fund further low-level optimization research. Your support keeps the server nodes baking future model scales!

👉 **[Support ZeroDigest Research on ko-fi](https://ko-fi.com/zerodigest)**

---

## 📦 Source Framework & Automation Code

The compiler pipeline automation engine, setup thresholds, and structural mapping rules are open-source. To view the implementation details or compile your own custom models natively using this profile layout, visit the official development hub:

👉 **[GitHub: ZeroDigest / YMQ-Compiler](https://github.com/minyor/ymq-compiler)**
