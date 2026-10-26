---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - glm-5
  - glm-5.3-flash
  - moe
  - reap
  - expert-pruning
  - nvfp4
  - fp4
  - compressed-tensors
  - blackwell
  - jetson-thor
  - vllm
  - sglang
---

<div align="center">

# GLM-5.3-Flash — REAP-50 (NVFP4 v2 — Domain-Balanced + Calibrated Input Scales)

[![Base Model](https://img.shields.io/badge/Base_Model-zai--org%2FGLM--5.3--Flash-blue)](https://huggingface.co/zai-org/GLM-5.3-Flash)
[![Compression](https://img.shields.io/badge/REAP--50-128%2F256_Experts_Kept_(157B_Total_%2F_12.5B_Active)-orange)](#what-changed-in-v2-vs-v1)
[![Format](https://img.shields.io/badge/Format-NVFP4_compressed--tensors_(94.6_GiB)-76b900)](#tensor-precision-map)
[![Hardware](https://img.shields.io/badge/Target-Blackwell_SM100_%2F_Thor_SM110-brightgreen)](#intended-hardware--serving)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**94.6 GiB NVFP4 (`compressed-tensors`) checkpoint of 50%-pruned `zai-org/GLM-5.3-Flash` (128 of 256 routed experts/layer) with calibrated input activation scales for NVIDIA Blackwell / Jetson AGX Thor.**

| Sibling Repository | Format | Size | Description |
|---|---|---|---|
| **[`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)** | FP8 (`E4M3` + `F8_E8M0`) | 157.0 GiB | Bit-exact master checkpoint (5-pool domain-balanced keep-set + MTP layer 78) |
| **[`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)** *(this repo)* | NVFP4 (`compressed-tensors`) | 94.6 GiB | Blackwell / Thor SM110 `vLLM` / `SGLang` build with calibrated input scales |
| **[`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)** | GGUF (`IQ3_M`–`Q4_K_M` + `Q8_0` attn) | 72.9–94.8 GiB | Single-file GGUF builds for `llama.cpp` (`deepseek2` / `deepseek_v32` MLA + DSA) |
| **[`glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server)** | Pure C++/CUDA (`sm_110a`) | — | Custom Thor CUDA server (20.9 tok/s AR, 28.0–42.4 tok/s lossless MTP speculative decode) |

</div>

---

**GLM-5.3-Flash-REAP50-NVFP4-v2** is the **second-pass (v2) pruned + NVFP4-quantized** release of [Z.ai's `GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), combining **50% REAP expert pruning** (256 → 128 routed experts per MoE layer) with **NVIDIA Blackwell / Thor native NVFP4 (`compressed-tensors` `fp4-pack-quantized`)** quantization on all routed expert weights and **calibrated input activation scales (`input_scale`)** on every expert projection.

At **94.6 GiB on disk**, this checkpoint fits into a **single 128 GB NVIDIA Jetson AGX Thor (`sm_110a`)** or **DGX Spark (`GB10`)** with **>30 GiB of unified-memory headroom** for KV cache, activations, and CUDA workspace.

---

## ✨ Highlights

- **Single-Box 128 GB Blackwell / Thor Fit (`94.6 GiB` Total):** Compresses the 296 GiB 301B FP8 base model by **3.13×** (`296 GiB → 94.6 GiB` across 20 shards) while retaining all **12.5B active parameters per token**.
- **Domain-Balanced 5-Pool REAP-50 Pruning (`v2`):** Upgrades the v1 single-pool keep-set with 5-domain balanced saliency (`general`, `code`, `math`, `zh`, `tool`), union top-16 specialist protection per domain, and a minimum activation frequency floor (`f_j >= 40`).
- **Calibrated `input_scale` on Every Expert Projection (`300 / 300` Projections):** Pre-calibrates per-expert-group `input_scale` (`max(|X|) / (6.0 * 448.0)`) on 100% of routed expert projections across all 75 MoE layers so `vLLM` / `SGLang` / `TensorRT-LLM` CUTLASS FP4×FP4 tensor-core kernels run without dynamic input-max reductions.
- **Unmodified FP8 MLA Attention, DSA Indexer, Shared Experts & Dense Layers:** Keeps all latency-critical and sensitivity-critical dense/attention/shared-expert paths in their exact upstream FP8 (`E4M3` + `F8_E8M0`) or BF16 representations, achieving **`0.3407` top-32 KL divergence** and **`82.93%` top-1 agreement** (`95.84%` top-5 agreement) against the 301B FP8 teacher across 262,144 held-out tokens.

---

## 📋 Model Summary

| Property | Value |
|---|---|
| **Base Model** | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (301B total, 12.5B active, MIT) |
| **Master FP8 Checkpoint** | [`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2) (157.0 GiB) |
| **Architecture** | `GlmMoeDsaForCausalLM` (`deepseek_v32`): 78 layers (3 dense + 75 MoE, layers `0..77`) |
| **Routed Experts** | **128 / 256 kept per layer** (`top-8` active + 1 shared expert), quantized to **NVFP4** (`E2M1` + `F8_E4M3` group-16 scales + `F32` `weight_global_scale` + calibrated `input_scale`) |
| **Dense & Attention Path** | Native upstream FP8 (`F8_E4M3` + `F8_E8M0` 128-block scales) on Dense L0–L2, Shared Experts, and MLA; BF16/FP8 on DSA Indexer |
| **Checkpoint Size** | **94.6 GiB** (20 `safetensors` shards) |
| **Held-Out Evaluation (262K tok)** | `kl_top32 = 0.3407`, `top1_agree = 82.93%`, `top5_agree = 95.84%`, `NLL = 1.5030` (`+0.1228` vs 301B teacher) |
| **Supersedes** | [`patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4) (v1 pass-1 checkpoint) |
| **License** | MIT |

---

<!-- CORRECTIONS:BEGIN -->
## Corrections & Scope Notes (2026-10-24)

1. **Correction to the earlier "Per-Layer Ridge-Regression Router Healing" claim (`+0.00208` ΔNLL).** An earlier revision of this card stated that per-layer ridge regression of the router weights `W_router` improved teacher-forced NLL by `-0.00208` nats across 70/75 MoE layers and was applied to the checkpoint. **That is false:**
   - Every `model.layers.*.mlp.gate.weight` (shape `[128, 5120]`, BF16) and `e_score_correction_bias` in this checkpoint is **100% bit-identical** to the raw unhealed REAP-50 slice of `zai-org/GLM-5.3-Flash` (`max_abs_diff = 0.0` across all 75 MoE layers).
   - When router ridge regression was actually tested in the follow-up (`v3` pipeline), hard top-8 re-selection caused **severe cascading exposure bias** across the 75 MoE layers (`kl_top32` degraded from **0.311 → 2.66**, `top1_agree` dropped from **84.2% → 51.0%**), and was discarded.
2. **Scope of this `NVFP4-v2` repo vs `FP8-v2` and `GGUF` (`v3`):**
   - **78 layers (`0..77`, no layer-78 MTP draft head):** Unlike [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2) and the `v3` GGUFs (`Q4_K_M`, `Q3_K_M`), this NVFP4 checkpoint contains only the 78 main-model layers (`0..77`, `num_nextn_predict_layers=0`).
   - **Unmodified (`1.0x`) layer-77 shared expert:** The `0.7209x` rescale on `layers.77.mlp.shared_experts.down_proj` that was later applied to the `v3` GGUFs (`Q4_K_M` and `Q3_K_M`) was developed *after* `NVFP4-v2` was published and is **not** in this repo (`shared_experts.down_proj` is bit-identical to base `zai-org/GLM-5.3-Flash`).
<!-- CORRECTIONS:END -->

---

<!-- EVAL:BEGIN -->
## Held-Out Quality Evaluation (262,144 Tokens vs 301B FP8 Teacher)

After publishing `v2`, we ran a **strictly held-out 64-prompt × 4,096-token (262,144-token)** teacher-student evaluation across 5 domains (`general`, `code`, `math`, `zh`, `tool`), disjoint from all REAP calibration and `input_scale` calibration sequences, comparing every variant against the unpruned **301B `zai-org/GLM-5.3-Flash` FP8 teacher** (256 experts/layer):

| Model Variant | Size | Teacher Top-32 KL ↓ | Top-1 Agreement ↑ | Top-5 Agreement ↑ | Student NLL ↓ | ΔNLL vs 301B Teacher |
|---|---:|---:|---:|---:|---:|---:|
| `zai-org/GLM-5.3-Flash` (301B FP8 Teacher, 256 exp) | 296.0 GiB | 0.0000 | 100.00% | 100.00% | 1.3802 | 0.0000 |
| [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2) (157B FP8 Master, 128 exp) | 157.0 GiB | 0.3106 | 84.22% | 96.26% | 1.4823 | +0.1021 |
| [`GLM-5.3-Flash-REAP50-GGUF` (`Q4_K_M`, `v3`)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) | 94.8 GiB | 0.3204 | 83.39% | 96.03% | 1.4910 | +0.1108 |
| **`GLM-5.3-Flash-REAP50-NVFP4-v2` (this repo)** | **94.6 GiB** | **0.3407** | **82.93%** | **95.84%** | **1.5030** | **+0.1228** |
| [`GLM-5.3-Flash-REAP50-GGUF` (`Q3_K_M`, `v3`)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) | 77.3 GiB | 0.3367 | 82.70% | 95.81% | 1.5055 | +0.1253 |

### Per-Domain Breakdown (`GLM-5.3-Flash-REAP50-NVFP4-v2` vs `FP8-v2` vs Teacher)

| Domain | Tokens | Teacher NLL | `FP8-v2` NLL | `NVFP4-v2` NLL | `NVFP4-v2` Top-32 KL ↓ | `NVFP4-v2` Top-1 Agree ↑ | `NVFP4-v2` Top-5 Agree ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Math** (`math`) | 49,152 | 0.8920 | 0.9462 | **0.9585** | **0.2148** | **87.65%** | **97.43%** |
| **Code** (`code`) | 57,344 | 1.0772 | 1.1489 | **1.1642** | **0.2512** | **86.08%** | **96.95%** |
| **Chinese** (`zh`) | 49,152 | 1.4820 | 1.5762 | **1.5956** | **0.2519** | **86.05%** | **96.60%** |
| **Tool-Use** (`tool`) | 49,152 | 1.4253 | 1.5594 | **1.5866** | **0.4682** | **79.02%** | **94.57%** |
| **General** (`general`) | 57,344 | 1.9762 | 2.1325 | **2.1622** | **0.4952** | **76.76%** | **93.92%** |
| **All Domains** | **262,144** | **1.3802** | **1.4823** | **1.5030** | **0.3407** | **82.93%** | **95.84%** |

Converting the 128 surviving routed experts per layer from FP8 (157.0 GiB) to NVFP4 (94.6 GiB) saves **62.4 GiB of memory** at a cost of just **`+0.0301` top-32 KL** (`0.3106 → 0.3407`) and **`+0.0207` NLL** (`1.4823 → 1.5030`).
<!-- EVAL:END -->

---

## Model Family & Related Checkpoints

| Checkpoint | Format | Size | Routed Experts | MTP Layer 78 | Notes |
|---|---|---:|---|---|---|
| [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) | FP8 (`E4M3` + `E8M0`) | 296 GiB | 256 / layer (301B total) | Yes | Official upstream base model |
| **[`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)** | FP8 (`E4M3` + `E8M0`) | 157.0 GiB | **128 / layer** (157B total) | **Yes** | **v2 FP8 master** (5-domain REAP + MTP layer 78) |
| **[`patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)** *(this repo)* | **NVFP4 (`compressed-tensors`)** | **94.6 GiB** | **128 / layer** (157B total) | No (78 layers) | **v2 NVFP4** (calibrated `input_scale` on all 300 expert projections) |
| **[`patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)** | GGUF (`IQ3_M`–`Q4_K_M`) | 72.9–94.8 GiB | **128 / layer** (157B total) | Yes (`Q4_K_M`, `Q3_K_M`) | Single-file GGUFs for `llama.cpp` (`Q8_0` attention + shared experts) |
| [`patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4) | NVFP4 | 94.6 GiB | 128 / layer | No | Pass-1 (v1) checkpoint — superseded by `NVFP4-v2` |

---

## What Changed in v2 (vs v1 `GLM-5.3-Flash-REAP50-NVFP4`)

| Feature | v1 (`GLM-5.3-Flash-REAP50-NVFP4`) | **v2 (`GLM-5.3-Flash-REAP50-NVFP4-v2`)** |
|---|---|---|
| **Calibration Corpus** | 320 sequences × 2048 tokens (655K tokens, general text only) | **512 sequences × 4096 tokens (2.10M tokens)** across **5 balanced domains** (General, Code, Math, Chinese/Multilingual, Agentic/Tool-use) + long-range positions up to 4096 |
| **Domain Specialist Protection** | None (single global pool) | **Union of top-16 experts per domain** guaranteed kept in every layer before global ranking |
| **Dead/Rare Tail Handling** | Raw mean `\|f(x)\|_2 * g` (vulnerable to single-token noise) | **Frequency floor (`f_j >= 40`)**: experts below 40 active tokens across 2.1M tokens are ranked by total router mass rather than noisy sample mean |
| **NVFP4 `input_scale` Calibration** | Weight-only (`weight_packed`, `weight_scale`, `weight_global_scale` only) | **Calibrated `input_scale` (`F32` scalar)** included for **all 300 expert projections** (`gate_proj`, `up_proj`, `down_proj` across 75 MoE layers), computed from post-pruning activations |
| **Router Weights (`gate.weight`)** | Unmodified slice of surviving rows | **Unmodified slice of surviving rows** (kept bit-exact to base; see Corrections note above) |

---

## Methodology

### 1. Domain-Balanced REAP-50 Expert Pruning (256 → 128 Routed Experts)
We use **REAP (Router-weighted Expert Activation Pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999))**, which scores each routed expert $j \in \{0, \dots, 255\}$ in layer $l \in \{3, \dots, 77\}$ by the expected L2 norm of its gated contribution to the residual stream over tokens $T_j$ where expert $j$ is selected in the top-8:

$$S_j = \frac{1}{|T_j|} \sum_{t \in T_j} g_{t,j} \, \bigl\| W_{\text{down},j} \bigl( \text{SiLU}(W_{\text{gate},j} x_t) \odot (W_{\text{up},j} x_t) \bigr) \bigr\|_2$$

To prevent domain-specific experts (especially Chinese, formal math, and JSON/tool-call formatting experts) from being pruned when their overall token share is small, v2 records per-domain REAP scores $(S_j^{(d)}, f_j^{(d)})$ across 5 domains $d \in \{\text{general, code, math, zh, tool}\}$:
1. **Per-domain specialist lock:** In every MoE layer, the top-16 experts by $S_j^{(d)}$ (with $f_j^{(d)} \ge 16$) in each of the 5 domains are locked into the keep-set (protecting ~23–54 domain specialists per layer).
2. **Frequency-floored cross-domain ranking:** Remaining slots up to 128 are filled using max-normalized cross-domain REAP score among experts with global activation count $f_j \ge 40$.

### 2. Native NVFP4 Quantization with Calibrated `input_scale`
Starting from the bit-exact v2 FP8 checkpoint (`GLM-5.3-Flash-REAP50-FP8-v2`), all 128 surviving routed experts in layers 3..77 (`mlp.experts.{0..127}.{gate_proj,up_proj,down_proj}`) are converted from fine-grained FP8 (`F8_E4M3` + `F8_E8M0` block-128 scale) to **NVIDIA NVFP4 (`compressed-tensors` `fp4-pack-quantized`)**:
- **`weight_packed` (`U8`, shape `[out, in // 2]`):** Two signed E2M1 FP4 values (`{0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}`) packed per byte (`low nibble = even col`, `high nibble = odd col`).
- **`weight_scale` (`F8_E4M3`, shape `[out, in // 16]`):** Per-16-element group scale factor stored in FP8 E4M3.
- **`weight_global_scale` (`F32`, shape `[1]`):** Tensor-level global scale factor (`max(|W|) / (6.0 * 448.0)`).
- **`input_scale` (`F32`, shape `[1]`):** Pre-calibrated activation scale factor (`max(|X|) / (6.0 * 448.0)`), measured across a 64-sequence × 2048-token multi-domain forward pass on the pruned v2 model. Both `gate_proj` and `up_proj` share a unified per-layer MoE input scale (`max_j input_scale_j`), while each `down_proj` has its own per-expert intermediate activation scale (with a layer-median fallback for any rarely-hit expert).

### Round-Trip Dequantization Formula
To dequantize any NVFP4 expert weight tensor back to BF16/FP32 for inspection or custom kernels:

```python
import torch

# lut maps 4-bit nibble [0..15] to signed E2M1 float
E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

def dequant_nvfp4(weight_packed, weight_scale, weight_global_scale):
    out_dim, half_in = weight_packed.shape
    low = weight_packed & 0x0F
    high = (weight_packed >> 4) & 0x0F
    nibbles = torch.stack([low, high], dim=-1).reshape(out_dim, half_in * 2)
    fp4_vals = E2M1_LUT.to(weight_packed.device)[nibbles.long()]
    # Effective group scale = weight_scale (F8_E4M3) / weight_global_scale (F32)
    scales = (weight_scale.float() / weight_global_scale.float()).repeat_interleave(16, dim=1)
    return (fp4_vals * scales).to(torch.bfloat16)
```

**Measured FP8 → NVFP4 Round-Trip Fidelity (Layer 3 Expert 0):**
- Relative Frobenius error: **6.19%**
- Cosine similarity with original FP8 expert weights: **0.9981**

---

## Tensor Precision Map

| Component | Layers | Precision | Details |
|---|---|---|---|
| **Routed Experts (`mlp.experts.0..127`)** | 3–77 (75 layers) | **NVFP4 (`E2M1` + `F8_E4M3` group-16 + `F32` global + `F32` `input_scale`)** | 128 experts/layer × 3 projections (`gate_proj`, `up_proj`, `down_proj`) |
| **Shared Experts (`mlp.shared_experts`)** | 3–77 (75 layers) | **FP8 (`F8_E4M3` + `F8_E8M0` block-128)** | Unpruned, bit-identical to upstream `GLM-5.3-Flash` |
| **Dense MLP (`mlp`)** | 0–2 (3 layers) | **FP8 (`F8_E4M3` + `F8_E8M0` block-128)** | Unpruned (`intermediate_size = 12288`), bit-identical to upstream |
| **MLA Attention (`self_attn`)** | 0–77 (78 layers) | **FP8 (`F8_E4M3` + `F8_E8M0` block-128)** | `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` kept in native FP8 |
| **DSA Indexer (`self_attn.indexer`)** | 0–77 (78 layers) | **BF16 (`wq_b`, `wk`, `k_norm`) + FP8 (`weights_proj`)** | Dynamic Sparse Attention indexer preserved intact |
| **MoE Router (`mlp.gate`)** | 3–77 (75 layers) | **BF16 (`weight [128, 5120]`) + FP32 (`e_score_correction_bias [128]`)** | Bit-exact slice of surviving 128 experts from upstream `zai-org/GLM-5.3-Flash` |
| **Embeddings, RMSNorms, `lm_head`** | Global | **BF16 / FP32** | Unmodified from upstream |

---

## Architecture Summary ([`config.json`](./config.json))

- **Architecture:** `GlmMoeDsaForCausalLM` (`model_type: "deepseek_v32"`; see [`wiki/10-target-model.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/10-target-model.md))
- **Total Layers:** 78 (Layers 0–2 dense MLP, Layers 3–77 MoE)
- **Hidden Size:** 5,120
- **Attention:** Multi-head Latent Attention (MLA) + Dynamic Sparse Attention (DSA Indexer, `index_topk = 2048`)
  - `num_attention_heads = 64`, `q_lora_rank = 2048`, `kv_lora_rank = 512`, `qk_rope_head_dim = 64`, `qk_nope_head_dim = 192`, `v_head_dim = 256`
- **MoE Configuration:**
  - `n_routed_experts = 128` *(pruned from 256; see [`artifacts/masks/mask.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/masks/mask.json))*
  - `num_experts_per_tok = 8` *(unchanged — same active parameter count per token)*
  - `n_shared_experts = 1`, `moe_intermediate_size = 1536`, `routed_scaling_factor = 2.5`, `scoring_func = "sigmoid"`
- **Context Length:** `202,752` tokens (YaRN `factor = 4.0`)
- **Total Checkpoint Size:** **94.6 GiB** across 62 `safetensors` shards (`model-00001-of-00062.safetensors` .. `model-00062-of-00062.safetensors`, indexed in [`model.safetensors.index.json`](./model.safetensors.index.json))

---

## Intended Hardware & Serving

This checkpoint uses the standard `compressed-tensors` `fp4-pack-quantized` (`NVFP4`) layout with pre-calibrated `input_scale` tensors for:
- **NVIDIA Jetson AGX Thor (`sm_110a`, 128 GB unified LPDDR5X)**
- **NVIDIA DGX Spark (`GB10`, 128 GB unified memory)**
- **NVIDIA RTX PRO 6000 Blackwell (96 GB) / B200 / GB200 (`sm_100a` / `sm_120a`)**

### Example (`vLLM` / `SGLang` with `deepseek_v32` + `compressed-tensors` NVFP4 support)

```bash
vllm serve patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

> **Note for `llama.cpp` users:** If you want to run this model in `llama.cpp` on Jetson AGX Thor, DGX Spark, Mac Studio, or Strix Halo, use the companion GGUF repository [**`patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF`**](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) (`IQ3_M` 72.9 GiB up to `Q4_K_M` 94.8 GiB), or our custom pure-CUDA Thor server [**`patrickbdevaney/glm-5.3-flash-cuda-server`**](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server) (`20.9 tok/s` AR, `28.0–42.4 tok/s` with lossless MTP speculative decoding; see [`docs/ROOFLINE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/ROOFLINE.md)).

---

## 🔎 Provenance & Verification Ledger

Every claim, metric, and tensor layout on this model card can be audited directly against the files in this repository and the companion open-source repositories:

| Claim / Artifact | Primary Verification Source |
|---|---|
| **Upstream Base Model (`301B` FP8, MIT)** | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) · [`./LICENSE`](./LICENSE) |
| **REAP Pruning Formulation** | Lasby et al., 2025 — [*REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*](https://arxiv.org/abs/2510.13999) (`arXiv:2510.13999`) |
| **Model Config, Shard Index & Pass-3 Subtree** | [`./config.json`](./config.json) · [`./model.safetensors.index.json`](./model.safetensors.index.json) · [`./pass3-hope/README.md`](./pass3-hope/README.md) · [`./chat_template.jinja`](./chat_template.jinja) |
| **REAP-50 Pipeline & Integrity Audit** | [`patrickbdevaney/glm-5.3-reap`](https://github.com/patrickbdevaney/glm-5.3-reap) (`main` @ `8540a0a`) · [`EVAL_INTEGRITY.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/EVAL_INTEGRITY.md) · [`wiki/97-end-to-end-audit.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/97-end-to-end-audit.md) |
| **128-Expert Keep-Set & Evaluation Logs** | [`artifacts/masks/mask.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/masks/mask.json) · [`artifacts/mask_eval.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/mask_eval.json) · [`wiki/97-evaluation.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/97-evaluation.md) |
| **NVFP4 Quantization & NIAH Verification** | [`wiki/60-quantization.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/60-quantization.md) · [`results/niah_nvfp4_dsa.json`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/results/niah_nvfp4_dsa.json) · [`include/nvfp4.cuh`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/include/nvfp4.cuh) |
| **Router KD Audit & Healing Analysis** | [`wiki/70-healing.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/70-healing.md) · [`wiki/98-router-kd-global.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/98-router-kd-global.md) · [`artifacts/heal_refit.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/heal_refit.json) |
| **Pure-CUDA Thor Engine (`sm_110a`)** | [`patrickbdevaney/glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server) (`main` @ `399a837`) · [`docs/ROOFLINE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/ROOFLINE.md) · [`docs/SPEC_DECODE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/SPEC_DECODE.md) |

---

## License & Citation

This model is a derivative of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) and is released under the **MIT License** ([`./LICENSE`](./LICENSE)).

```bibtex
@misc{glm53flash_reap50_v2_2026,
  title        = {GLM-5.3-Flash-REAP50-v2: Domain-Balanced 50\% Expert-Pruned GLM-5.3-Flash in FP8, NVFP4, and GGUF},
  author       = {Devaney, Patrick},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2}}
}
```
