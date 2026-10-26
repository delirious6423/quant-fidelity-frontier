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
  - fp8
  - speculative-decoding
  - mtp
  - vllm
  - sglang
---

<div align="center">

# GLM-5.3-Flash — REAP-50 (FP8 v2 Master Checkpoint + MTP Layer 78)

[![Base Model](https://img.shields.io/badge/Base_Model-zai--org%2FGLM--5.3--Flash-blue)](https://huggingface.co/zai-org/GLM-5.3-Flash)
[![Compression](https://img.shields.io/badge/REAP--50-128%2F256_Experts_Kept_(157B_Total_%2F_12.5B_Active)-orange)](#methodology)
[![Format](https://img.shields.io/badge/Format-Bit--Exact_FP8_(157.0_GiB)-green)](#tensor-precision-map)
[![Fidelity](https://img.shields.io/badge/Top--1_Agreement-84.22%25_(KL_0.3106)-brightgreen)](#held-out-quality-evaluation-262144-tokens-vs-301b-fp8-teacher)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Bit-exact 50% expert-pruned master FP8 checkpoint (`157.0 GiB`, 128 of 256 routed experts/layer + MTP Layer 78) of `zai-org/GLM-5.3-Flash` (301B → 157B total, 12.5B active).**

| Sibling Repository | Format | Size | Description |
|---|---|---|---|
| **[`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)** *(this repo)* | FP8 (`E4M3` + `F8_E8M0`) | 157.0 GiB | Bit-exact master checkpoint (5-pool domain-balanced keep-set + MTP layer 78) |
| **[`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)** | NVFP4 (`compressed-tensors`) | 94.6 GiB | Blackwell / Thor SM110 `vLLM` / `SGLang` build with calibrated input scales |
| **[`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)** | GGUF (`IQ3_M`–`Q4_K_M` + `Q8_0` attn) | 72.9–94.8 GiB | Single-file GGUF builds for `llama.cpp` (`deepseek2` / `deepseek_v32` MLA + DSA) |
| **[`glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server)** | Pure C++/CUDA (`sm_110a`) | — | Custom Thor CUDA server (20.9 tok/s AR, 28.0–42.4 tok/s lossless MTP speculative decode) |

</div>

---

**GLM-5.3-Flash-REAP50-FP8-v2** is the **second-pass (v2) bit-exact FP8 master checkpoint** of [Z.ai's `GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), pruned from **256 to 128 routed experts per MoE layer** (301B → **157B total parameters**, **12.5B active parameters per token**) using a **5-domain balanced REAP calibration** while **preserving the upstream Multi-Token Prediction (MTP) speculative decoding head (`model.layers.78`, `num_nextn_predict_layers = 1`)**.

Every surviving weight tensor (`F8_E4M3` weights and `F8_E8M0` block-128 scales) is **copied byte-for-byte without re-quantization** from the upstream `zai-org/GLM-5.3-Flash` checkpoint.

---

## ✨ Highlights

- **Bit-Exact Upstream FP8 Numerics (`157.0 GiB`, Down from `296 GiB`):** All 128 surviving routed experts per MoE layer (`layers 3..77`), all shared experts, all dense MLP layers (`0..2`), and all Multi-head Latent Attention (MLA) + Dynamic Sparse Attention (DSA Indexer) tensors are copied byte-for-byte from `zai-org/GLM-5.3-Flash` with zero re-quantization loss.
- **5-Domain Balanced REAP-50 Pruning (`v2`):** Selected over 2.10M tokens (`512 × 4096`) spanning General English, Source Code, Competition Math, Chinese, and Agentic Tool-Calling, locking the union of the top-16 specialist experts per domain in every layer and enforcing a minimum activation frequency floor (`f_j >= 40`).
- **Preserved Layer-78 MTP Speculative Decoding Head:** Retains the complete upstream Multi-Token Prediction (`model.layers.78`, `num_nextn_predict_layers = 1`) draft block in native FP8/BF16 for native speculative decoding in `vLLM` and `SGLang`.
- **84.22% Top-1 & 96.26% Top-5 Teacher Agreement (`262,144` Held-Out Tokens):** Achieves **`0.3106` top-32 KL divergence** and **`+0.1021` ΔNLL** (`1.4823` vs `1.3802`) against the unpruned 301B FP8 teacher (`88.81%` top-1 on math, `87.31%` on Chinese, `87.28%` on code).

---

## 📋 Model Summary

| Property | Value |
|---|---|
| **Base Model** | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (301B total, 12.5B active, MIT) |
| **Architecture** | `GlmMoeDsaForCausalLM` (`deepseek_v32`): 78 main layers (`0..77`) + 1 MTP draft layer (`78`) |
| **Routed Experts** | **128 / 256 kept per layer** (`layers 3..77`, `top-8` active + 1 shared expert), bit-exact FP8 (`F8_E4M3` + `F8_E8M0`) |
| **MTP Layer (`layers.78`)** | Preserved intact (`num_nextn_predict_layers = 1`, unpruned 256 experts + shared expert + MLA + `eh_proj` + norms) |
| **Checkpoint Size** | **157.0 GiB** across 32 `safetensors` shards (47% smaller than the 296 GiB base model) |
| **Held-Out Evaluation (262K tok)** | `kl_top32 = 0.3106`, `top1_agree = 84.22%`, `top5_agree = 96.26%`, `NLL = 1.4823` (`+0.1021` vs 301B teacher) |
| **Supersedes** | [`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8) (v1 pass-1 checkpoint) |
| **License** | MIT |

---

<!-- CORRECTIONS:BEGIN -->
## Corrections & Scope Notes (2026-10-24)

1. **Correction to the earlier "Per-Layer Ridge-Regression Router Healing" claim (`+0.00208` ΔNLL).** An earlier revision of this card stated that per-layer ridge regression of the router weights `W_router` improved teacher-forced NLL by `-0.00208` nats across 70/75 MoE layers and was applied to this checkpoint. **That is false:**
   - Every `model.layers.*.mlp.gate.weight` (shape `[128, 5120]`, BF16) and `e_score_correction_bias` in this checkpoint is **100% bit-identical** to the raw unhealed REAP-50 slice of `zai-org/GLM-5.3-Flash` (`max_abs_diff = 0.0` across all 75 MoE layers).
   - When router ridge regression was actually tested in the follow-up (`v3` pipeline), hard top-8 re-selection caused **severe cascading exposure bias** across the 75 MoE layers (`kl_top32` degraded from **0.311 → 2.66**, `top1_agree` dropped from **84.2% → 51.0%**), and was discarded.
2. **Scope of this `FP8-v2` repo vs `GGUF` (`v3`):**
   - **Every surviving tensor in this `FP8-v2` checkpoint is 100% bit-identical to upstream `zai-org/GLM-5.3-Flash`** (including `layers.77.mlp.shared_experts.down_proj` at `1.0x`).
   - The `0.7209x` rescale on `layers.77.mlp.shared_experts.down_proj` was developed later for the `v3` GGUFs (`Q4_K_M` and `Q3_K_M` in [`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)) because scaling a fine-grained FP8 tensor with power-of-two `F8_E8M0` block scales by a non-power-of-two factor (`0.7209`) requires re-quantizing the FP8 mantissas, whereas this master repo intentionally preserves exact byte-for-byte upstream FP8 weights.
   - Note that `model.layers.78` (the MTP draft head) in this `FP8-v2` repo retains all **256** upstream routed experts (unpruned), whereas in the `v3` GGUFs (`Q4_K_M`, `Q3_K_M`) layer 78 is pruned to the 128-expert keep-set so it fits the uniform `n_expert = 128` GGUF header.
<!-- CORRECTIONS:END -->

---

<!-- EVAL:BEGIN -->
## Held-Out Quality Evaluation (262,144 Tokens vs 301B FP8 Teacher)

Evaluated on a **strictly held-out 64-prompt × 4,096-token (262,144-token)** multi-domain evaluation suite (disjoint from all REAP calibration data) across 5 domains (`general`, `code`, `math`, `zh`, `tool`), comparing token-by-token logits against the unpruned **301B `zai-org/GLM-5.3-Flash` FP8 teacher** (256 experts/layer):

| Model Variant | Size | Teacher Top-32 KL ↓ | Top-1 Agreement ↑ | Top-5 Agreement ↑ | Student NLL ↓ | ΔNLL vs 301B Teacher |
|---|---:|---:|---:|---:|---:|---:|
| `zai-org/GLM-5.3-Flash` (301B FP8 Teacher, 256 exp) | 296.0 GiB | 0.0000 | 100.00% | 100.00% | 1.3802 | 0.0000 |
| **`GLM-5.3-Flash-REAP50-FP8-v2` (this repo — 157B FP8 Master)** | **157.0 GiB** | **0.3106** | **84.22%** | **96.26%** | **1.4823** | **+0.1021** |
| [`GLM-5.3-Flash-REAP50-GGUF` (`Q4_K_M`, `v3`)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) | 94.8 GiB | 0.3204 | 83.39% | 96.03% | 1.4910 | +0.1108 |
| [`GLM-5.3-Flash-REAP50-GGUF` (`Q4_K_S`, `v2`)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) | 90.0 GiB | 0.3237 | 83.62% | 96.05% | 1.4945 | +0.1143 |
| [`GLM-5.3-Flash-REAP50-GGUF` (`IQ4_XS`, `v2`)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) | 85.6 GiB | 0.3247 | 83.62% | 96.04% | 1.4947 | +0.1145 |
| [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2) | 94.6 GiB | 0.3407 | 82.93% | 95.84% | 1.5030 | +0.1228 |
| [`GLM-5.3-Flash-REAP50-GGUF` (`Q3_K_M`, `v3`)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) | 77.3 GiB | 0.3367 | 82.70% | 95.81% | 1.5055 | +0.1253 |
| [`GLM-5.3-Flash-REAP50-GGUF` (`IQ3_M`, `v2`)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF) | 72.9 GiB | 0.3452 | 82.78% | 95.79% | 1.5112 | +0.1310 |

### Per-Domain Breakdown (`GLM-5.3-Flash-REAP50-FP8-v2` vs 301B FP8 Teacher)

| Domain | Tokens | Teacher NLL | `FP8-v2` Student NLL | ΔNLL | Teacher Top-32 KL ↓ | Top-1 Agreement ↑ | Top-5 Agreement ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Math** (`math`) | 49,152 | 0.8920 | **0.9462** | +0.0542 | **0.1945** | **88.81%** | **97.77%** |
| **Code** (`code`) | 57,344 | 1.0772 | **1.1489** | +0.0717 | **0.2286** | **87.28%** | **97.26%** |
| **Chinese** (`zh`) | 49,152 | 1.4820 | **1.5762** | +0.0942 | **0.2285** | **87.31%** | **96.95%** |
| **Tool-Use** (`tool`) | 49,152 | 1.4253 | **1.5594** | +0.1341 | **0.4299** | **80.49%** | **95.12%** |
| **General** (`general`) | 57,344 | 1.9762 | **2.1325** | +0.1563 | **0.4514** | **78.23%** | **94.48%** |
| **All Domains** | **262,144** | **1.3802** | **1.4823** | **+0.1021** | **0.3106** | **84.22%** | **96.26%** |
<!-- EVAL:END -->

---

## Model Family & Related Checkpoints

| Checkpoint | Format | Size | Routed Experts | MTP Layer 78 | Notes |
|---|---|---:|---|---|---|
| [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) | FP8 (`E4M3` + `E8M0`) | 296 GiB | 256 / layer (301B total) | Yes | Official upstream base model |
| **[`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)** *(this repo)* | **FP8 (`E4M3` + `E8M0`)** | **157.0 GiB** | **128 / layer** (157B total) | **Yes** | **v2 FP8 master** (5-domain REAP + MTP layer 78 preserved) |
| **[`patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)** | NVFP4 (`compressed-tensors`) | 94.6 GiB | **128 / layer** (157B total) | No (78 layers) | v2 NVFP4 (calibrated `input_scale` on all 300 expert projections) |
| **[`patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)** | GGUF (`IQ3_M`–`Q4_K_M`) | 72.9–94.8 GiB | **128 / layer** (157B total) | Yes (`Q4_K_M`, `Q3_K_M`) | Single-file GGUFs for `llama.cpp` (`Q8_0` attention + shared experts) |
| [`patrickbdevaney/GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8) | FP8 (`E4M3` + `E8M0`) | 157.0 GiB | 128 / layer | No | Pass-1 (v1) checkpoint — superseded by `FP8-v2` |

---

## What Changed in v2 (vs v1 `GLM-5.3-Flash-REAP50-FP8`)

| Feature | v1 (`GLM-5.3-Flash-REAP50-FP8`) | **v2 (`GLM-5.3-Flash-REAP50-FP8-v2`)** |
|---|---|---|
| **Calibration Corpus** | 320 sequences × 2048 tokens (655K tokens, general text only) | **512 sequences × 4096 tokens (2.10M tokens)** across **5 balanced domains** (General, Code, Math, Chinese/Multilingual, Agentic/Tool-use) + long-range positions up to 4096 |
| **Domain Specialist Protection** | None (single global pool) | **Union of top-16 experts per domain** guaranteed kept in every layer before global ranking |
| **Dead/Rare Tail Handling** | Raw mean `\|f(x)\|_2 * g` (vulnerable to single-token noise) | **Frequency floor (`f_j >= 40`)**: experts below 40 active tokens across 2.1M tokens are ranked by total router mass rather than noisy sample mean |
| **MTP Speculative Decoding Layer (`model.layers.78`)** | Stripped (`num_nextn_predict_layers = 0`) | **Preserved intact (`model.layers.78`, `num_nextn_predict_layers = 1`)** in native FP8/BF16 for speculative decoding in `vLLM` / `SGLang` |
| **Router Weights (`gate.weight`)** | Unmodified slice of surviving rows | **Unmodified slice of surviving rows** (bit-exact to base; see Corrections note above) |

---

## Methodology

### 1. Domain-Balanced REAP-50 Expert Pruning (256 → 128 Routed Experts)
We use **REAP (Router-weighted Expert Activation Pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999))**, which scores each routed expert $j \in \{0, \dots, 255\}$ in layer $l \in \{3, \dots, 77\}$ by the expected L2 norm of its gated contribution to the residual stream over tokens $T_j$ where expert $j$ is selected in the top-8:

$$S_j = \frac{1}{|T_j|} \sum_{t \in T_j} g_{t,j} \, \bigl\| W_{\text{down},j} \bigl( \text{SiLU}(W_{\text{gate},j} x_t) \odot (W_{\text{up},j} x_t) \bigr) \bigr\|_2$$

To prevent domain-specific experts (especially Chinese, formal math, and JSON/tool-call formatting experts) from being pruned when their overall token share is small, v2 records per-domain REAP scores $(S_j^{(d)}, f_j^{(d)})$ across 5 domains $d \in \{\text{general, code, math, zh, tool}\}$:
1. **Per-domain specialist lock:** In every MoE layer, the top-16 experts by $S_j^{(d)}$ (with $f_j^{(d)} \ge 16$) in each of the 5 domains are locked into the keep-set (protecting ~23–54 domain specialists per layer).
2. **Frequency-floored cross-domain ranking:** Remaining slots up to 128 are filled using max-normalized cross-domain REAP score among experts with global activation count $f_j \ge 40$.

### 2. Multi-Token Prediction (MTP) Speculative Decoding Layer (`model.layers.78`)
Unlike v1 (which stripped layer 78), `GLM-5.3-Flash-REAP50-FP8-v2` preserves the complete upstream **MTP layer (`model.layers.78`, `num_nextn_predict_layers = 1`)** including:
- `eh_proj`, `enorm`, `hnorm`, `shared_head.norm`
- Full MLA + DSA Indexer attention (`layers.78.self_attn.*`)
- Full MoE block (`layers.78.mlp.*`)

This enables native **MTP speculative decoding** in `vLLM` (`--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'`) and `SGLang` (`--speculative-algorithm EAGLE/NEXTN`).

---

## Tensor Precision Map

| Component | Layers | Precision | Details |
|---|---|---|---|
| **Routed Experts (`mlp.experts.0..127`)** | 3–77 (75 layers) | **FP8 (`F8_E4M3` + `F8_E8M0` block-128)** | 128 experts/layer × 3 projections (`gate_proj`, `up_proj`, `down_proj`), copied byte-for-byte from upstream |
| **Shared Experts (`mlp.shared_experts`)** | 3–77 (75 layers) | **FP8 (`F8_E4M3` + `F8_E8M0` block-128)** | Unpruned, bit-identical to upstream `GLM-5.3-Flash` |
| **Dense MLP (`mlp`)** | 0–2 (3 layers) | **FP8 (`F8_E4M3` + `F8_E8M0` block-128)** | Unpruned (`intermediate_size = 12288`), bit-identical to upstream |
| **MLA Attention (`self_attn`)** | 0–78 (79 layers) | **FP8 (`F8_E4M3` + `F8_E8M0` block-128)** | `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` kept in native FP8 |
| **DSA Indexer (`self_attn.indexer`)** | 0–78 (79 layers) | **BF16 (`wq_b`, `wk`, `k_norm`) + FP8 (`weights_proj`)** | Dynamic Sparse Attention indexer preserved intact |
| **MoE Router (`mlp.gate`)** | 3–77 (75 layers) | **BF16 (`weight [128, 5120]`) + FP32 (`e_score_correction_bias [128]`)** | Bit-exact slice of surviving 128 experts from upstream `zai-org/GLM-5.3-Flash` |
| **MTP Layer (`model.layers.78`)** | 78 (1 layer) | **FP8 + BF16** | Full upstream Multi-Token Prediction block preserved |
| **Embeddings, RMSNorms, `lm_head`** | Global | **BF16 / FP32** | Unmodified from upstream |

---

## Architecture Summary ([`config.json`](./config.json))

- **Architecture:** `GlmMoeDsaForCausalLM` (`model_type: "deepseek_v32"`; see [`wiki/10-target-model.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/10-target-model.md))
- **Total Layers:** 78 main layers (Layers 0–2 dense MLP, Layers 3–77 MoE) + 1 MTP layer (Layer 78, `num_nextn_predict_layers = 1`)
- **Hidden Size:** 5,120
- **Attention:** Multi-head Latent Attention (MLA) + Dynamic Sparse Attention (DSA Indexer, `index_topk = 2048`)
  - `num_attention_heads = 64`, `q_lora_rank = 2048`, `kv_lora_rank = 512`, `qk_rope_head_dim = 64`, `qk_nope_head_dim = 192`, `v_head_dim = 256`
- **MoE Configuration:**
  - `n_routed_experts = 128` *(pruned from 256; see [`./reap_metadata.json`](./reap_metadata.json) and [`artifacts/masks/mask.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/masks/mask.json))*
  - `num_experts_per_tok = 8` *(unchanged — same active parameter count per token)*
  - `n_shared_experts = 1`, `moe_intermediate_size = 1536`, `routed_scaling_factor = 2.5`, `scoring_func = "sigmoid"`
- **Context Length:** `202,752` tokens (YaRN `factor = 4.0`)
- **Total Checkpoint Size:** **157.0 GiB** across 62 `safetensors` shards (`model-00001-of-00062.safetensors` .. `model-00062-of-00062.safetensors`, indexed in [`./model.safetensors.index.json`](./model.safetensors.index.json))

---

## Serving Examples

### `vLLM` (with MTP Speculative Decoding on 2× H100 / 2× B200 / 2× RTX PRO 6000)

```bash
vllm serve patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2 \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --max-model-len 65536 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

### Single-Node 128 GB Deployment (Jetson AGX Thor / DGX Spark)
For single-board 128 GB deployments, use one of the derived 4-bit checkpoints built directly from this v2 master:
- **NVFP4 (`94.6 GiB`):** [`patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)
- **GGUF (`72.9–94.8 GiB`):** [`patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)
- **Custom Pure-CUDA Thor Engine (`20.9 tok/s` AR, `28.0–42.4 tok/s` MTP):** [`patrickbdevaney/glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server)

---

## 🔎 Provenance & Verification Ledger

Every claim, metric, and tensor layout on this model card can be audited directly against the files in this repository and the companion open-source repositories:

| Claim / Artifact | Primary Verification Source |
|---|---|
| **Upstream Base Model (`301B` FP8, MIT)** | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) · [`./LICENSE`](./LICENSE) |
| **REAP Pruning Formulation** | Lasby et al., 2025 — [*REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*](https://arxiv.org/abs/2510.13999) (`arXiv:2510.13999`) |
| **In-Repo REAP Metadata, First-Moment Gains & Pass-3 Subtree** | [`./reap_metadata.json`](./reap_metadata.json) · [`./adapters/first_moment_gains.json`](./adapters/first_moment_gains.json) · [`./pass3-hope/README.md`](./pass3-hope/README.md) · [`./pass3-hope/reap_metadata.json`](./pass3-hope/reap_metadata.json) |
| **Model Config & 62-Shard Safetensors Index** | [`./config.json`](./config.json) · [`./model.safetensors.index.json`](./model.safetensors.index.json) · [`./chat_template.jinja`](./chat_template.jinja) |
| **REAP-50 Pipeline & Integrity Audit** | [`patrickbdevaney/glm-5.3-reap`](https://github.com/patrickbdevaney/glm-5.3-reap) (`main` @ `8540a0a`) · [`EVAL_INTEGRITY.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/EVAL_INTEGRITY.md) · [`wiki/97-end-to-end-audit.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/97-end-to-end-audit.md) |
| **128-Expert Keep-Set & Evaluation Logs** | [`artifacts/masks/mask.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/masks/mask.json) · [`artifacts/mask_eval.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/mask_eval.json) · [`wiki/97-evaluation.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/97-evaluation.md) |
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
  howpublished = {\url{https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2}}
}
```
