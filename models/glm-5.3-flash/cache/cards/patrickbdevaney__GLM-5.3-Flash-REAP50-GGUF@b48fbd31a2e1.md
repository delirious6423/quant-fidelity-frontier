---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - gguf
  - llama-cpp
  - glm-5.3
  - glm-5.3-flash
  - deepseek_v32
  - moe
  - reap
  - expert-pruning
  - imatrix
  - speculative-decoding
  - mtp
  - edge-ai
  - jetson-thor
---

<div align="center">

# GLM-5.3-Flash — REAP-50 (GGUF `IQ3_M` / `Q3_K_M` / `IQ4_XS` / `Q4_K_S` / `Q4_K_M`)

[![Base Model](https://img.shields.io/badge/Base_Model-zai--org%2FGLM--5.3--Flash-blue)](https://huggingface.co/zai-org/GLM-5.3-Flash)
[![Compression](https://img.shields.io/badge/REAP--50-128%2F256_Experts_Kept_(157B_Total_%2F_12.5B_Active)-orange)](#how-this-model-was-built)
[![Format](https://img.shields.io/badge/Format-GGUF_(72.9--94.8_GiB)-green)](#files-in-this-repo)
[![Hardware](https://img.shields.io/badge/Tested_On-128GB_Jetson_AGX_Thor-76b900)](#performance-on-jetson-agx-thor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**50% expert-pruned `zai-org/GLM-5.3-Flash` (301B → 157B total, 12.5B active) in GGUF (`IQ3_M` to `Q4_K_M`, 72.9–94.8 GiB) with preserved `Q8_0` MLA + Indexer attention.**

| Sibling Repository | Format | Size | Description |
|---|---|---|---|
| **[`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)** | FP8 (`E4M3` + `F8_E8M0`) | 157.0 GiB | Bit-exact master checkpoint (5-pool domain-balanced keep-set + MTP layer 78) |
| **[`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)** | NVFP4 (`compressed-tensors`) | 94.6 GiB | Blackwell / Thor SM110 `vLLM` / `SGLang` build with calibrated input scales |
| **[`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)** *(this repo)* | GGUF (`IQ3_M`–`Q4_K_M` + `Q8_0` attn) | 72.9–94.8 GiB | Single-file GGUF builds for `llama.cpp` (`deepseek2` / `deepseek_v32` MLA + DSA) |
| **[`glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server)** | Pure C++/CUDA (`sm_110a`) | — | Custom Thor CUDA server (20.9 tok/s AR, 28.0–42.4 tok/s lossless MTP speculative decode) |

</div>

---

**GLM-5.3-Flash-REAP50-GGUF** provides single-file GGUF quantizations of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) compressed with **REAP-50** (Router-weighted Expert Activation Pruning, retaining **128 of 256 routed experts per MoE layer**). Total parameters drop from **301B to 157B** while active parameters per token remain **12.5B** (top-8 of 128 routed experts + 1 shared expert + Multi-head Latent Attention + Dynamic Sparse Attention Indexer), fitting comfortably inside a single **128 GB unified-memory system** (NVIDIA Jetson AGX Thor, DGX Spark, Mac Studio, or Strix Halo) with up to 202K context.

---

## ✨ Highlights

- **50% Routed-Expert Compression (301B → 157B Total, 12.5B Active):** Retains 128 of 256 routed experts across all 75 MoE layers (`layers 3..77`) using a 5-pool domain-balanced REAP saliency calibration (`general`, `code`, `math`, `Chinese`, `tool-use`) with union top-16 specialist protection and frequency-floor guards (`f_j >= 40`).
- **Two Weight Vintages Disclosed Transparently (`v2` vs `v3`):**
  - **`Q4_K_M` (94.8 GiB) & `Q3_K_M` (77.3 GiB) are `v3`** (uploaded 2026-10-20): include the **layer-78 MTP draft head** (`nextn.*` tensors) and the **layer-77 shared-expert output-scale correction (`0.7209x`)** that removes high-entropy tail collapse (`kl_full` reduced from `0.506` to `0.388`).
  - **`Q4_K_S` (90.0 GiB), `IQ4_XS` (85.6 GiB), & `IQ3_M` (72.9 GiB) are `v2`** (uploaded 2026-10-16): 78-layer (`0..77`) builds without the layer-77 shared-expert scale or layer-78 MTP tensors.
- **Uncompromised `Q8_0` Attention & DSA Indexer Path:** Every quantization preserves **all MLA (`wq_a`, `wq_b`, `wkv_a_mqa`, `wkv_b`, `wo`), DSA Indexer (`indexer.wq_b`, `indexer.wk`, `indexer.k_norm`, `indexer.weights_proj`), shared experts (`ffn_*_shexp`), and dense layers (`0..2`) at `Q8_0` or `F32`**, quantizing only the routed expert banks (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`).
- **Paired Held-Out Teacher Evaluation (`262,144` Tokens across 64 Prompts × 4,096 Context):** Evaluates directly against the unpruned 301B `zai-org/GLM-5.3-Flash` teacher: `Q4_K_M` achieves **`0.3204` top-32 KL divergence** and **`83.39%` top-1 argmax agreement** (`88.2%` on math, `86.7%` on Chinese, `86.5%` on code).

---

## 📋 Model Summary

| Property | Value |
|---|---|
| **Base Model** | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) (301B total, 12.5B active, MIT) |
| **Architecture** | `deepseek_v32` (`deepseek2` in GGUF): 78 layers (3 dense + 75 MoE) + Layer 78 MTP draft block (in `v3` files) |
| **MoE Configuration** | 128 routed experts/layer (pruned from 256), top-8 active (`n_group=1`, sigmoid gating) + 1 shared expert |
| **Attention & Indexer** | Multi-head Latent Attention (`kv_lora_rank=512`, `q_lora_rank=2048`) + 32-head FP8/Q8_0 DSA Indexer (`top-2048` tokens) |
| **Context Length** | **202,752 tokens** (YaRN `factor=4.0` over 50,688 native window) |
| **Vocabulary & EOG Tokens** | 154,880 (`<\|endoftext\|>` = `154820`, `<\|user\|>` = `154827`, `<\|observation\|>` = `154829`) |
| **Quantization Variants** | `Q4_K_M` (94.8 GiB, `v3`), `Q4_K_S` (90.0 GiB, `v2`), `IQ4_XS` (85.6 GiB, `v2`), `Q3_K_M` (77.3 GiB, `v3`), `IQ3_M` (72.9 GiB, `v2`) |
| **Recommended EOG Flag** | `--override-kv deepseek2.ggml.eos_token_ids=arr:i32:154820,154827,154829` (for `v2` files; `v3` sets `eos_token_id = 154827`) |
| **License** | MIT |

---

<!-- CORRECTIONS:BEGIN -->
## Corrections (2026-10-24)

1. **Two weight vintages live in this repo (`v2` vs `v3`).** Look at the upload timestamps in the file browser:
   - **`Q4_K_M` and `Q3_K_M` were re-uploaded on 2026-10-20 (`v3`).** They are **79-layer** GGUFs (`layers 0..78`, i.e. 78 main-model layers + 1 MTP draft layer `blk.78.nextn.*`, **1,089 tensors**, `eos_token_id = 154827`). More importantly, they include a **post-REAP healing pass on layer 77**: `blk.77.ffn_down_shexp.weight` is scaled by **`0.7209x`** (derived by least-squares fit of the final norm input to the unpruned teacher on calibration data; raw cosine similarity vs `v2` is `0.9999999`, ratio of L2 norms is `0.720892`). That single-tensor rescale fixes a rare high-entropy tail collapse in `v2` where the student occasionally emitted a token at teacher probability `<1e-6`, cutting full-vocab KL on the 131K calibration eval from **0.506 → 0.388** while leaving `kl_top32` (`0.320`) and `top1_agree` (`83.4%`) essentially unchanged.
   - **`Q4_K_S`, `IQ4_XS`, and `IQ3_M` were uploaded on 2026-10-16 (`v2`).** They are **78-layer** GGUFs (`layers 0..77`, **1,071 tensors**, no `nextn` MTP tensors, `eos_token_id = 154820`) with the unscaled layer-77 shared expert (`1.0x`).
   - **[`imatrix-GLM-5.3-Flash-REAP50.dat`](./imatrix-GLM-5.3-Flash-REAP50.dat) was updated on 2026-10-19** (`905 entries`, including `blk.78.*`).
2. **Correction to the earlier "Per-Layer Ridge-Regression Router Healing" section (`+0.00208` ΔNLL).** The card previously claimed that per-layer ridge regression of the router weights `W_router` on 1,024 calibration sequences improved teacher-forced NLL by `-0.00208` nats across 70/75 MoE layers and was baked into the weights. **Neither is true:**
   - Every `blk.*.ffn_gate_inp.weight` (shape `[128, 5120]`) and `exp_probs_b.bias` in **both** the `v2` and `v3` GGUFs is **100% bit-identical** to the raw unhealed REAP-50 slice of `zai-org/GLM-5.3-Flash`.
   - When the router ridge-regression pass was actually tested in the follow-up (`v3` pipeline; [`wiki/98-router-kd-global.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/98-router-kd-global.md)), hard top-8 re-selection caused **severe cascading exposure bias** across the 75 MoE layers (`kl_top32` degraded from **0.311 → 2.66**, `top1_agree` dropped from **84.2% → 51.0%**), and was discarded.
   - The only healing present in any published weight is the **layer-77 shared-expert output scale (`0.7209x`)** in the `v3` files (`Q4_K_M` and `Q3_K_M`; [`artifacts/heal_refit.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/heal_refit.json)).
<!-- CORRECTIONS:END -->

---

## Versions — which weights are which

Because two vintages of GGUF files share this repository (`v2` from 2026-10-16 and `v3` from 2026-10-20), here is the exact inventory across all published `GLM-5.3-Flash-REAP50` repos so you know what is in each file before downloading:

| Repo / File | Vintage | Uploaded | Layers / Tensors | Router `gate.weight` (`[128, 5120]`) | Layer-77 Shared Expert (`down_proj`) | Layer-78 MTP (`nextn.*`) | Stop Token (`eos_token_id`) |
|---|---|---|---|---|---|---|---|
| **[`GLM-5.3-Flash-REAP50-Q4_K_M.gguf`](./GLM-5.3-Flash-REAP50-Q4_K_M.gguf)** | **`v3`** | 2026-10-20 | 79 layers (`0..78`), 1,089 tensors | Bit-exact slice of base (`v2` keep-set) | **Scaled `0.7209x`** (post-REAP tail fix) | **Included** (`blk.78.nextn.*`) | `154827` (`<\|user\|>`) |
| **[`GLM-5.3-Flash-REAP50-Q3_K_M.gguf`](./GLM-5.3-Flash-REAP50-Q3_K_M.gguf)** | **`v3`** | 2026-10-20 | 79 layers (`0..78`), 1,089 tensors | Bit-exact slice of base (`v2` keep-set) | **Scaled `0.7209x`** (post-REAP tail fix) | **Included** (`blk.78.nextn.*`) | `154827` (`<\|user\|>`) |
| **[`GLM-5.3-Flash-REAP50-Q4_K_S.gguf`](./GLM-5.3-Flash-REAP50-Q4_K_S.gguf)** | `v2` | 2026-10-16 | 78 layers (`0..77`), 1,071 tensors | Bit-exact slice of base (`v2` keep-set) | Unmodified (`1.0x` base) | Not included | `154820` (`<\|endoftext\|>`) |
| **[`GLM-5.3-Flash-REAP50-IQ4_XS.gguf`](./GLM-5.3-Flash-REAP50-IQ4_XS.gguf)** | `v2` | 2026-10-16 | 78 layers (`0..77`), 1,071 tensors | Bit-exact slice of base (`v2` keep-set) | Unmodified (`1.0x` base) | Not included | `154820` (`<\|endoftext\|>`) |
| **[`GLM-5.3-Flash-REAP50-IQ3_M.gguf`](./GLM-5.3-Flash-REAP50-IQ3_M.gguf)** | `v2` | 2026-10-16 | 78 layers (`0..77`), 1,071 tensors | Bit-exact slice of base (`v2` keep-set) | Unmodified (`1.0x` base) | Not included | `154820` (`<\|endoftext\|>`) |
| [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2) | `v2` | 2026-10-15 | 79 layers (`0..78`), 62 shards | Bit-exact slice of base (`v2` keep-set) | Unmodified (`1.0x` base) | **Included** (unpruned 256-expert layer 78) | `[154820, 154827, 154829]` |
| [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2) | `v2` | 2026-10-16 | 78 layers (`0..77`), 62 shards | Bit-exact slice of base (`v2` keep-set) | Unmodified (`1.0x` base) | Not included | `[154820, 154827, 154829]` |
| [`GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8) *(v1, superseded)* | `v1` | 2026-10-15 | 78 layers (`0..77`), 62 shards | Bit-exact slice of base (`v1` keep-set) | Unmodified (`1.0x` base) | Not included | `[154820, 154827, 154829]` |
| [`GLM-5.3-Flash-REAP50-NVFP4`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4) *(v1, superseded)* | `v1` | 2026-10-15 | 78 layers (`0..77`), 62 shards | Bit-exact slice of base (`v1` keep-set) | Unmodified (`1.0x` base) | Not included | `[154820, 154827, 154829]` |

**Summary of the three passes (`v1` → `v2` → `v3`):**
- **`v1` (`-FP8`, `-NVFP4`):** Initial single-pool REAP-50 pass (320 general calibration sequences, no per-domain specialist protection, no frequency floor). Kept for archival reproducibility; use `v2` or `v3` instead.
- **`v2` (`-FP8-v2`, `-NVFP4-v2`, and GGUF `Q4_K_S` / `IQ4_XS` / `IQ3_M`):** 5-pool domain-balanced REAP-50 keep-set (general, code, math, Chinese, tool-use; union top-16 specialist protection per domain + frequency floor `f_j >= 40`). All surviving tensors are bit-exact slices of `zai-org/GLM-5.3-Flash`.
- **`v3` (GGUF `Q4_K_M` and `Q3_K_M`):** Same `v2` keep-set, plus:
  1. **Layer-77 shared-expert output rescale (`0.7209x`)** on `blk.77.ffn_down_shexp.weight` (reduces full-vocab KL on the 131K calibration eval from `0.506 → 0.388` by suppressing rare `<1e-6` tail outliers).
  2. **Layer-78 MTP draft head (`blk.78.nextn.*`)** included in the GGUF (pruned to the `v2` 128-expert keep-set so it fits the uniform `n_expert=128` GGUF schema).
  3. **Stop-token header fix:** `deepseek2.ggml.eos_token_id` set to `154827` (`<|user|>`), the primary chat turn terminator in GLM-5.3-Flash's chat template.

**Which file should I pick?**
- If you have **~100+ GiB of unified RAM / VRAM** (128 GB Jetson AGX Thor, DGX Spark, Mac Studio, Strix Halo): **`GLM-5.3-Flash-REAP50-Q4_K_M.gguf` (`v3`, 94.8 GiB)** is the flagship build — it has the `v3` layer-77 shared-expert scale, the MTP layer, and the highest quality (`kl_top32 = 0.3204`, `top1_agree = 83.39%`).
- If you want the **smallest footprint with the `v3` layer-77 fix** (or room for very large KV caches on 96–128 GB machines): **`GLM-5.3-Flash-REAP50-Q3_K_M.gguf` (`v3`, 77.3 GiB)**.
- If you specifically need an intermediate size between 77 and 95 GiB: **`Q4_K_S` (90.0 GiB)** or **`IQ4_XS` (85.6 GiB)** (`v2` vintage; pass `--override-kv deepseek2.ggml.eos_token_ids=arr:i32:154820,154827,154829` for chat stop tokens).

---

<!-- EVAL:BEGIN -->
## Quality Evaluation vs Unpruned 301B Teacher

Evaluated on a **strictly held-out 64-prompt × 4,096-token (262,144-token)** multi-domain evaluation suite (disjoint from all REAP calibration and imatrix data) across 5 domains (`general`, `code`, `math`, `zh`, `tool`). Logits from the unpruned **301B `zai-org/GLM-5.3-Flash` FP8 teacher** (256 experts/layer, running natively in PyTorch with exact FP8 block dequantization and full MLA + Indexer attention) were compared token-by-token at all 262,144 positions against each model variant:

| Model Variant | Size | Teacher Top-32 KL ↓ | Top-1 Agreement ↑ | Top-5 Agreement ↑ | Student NLL ↓ | ΔNLL vs 301B Teacher |
|---|---:|---:|---:|---:|---:|---:|
| `zai-org/GLM-5.3-Flash` (301B FP8 Teacher, 256 exp) | 296.0 GiB | 0.0000 | 100.00% | 100.00% | 1.3802 | 0.0000 |
| **[`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)** (157B FP8 Master, 128 exp) | 157.0 GiB | **0.3106** | **84.22%** | **96.26%** | **1.4823** | **+0.1021** |
| **`GLM-5.3-Flash-REAP50-Q4_K_M.gguf`** (`v3` — 79L + L77 `0.7209x`) | **94.8 GiB** | **0.3204** | **83.39%** | **96.03%** | **1.4910** | **+0.1108** |
| **`GLM-5.3-Flash-REAP50-Q4_K_S.gguf`** (`v2` — 78L) | **90.0 GiB** | **0.3237** | **83.62%** | **96.05%** | **1.4945** | **+0.1143** |
| **`GLM-5.3-Flash-REAP50-IQ4_XS.gguf`** (`v2` — 78L, imatrix) | **85.6 GiB** | **0.3247** | **83.62%** | **96.04%** | **1.4947** | **+0.1145** |
| [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2) (157B NVFP4, 128 exp) | 94.6 GiB | 0.3407 | 82.93% | 95.84% | 1.5030 | +0.1228 |
| **`GLM-5.3-Flash-REAP50-Q3_K_M.gguf`** (`v3` — 79L + L77 `0.7209x`, imatrix) | **77.3 GiB** | **0.3367** | **82.70%** | **95.81%** | **1.5055** | **+0.1253** |
| **`GLM-5.3-Flash-REAP50-IQ3_M.gguf`** (`v2` — 78L, imatrix) | **72.9 GiB** | **0.3452** | **82.78%** | **95.79%** | **1.5112** | **+0.1310** |

### Per-Domain Breakdown (262,144 Held-Out Tokens)

| Domain | Tokens | Teacher NLL | `FP8-v2` KL / Top-1 | `Q4_K_M` (`v3`) KL / Top-1 | `Q4_K_S` (`v2`) KL / Top-1 | `IQ4_XS` (`v2`) KL / Top-1 | `Q3_K_M` (`v3`) KL / Top-1 | `IQ3_M` (`v2`) KL / Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Math** (`math`) | 49,152 | 0.8920 | 0.1945 / 88.8% | **0.1972 / 88.2%** | 0.2038 / 88.4% | 0.2054 / 88.3% | 0.2073 / 87.7% | 0.2188 / 87.8% |
| **Code** (`code`) | 57,344 | 1.0772 | 0.2286 / 87.3% | **0.2367 / 86.5%** | 0.2389 / 86.7% | 0.2389 / 86.7% | 0.2498 / 85.9% | 0.2547 / 86.0% |
| **Chinese** (`zh`) | 49,152 | 1.4820 | 0.2285 / 87.3% | **0.2371 / 86.7%** | 0.2384 / 86.8% | 0.2392 / 86.8% | 0.2512 / 86.1% | 0.2564 / 86.1% |
| **Tool-Use** (`tool`) | 49,152 | 1.4253 | 0.4299 / 80.5% | **0.4408 / 79.4%** | 0.4465 / 79.9% | 0.4480 / 79.8% | 0.4614 / 78.7% | 0.4744 / 78.9% |
| **General** (`general`) | 57,344 | 1.9762 | 0.4514 / 78.2% | **0.4671 / 77.2%** | 0.4706 / 77.4% | 0.4722 / 77.4% | 0.4896 / 76.4% | 0.5008 / 76.4% |
| **All Domains** | **262,144** | **1.3802** | **0.3106 / 84.22%** | **0.3204 / 83.39%** | **0.3237 / 83.62%** | **0.3247 / 83.62%** | **0.3367 / 82.70%** | **0.3452 / 82.78%** |

**Key takeaways from the held-out evaluation:**
1. **Quantization adds only `0.010–0.035` KL on top of REAP-50 pruning:** Because all attention projections, MLA latent bottleneck matrices, Indexer projections, and shared experts are kept at `Q8_0`, quantizing the 128 routed experts from FP8 (157.0 GiB) to `Q4_K_M` (94.8 GiB) increases KL by just **`+0.0098` nats** (`0.3106 → 0.3204`).
2. **`IQ4_XS` (85.6 GiB) ties `Q4_K_S` (90.0 GiB) while saving 4.4 GiB:** With the 512K-token importance matrix, `IQ4_XS` matches `Q4_K_S` to four decimal places in NLL (`1.4947` vs `1.4945`) and matches its top-1 agreement (`83.62%`).
3. **`Q4_K_M`, `Q4_K_S`, `IQ4_XS`, and `Q3_K_M` all beat `NVFP4-v2` on KL divergence:** Preserving `Q8_0` on the attention/shared-expert path gives the GGUF builds a slight accuracy edge over uniform NVFP4 (`0.3204–0.3367` vs `0.3407` KL).
<!-- EVAL:END -->

---

## Files in This Repo

| File | Vintage | Size (GiB) | Routed Experts (`ffn_*_exps`) | Attention / Indexer / Shared Exp / Dense L0-L2 | Notes |
|---|---|---:|---|---|---|
| [`GLM-5.3-Flash-REAP50-Q4_K_M.gguf`](./GLM-5.3-Flash-REAP50-Q4_K_M.gguf) | **`v3`** | **94.8 GiB** | `Q4_K` (`gate`/`up`), `Q4_K`/`Q6_K` (`down`) | `Q8_0` (norms in `F32`) | **Recommended flagship (96–128 GB RAM/VRAM)** — includes L77 `0.7209x` scale + L78 MTP |
| [`GLM-5.3-Flash-REAP50-Q4_K_S.gguf`](./GLM-5.3-Flash-REAP50-Q4_K_S.gguf) | `v2` | **90.0 GiB** | `Q4_K` (`gate`/`up`/`down`) | `Q8_0` (norms in `F32`) | Tight 4-bit K-quant (`v2` 78-layer build) |
| [`GLM-5.3-Flash-REAP50-IQ4_XS.gguf`](./GLM-5.3-Flash-REAP50-IQ4_XS.gguf) | `v2` | **85.6 GiB** | `IQ4_XS` (`gate`/`up`), `IQ4_XS`/`Q4_K` (`down`) | `Q8_0` (norms in `F32`) | **Best ~85 GiB sweet spot** (imatrix-guided non-linear 4-bit, `v2` 78-layer build) |
| [`GLM-5.3-Flash-REAP50-Q3_K_M.gguf`](./GLM-5.3-Flash-REAP50-Q3_K_M.gguf) | **`v3`** | **77.3 GiB** | `Q3_K` (`gate`/`up`), `Q3_K`/`Q4_K` (`down`) | `Q8_0` (norms in `F32`) | **Recommended 3-bit (`v3`)** — includes L77 `0.7209x` scale + L78 MTP + imatrix |
| [`GLM-5.3-Flash-REAP50-IQ3_M.gguf`](./GLM-5.3-Flash-REAP50-IQ3_M.gguf) | `v2` | **72.9 GiB** | `IQ3_M` (`gate`/`up`), `IQ3_M`/`Q3_K` (`down`) | `Q8_0` (norms in `F32`) | Smallest footprint (`v2` 78-layer build, imatrix) |
| [`mmproj-GLM-5.3-Flash-REAP50-F16.gguf`](./mmproj-GLM-5.3-Flash-REAP50-F16.gguf) | `v3` | **1.05 GiB** | — | `F16` / `F32` | Vision multimodal projector (`mmproj`) for image+text inputs in `llama.cpp` |
| [`imatrix-GLM-5.3-Flash-REAP50.dat`](./imatrix-GLM-5.3-Flash-REAP50.dat) | `v3` | **241 MB** | — | — | Activation importance matrix (905 tensors across layers `0..78`, 128 × 4,096 = 524,288 calibration tokens) |
| [`glm5-next-llama.cpp.patch`](./glm5-next-llama.cpp.patch) / [`glm5-next-llama.cpp.bundle`](./glm5-next-llama.cpp.bundle) | `v3` | **0.29 MB / 308 MB** | — | — | `llama.cpp` patch and self-contained git bundle adding `glm5-next` / `deepseek_v32` MLA + DSA Indexer + MTP support |
| [`make_mtp_draft.py`](./make_mtp_draft.py) & [`research/MTP_CAPTURE_FINDINGS.md`](./research/MTP_CAPTURE_FINDINGS.md) | `v3` | — | — | — | Standalone MTP draft extractor & teacher-trace capture scripts ([`research/nvfp4_mtp_gate.py`](./research/nvfp4_mtp_gate.py), [`research/quant_transfer.py`](./research/quant_transfer.py)) |

---

## Architecture Overview (`zai-org/GLM-5.3-Flash`)

`GLM-5.3-Flash` uses the **`GlmMoeDsaForCausalLM` (`deepseek_v32` / `deepseek2`)** architecture combining **Multi-head Latent Attention (MLA)** with **Dynamic Sparse Attention (DSA)** and fine-grained **Mixture-of-Experts (MoE)** (see [`wiki/10-target-model.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/10-target-model.md) and [`docs/ARCH_DELTA.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/ARCH_DELTA.md)):

| Parameter | Base ([`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)) | This Model (`REAP-50`) |
|---|---|---|
| **Architecture** | `GlmMoeDsaForCausalLM` (`deepseek_v32`) | `deepseek2` (GGUF) |
| **Total Parameters** | ~301B | **~157B** |
| **Active Parameters / Token** | ~12.5B | **~12.5B** (identical) |
| **Layers** | 78 (3 dense + 75 MoE) + 1 MTP layer (78) | **78 (3 dense + 75 MoE)** (+ L78 MTP in `v3` files) |
| **Hidden Size (`n_embd`)** | 5,120 | 5,120 |
| **Dense FFN Length (Layers 0–2)** | 12,288 | 12,288 (unpruned) |
| **Routed Experts per MoE Layer** | 256 | **128** (REAP-50 pruned; [`artifacts/masks/mask.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/masks/mask.json)) |
| **Active Routed Experts (`top_k`)** | 8 (`n_group=1`, sigmoid gating) | **8** (`routed_scaling_factor=2.5`) |
| **Shared Experts per MoE Layer** | 1 (`moe_intermediate_size=1536`) | **1** (unpruned, `Q8_0`) |
| **Expert Intermediate Size** | 1,536 | 1,536 |
| **MLA Attention** | 64 heads (`q_lora_rank=2048`, `kv_lora_rank=512`, `qk_rope_head_dim=64`, `qk_nope_head_dim=192`, `v_head_dim=256`) | Unchanged (`Q8_0`) |
| **DSA Indexer** | 32 heads, `head_dim=128`, `index_topk=2048` | Unchanged (`Q8_0` / `F32`) |
| **Max Context Length** | 202,752 (YaRN `factor=4.0`, `beta_fast=32`, `beta_slow=1`) | **202,752** |
| **Vocabulary Size** | 154,880 | 154,880 |

---

## How This Model Was Built

### 1. Domain-Balanced REAP-50 Pruning (256 → 128 Experts)
Rather than naive frequency or magnitude pruning, we implemented **REAP (Router-weighted Expert Activation Pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999))** across all 75 MoE layers (`layers 3..77`) using exact blockwise FP8 (`E4M3` + `F8_E8M0` scale) dequantization on the 301B base model (documented in [`wiki/30-reap.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/30-reap.md) and [`wiki/31-pass3-hope-method.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/31-pass3-hope-method.md)):

$$\text{REAP}_j = \frac{1}{|T_j|} \sum_{t \in T_j} g_{j,t} \cdot \|\text{FFN}_j(x_t)\|_2$$

To prevent domain collapse on specialized capabilities (code, math, Chinese, and agentic tool calling), calibration activations were collected across **5 balanced domain pools** ([`wiki/80-calibration.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/80-calibration.md), [`wiki/85-corpus-sources.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/85-corpus-sources.md)):
1. **General English** (`general`)
2. **Source Code & Software Engineering** (`code`)
3. **Competition Mathematics & Symbolic Reasoning** (`math`)
4. **Chinese Language & Cultural Knowledge** (`zh`)
5. **Structured Tool-Calling & Agentic Traces** (`tool`)

For every MoE layer, we took the **union of the top-16 specialist experts from each of the 5 domains** (protecting domain-critical specialists even if their global frequency is low), enforced a minimum activation frequency floor (`f_j >= 40`), and filled the remaining slots up to **128 experts** using normalized cross-domain REAP saliency ([`artifacts/masks/mask.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/masks/mask.json)).

### 2. Post-REAP Layer-77 Shared-Expert Scaling (`v3` files: `Q4_K_M`, `Q3_K_M`)
When pruning 256 → 128 routed experts with `norm_topk_prob=true`, the surviving top-8 routed expert weights renormalize to sum to `1.0`, slightly shifting the variance ratio between the routed MoE branch and the unpruned shared expert entering the final RMSNorm (`model.norm`).
- We tested per-layer router ridge regression (`W_router`), per-layer scalar `alpha_l`, and final-layer corrections on held-out calibration data ([`wiki/70-healing.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/70-healing.md), [`wiki/98-router-kd-global.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/98-router-kd-global.md)).
- **Finding:** Modifying router weights across 75 layers causes severe cascading route flips (`kl_top32` `0.311 → 2.66`), so **all router weights (`blk.*.ffn_gate_inp.weight`) and biases (`exp_probs_b.bias`) are kept 100% bit-identical to the base checkpoint's surviving rows**.
- Rescaling only the final MoE layer's shared-expert output projection (`blk.77.ffn_down_shexp.weight *= 0.7209`, fit via least-squares against the teacher's pre-norm hidden state; see [`artifacts/heal_refit.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/heal_refit.json)) cannot cascade into any downstream router, preserves `kl_top32` (`0.3204`) and `top1_agree` (`83.39%`), and eliminates rare high-entropy tail outliers (`kl_full` reduced from `0.506 → 0.388`). This rescale is included in the `v3` GGUFs ([`GLM-5.3-Flash-REAP50-Q4_K_M.gguf`](./GLM-5.3-Flash-REAP50-Q4_K_M.gguf) and [`GLM-5.3-Flash-REAP50-Q3_K_M.gguf`](./GLM-5.3-Flash-REAP50-Q3_K_M.gguf)).

### 3. Direct Block-Streaming FP8 → GGUF Conversion (Zero BF16 Intermediate)
Standard `llama.cpp` conversion requires dequantizing FP8 checkpoints to a ~315 GB BF16 intermediate first. We built a custom streaming converter ([`wiki/60-quantization.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/60-quantization.md)) that dequantizes `zai-org/GLM-5.3-Flash`'s `F8_E4M3` + `F8_E8M0` (128-element exponent-scale blocks) on the fly and quantizes directly into `Q8_0` GGUF blocks on disk:
- **All attention, MLA latent projections, DSA Indexer tensors, shared experts, token embeddings, output head, and dense layers (`0..2`) are quantized directly to `Q8_0` (or `F32` for norms/biases).**
- **Only the 3D packed routed expert tensors (`blk.*.ffn_gate_exps.weight`, `blk.*.ffn_up_exps.weight`, `blk.*.ffn_down_exps.weight`)** are quantized to `Q4_K_M`, `Q4_K_S`, `IQ4_XS`, `Q3_K_M`, or `IQ3_M`, with `--output-tensor-type q8_0 --token-embd-type q8_0` ([`artifacts/output_tensor_inventory.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/output_tensor_inventory.json)).
- Importance matrix use differs by file, as recorded in each file's own GGUF metadata: **`IQ3_M`, `IQ4_XS` and `Q3_K_M` were quantised with the published [`imatrix-GLM-5.3-Flash-REAP50.dat`](./imatrix-GLM-5.3-Flash-REAP50.dat); `Q4_K_M` and `Q4_K_S` were not** (for `Q4_K_M`, an A/B check on the confirmation split showed no gain from the imatrix once the attention/shared-expert path is locked at `Q8_0`).

---

## Performance on Jetson AGX Thor (128 GB Unified LPDDR5X, SM110)

### 1. `llama.cpp` (`b8809+`, CUDA 13.0, Unified Memory)

Measured on a single **NVIDIA Jetson AGX Thor (128 GB unified memory, 273 GB/s LPDDR5X)** (perplexity and NIAH logs in [`artifacts/ppl_GLM-5.3-Flash-REAP50-Q4_K_M.txt`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/ppl_GLM-5.3-Flash-REAP50-Q4_K_M.txt) and [`artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json)):

| Quant | Vintage | File Size | Prompt Eval (`pp64`) | Token Generation (`tg64`) | Memory Headroom on 128 GB Thor |
|---|---|---:|---:|---:|---:|
| **[`Q4_K_M`](./GLM-5.3-Flash-REAP50-Q4_K_M.gguf)** | `v3` | 94.8 GiB | **75.9 tok/s** | **13.5 tok/s** | ~28 GiB free for KV cache / OS |
| **[`Q4_K_S`](./GLM-5.3-Flash-REAP50-Q4_K_S.gguf)** | `v2` | 90.0 GiB | **76.4 tok/s** | **13.9 tok/s** | ~33 GiB free for KV cache / OS |
| **[`IQ4_XS`](./GLM-5.3-Flash-REAP50-IQ4_XS.gguf)** | `v2` | 85.6 GiB | **74.2 tok/s** | **13.8 tok/s** | ~37 GiB free for KV cache / OS |
| **[`Q3_K_M`](./GLM-5.3-Flash-REAP50-Q3_K_M.gguf)** | `v3` | 77.3 GiB | **71.8 tok/s** | **14.3 tok/s** | ~45 GiB free for KV cache / OS |
| **[`IQ3_M`](./GLM-5.3-Flash-REAP50-IQ3_M.gguf)** | `v2` | 72.9 GiB | **70.5 tok/s** | **14.7 tok/s** | ~50 GiB free for KV cache / OS |

### 2. Custom Pure-CUDA Thor Engine ([`patrickbdevaney/glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server))

If you are running on an **NVIDIA Jetson AGX Thor (`sm_110a`)** and want maximum throughput + lossless **Multi-Token Prediction (MTP)** speculative decoding (`layer 78`), use our companion pure C++/CUDA server [`patrickbdevaney/glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server) (see [`docs/ROOFLINE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/ROOFLINE.md) and [`docs/SPEC_DECODE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/SPEC_DECODE.md)):

| Mode | Decode Throughput | Speedup vs `llama.cpp` (`13.5 tok/s`) | Notes & Verification Source |
|---|---:|---:|---|
| **Autoregressive (`draft_k=0`)** | **20.9 tok/s** | **1.55×** | Native `sm_110a` CUDA graph + zero-copy `mmap` (`14.11 GB/tok` = 93% of 273 GB/s roofline; [`docs/ROOFLINE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/ROOFLINE.md)) |
| **MTP Speculative (`draft_k=3`, English/narrative)** | **28.0 tok/s** | **2.07×** | Lossless greedy tree verification (`65%` draft acceptance; [`docs/SPEC_DECODE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/SPEC_DECODE.md)) |
| **MTP Speculative (`draft_k=3`, Math/Reasoning)** | **31.7 tok/s** | **2.35×** | Lossless greedy tree verification (`77%` draft acceptance; [`docs/SPEC_DECODE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/SPEC_DECODE.md)) |
| **MTP Speculative (`draft_k=3`, Structured JSON/Code)** | **38.8–42.4 tok/s** | **2.87×–3.14×** | Lossless greedy tree verification (`89–100%` draft acceptance; [`docs/OPTIMIZATION_LOG.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/OPTIMIZATION_LOG.md)) |

---

## How to Run with `llama.cpp`

Requires a recent `llama.cpp` build (`b8500+` or [`glm5-next-llama.cpp.patch`](./glm5-next-llama.cpp.patch)) with `deepseek2` / `deepseek_v32` Dynamic Sparse Attention Indexer support (`blk.*.indexer.*`).

> **Important Note on Stop Tokens (EOG):**
> GLM-5.3-Flash uses **three** stop tokens in its chat template: `<|endoftext|>` (`154820`), `<|user|>` (`154827`), and `<|observation|>` (`154829`).
> - In the **`v3` GGUFs (`Q4_K_M` and `Q3_K_M`)**, `eos_token_id` in the header is already set to `154827` (`<|user|>`), which stops normal single-turn and multi-turn user/assistant chat automatically.
> - In the **`v2` GGUFs (`Q4_K_S`, `IQ4_XS`, `IQ3_M`)**, `eos_token_id` is `154820` (`<|endoftext|>`), so without overriding EOG token IDs the model may generate past the end of an assistant turn into a simulated `<|user|>` turn.
> - For **all** files (especially when using tool calls that stop on `<|observation|>` = `154829`), we recommend passing `--override-kv deepseek2.ggml.eos_token_ids=arr:i32:154820,154827,154829` (or passing `stop: ["<|endoftext|>", "<|user|>", "<|observation|>"]` in your API request).

### CLI Example (`llama-cli`)

```bash
./llama-cli \
  -m GLM-5.3-Flash-REAP50-Q4_K_M.gguf \
  -ngl 99 \
  -c 32768 \
  -fa on \
  --override-kv deepseek2.ggml.eos_token_ids=arr:i32:154820,154827,154829 \
  -p "<|system|>\nYou are a helpful assistant.<|user|>\nProve that there are infinitely many primes congruent to 3 mod 4.<|assistant|>\n<think>\n"
```

### OpenAI-Compatible Server (`llama-server`)

```bash
./llama-server \
  -m GLM-5.3-Flash-REAP50-Q4_K_M.gguf \
  -ngl 99 \
  -c 65536 \
  -fa on \
  --override-kv deepseek2.ggml.eos_token_ids=arr:i32:154820,154827,154829 \
  --host 0.0.0.0 \
  --port 8080
```

> **Tip for Unified-Memory Systems (Jetson AGX Thor / DGX Spark / AMD Strix Halo):**
> When running with CUDA on Linux unified memory (`-ngl 99`), `llama.cpp` disables `mmap` by default if `cudaHostRegister` fails on large files, causing a double allocation during model load. If `Q4_K_M` (94.8 GiB) hits OOM during the initial load on a 128 GB system, either:
> 1. Use `Q4_K_S` (90.0 GiB), `IQ4_XS` (85.6 GiB), or `Q3_K_M` (77.3 GiB), or
> 2. Patch `ggml-cuda` to keep `use_mmap = true` on integrated/unified-memory GPUs (`integrated == true`), which keeps peak resident memory equal to the 94.8 GiB file size.

---

## 🔎 Provenance & Verification Ledger

Every claim, metric, and artifact on this model card can be audited directly against the files in this repository and the companion open-source pipelines:

| Claim / Artifact | Primary Verification Source |
|---|---|
| **Upstream Base Model (`301B` FP8, MIT)** | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| **REAP Pruning Formulation** | Lasby et al., 2025 — [*REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*](https://arxiv.org/abs/2510.13999) (`arXiv:2510.13999`) |
| **REAP-50 Pipeline & Integrity Audit** | [`patrickbdevaney/glm-5.3-reap`](https://github.com/patrickbdevaney/glm-5.3-reap) (`main` @ `8540a0a`) · [`EVAL_INTEGRITY.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/EVAL_INTEGRITY.md) · [`wiki/97-end-to-end-audit.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/97-end-to-end-audit.md) |
| **128-Expert Keep-Set & Evaluation Logs** | [`artifacts/masks/mask.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/masks/mask.json) · [`artifacts/mask_eval.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/mask_eval.json) · [`wiki/97-evaluation.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/97-evaluation.md) |
| **Layer-77 Shared-Expert Scale (`0.7209x`) & Router KD Audit** | [`artifacts/heal_refit.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/heal_refit.json) · [`wiki/70-healing.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/70-healing.md) · [`wiki/98-router-kd-global.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/98-router-kd-global.md) |
| **Per-Quant Perplexity, NIAH, HumanEval & BFCL Results** | [`artifacts/ppl_GLM-5.3-Flash-REAP50-Q4_K_M.txt`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/ppl_GLM-5.3-Flash-REAP50-Q4_K_M.txt) · [`artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/niah_GLM-5.3-Flash-REAP50-Q4_K_M.json) · [`artifacts/humaneval_glm5.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/humaneval_glm5.json) · [`artifacts/bfcl_glm5.json`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/artifacts/bfcl_glm5.json) |
| **Activation Importance Matrix & MTP Research** | [`./imatrix-GLM-5.3-Flash-REAP50.dat`](./imatrix-GLM-5.3-Flash-REAP50.dat) · [`./research/MTP_CAPTURE_FINDINGS.md`](./research/MTP_CAPTURE_FINDINGS.md) · [`./make_mtp_draft.py`](./make_mtp_draft.py) |
| **Pure-CUDA Thor Engine (`20.9 tok/s` AR, `28–42.4 tok/s` MTP)** | [`patrickbdevaney/glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server) (`main` @ `399a837`) · [`docs/ROOFLINE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/ROOFLINE.md) · [`docs/SPEC_DECODE.md`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server/blob/main/docs/SPEC_DECODE.md) |

---

## Citation & Credits

- **Base Model:** [Z.ai (`zai-org/GLM-5.3-Flash`)](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT License)
- **Pruning Method:** [REAP: Router-weighted Expert Activation Pruning](https://arxiv.org/abs/2510.13999)
- **Quantization & REAP-50 Pipeline:** [`patrickbdevaney/glm-5.3-reap`](https://github.com/patrickbdevaney/glm-5.3-reap) & [`patrickbdevaney/glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server)

