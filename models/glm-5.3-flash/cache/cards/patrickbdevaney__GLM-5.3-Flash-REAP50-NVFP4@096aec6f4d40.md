---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
new_version: patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2
library_name: transformers
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - glm-5.3-flash
  - deepseek_v32
  - moe
  - reap
  - expert-pruning
  - nvfp4
  - fp4
  - compressed-tensors
  - blackwell
  - jetson-thor
---

<div align="center">

# GLM-5.3-Flash REAP-50 (NVFP4 — Pass-1 Archive)

[![New Version Available](https://img.shields.io/badge/Superseded_By-GLM--5.3--Flash--REAP50--NVFP4--v2-brightgreen)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)
[![Base Model](https://img.shields.io/badge/Base_Model-zai--org%2FGLM--5.3--Flash-blue)](https://huggingface.co/zai-org/GLM-5.3-Flash)
[![Format](https://img.shields.io/badge/Format-NVFP4_compressed--tensors_(94.6_GiB)-76b900)](#summary)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Pass-1 (v1) 50% REAP expert-pruned NVFP4 (`94.6 GiB`) checkpoint of `zai-org/GLM-5.3-Flash` — superseded by [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2).**

</div>

> [!IMPORTANT]
> **Newer Version Available (`v2`):** This is the initial Pass-1 (`v1`) single-pool REAP-50 checkpoint, retained for archival reproducibility. For all new deployments, please use **[`patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)** (or **[`patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF)**), which adds a 5-domain balanced keep-set (`general`, `code`, `math`, `zh`, `tool`), union top-16 specialist protection, a frequency floor (`f_j >= 40`), and pre-calibrated `input_scale` tensors across all 300 expert projections.

---

50% expert-pruned derivative of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) produced with **REAP** (Router-weighted Expert Activation Pruning; [arXiv:2510.13999](https://arxiv.org/abs/2510.13999)) on a single NVIDIA Thor (128 GB unified memory).

---

## Summary

| Property | Original (`zai-org/GLM-5.3-Flash`) | This model (`REAP-50`) |
|---|---|---|
| Total parameters | ~301 B | **~157 B** |
| Active parameters / token | ~12.5 B | **~12.5 B** (unchanged) |
| Layers | 78 (3 dense + 75 MoE) | 78 (3 dense + 75 MoE) |
| Routed experts / MoE layer | 256 | **128** (50% pruned) |
| Active experts / token (`num_experts_per_tok`) | 8 | **8** |
| Shared expert / MoE layer | 1 | **1** (untouched) |
| Attention | MLA + DSA (`deepseek_v32`, `kv_lora_rank=512`, `q_lora_rank=2048`, `index_topk=2048`) | **Untouched** |
| Context length | 202,752 | **202,752** |
| Checkpoint Size | 296 GiB (FP8) | **94.6 GiB (NVFP4)** *(157.0 GiB in companion [`GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8))* |
| License | MIT | **MIT** |

---

## Pruning methodology

Each MoE layer `l in [3..77]` has 256 routed SwiGLU experts (`gate_proj`, `up_proj`, `down_proj`, intermediate size 1536) routed by a sigmoid gate (`routed_scaling_factor = 2.5`, `norm_topk_prob = true`, `e_score_correction_bias`).

1. **Calibration corpus**: 320 sequences x 2048 tokens (655,360 tokens) drawn from fineweb-edu, Wikipedia, Python/C++/Rust code, GSM8K/math, reasoning traces, and Chinese/English prompts (`calibration_tokens.pt`, SHA-256 recorded in `reap_plan.json`).
2. **Exact teacher forward**: Layers 0..77 were executed sequentially on GPU in native `bfloat16` using the exact `deepseek_v32` MLA + rotary + lightning-indexer attention implementation and exact block-scale FP8 (`F8_E4M3` x `F8_E8M0` UE8M0) dequantization (`saliency_pass.py`).
3. **REAP saliency metric** (Equation 6 of arXiv:2510.13999):
   For each routed expert `j` in layer `l`, over the set `T_j` of calibration tokens where expert `j` is in the router's top-8:
   ```
   S_j = (1 / |T_j|) * sum_{t in T_j} g_{t,j} * || down_proj_j( silu(gate_proj_j(h_t)) * up_proj_j(h_t) ) ||_2
   ```
   where `g_{t,j}` is the normalized top-8 routing weight (prior to `routed_scaling_factor`, which is a constant multiplier across experts).
4. **Expert selection & weight remapping**:
   In each MoE layer, the 128 experts with highest `S_j` are retained in ascending original-index order `kept = [k_0 < k_1 < ... < k_127]`:
   - `mlp.gate.weight` (`[256, 5120]` -> `[128, 5120]`) and `mlp.gate.e_score_correction_bias` (`[256]` -> `[128]`) are sliced to `kept`.
   - Routed experts `mlp.experts.{kept[new_idx]}.*` are renamed to `mlp.experts.{new_idx}.*` (`0..127`) and quantized to NVFP4 (`weight_packed`, `weight_scale`, `weight_global_scale`).
   - All attention projections, DSA indexer weights, dense layers (0..2), shared experts, norms, embeddings, and `lm_head` are preserved **bit-for-bit** from `zai-org/GLM-5.3-Flash`.
   - The training-only MTP head (`model.layers.78`, `num_nextn_predict_layers`) is omitted (`num_nextn_predict_layers = 0`).

Full per-layer keep/prune tables, activation frequencies, and REAP scores are documented in the companion pipeline repository [`patrickbdevaney/glm-5.3-reap`](https://github.com/patrickbdevaney/glm-5.3-reap) (and [`reap_metadata.json`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8/blob/main/reap_metadata.json) in the companion FP8 repo).

---

## Quantization format (NVFP4 `compressed-tensors`)

Routed expert projections (`mlp.experts.{0..127}.{gate_proj,up_proj,down_proj}`) are stored in `compressed-tensors` `fp4-pack-quantized` format for NVIDIA Blackwell / Thor (`sm_100` / `sm_110`):
- `weight_packed`: `uint8` `[out_features, in_features // 2]` (two signed E2M1 FP4 nibbles per byte; low nibble = even column, high nibble = odd column)
- `weight_scale`: `float8_e4m3fn` `[out_features, in_features // 16]` (group-size 16 scale)
- `weight_global_scale`: `float32` `[1]` (`dequant = E2M1(weight_packed) * (weight_scale / weight_global_scale)`)

All non-routed-expert weights (dense layers 0..2, shared experts, MLA attention, DSA indexer) remain in their original upstream `F8_E4M3` + `F8_E8M0` / `BF16` precision.

---

## Usage (vLLM / SGLang)

```bash
vllm serve patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --max-model-len 32768
```

---

## 🔎 Provenance & Verification Ledger

| Claim / Artifact | Primary Verification Source |
|---|---|
| **Upstream Base Model (`301B` FP8, MIT)** | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) · [`./LICENSE`](./LICENSE) |
| **Superseded By (v2 Domain-Balanced + Calibrated Input Scales)** | [`patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2) |
| **Model Config & 62-Shard Safetensors Index** | [`./config.json`](./config.json) · [`./model.safetensors.index.json`](./model.safetensors.index.json) · [`./chat_template.jinja`](./chat_template.jinja) |
| **REAP-50 Pipeline & Integrity Audit** | [`patrickbdevaney/glm-5.3-reap`](https://github.com/patrickbdevaney/glm-5.3-reap) (`main` @ `8540a0a`) · [`EVAL_INTEGRITY.md`](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/EVAL_INTEGRITY.md) |
| **Pure-CUDA Thor Engine (`sm_110a`)** | [`patrickbdevaney/glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server) (`main` @ `399a837`) |

---

## Attribution & License

- Base model: [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.ai (MIT License, [`./LICENSE`](./LICENSE)).
- Pruning method: *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression* (Lasby et al., 2025, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999)).
