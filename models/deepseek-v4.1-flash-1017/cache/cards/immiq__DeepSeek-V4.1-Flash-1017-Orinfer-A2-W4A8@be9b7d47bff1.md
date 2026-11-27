---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
language:
- zh
- en
tags:
- orinfer
- ascend910b
- w4a8
- smoothquant
- gptq
- safetensors
- dspark
- vision
inference: false
---

# DeepSeek-V4.1-Flash-1017 — Orinfer A2 W4A8 (Unrotated SmoothQuant + 262K Act-Order Hessian-GPTQ + L15/L39 W8)

Native Ascend 910B (TP8, CANN 9.2, Linux aarch64, glibc 2.34+) activation-calibrated quantized model data for **Orinfer**, supporting text, thinking, DSML function tools, single/multi-image vision, and DSpark parallel speculative decoding.

> **Breaking Weight Format Upgrade (2026-11-26)**: This revision replaces the legacy QuaRot32-rotated checkpoint with an **unrotated, activation-calibrated (SmoothQuant + 262K-token Act-Order Hessian-GPTQ)** re-quantization directly from `deepseek-ai/DeepSeek-V4.1-Flash-1017` and requires the matching `engram.cast_fp32` execution package below (not compatible with legacy rotated `v0.3.x`/`v0.4.0` execution packages). Historical tags (`v0.3.0`–`v0.4.0`) remain intact for rollback. The 96 Engram host table shards (`weights/host/`, 206 GiB) and `weights/dspark/embedding.safetensors` are byte-identical to earlier revisions.

## Matching Execution Package

```text
0a6c09f4722dfa655bdd2ae34acfa1301e70ff85ee933a6ac93009ac363af234
```

- **Source commit**: `14a30a9307476597ee2e1e325f0feae5c8e3f38e`
- **Execution archive**: `orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-norot-gptq-sq-20261126.tar.xz` (`sha256: dd025b1a40bbac004395a4c812cd67f087076c6467be2a69738c0545f8e51d27`)
- **Weight shards**: `205` safetensors containers (`508.41 GiB` total, including 206 GiB host-resident Engram tables)

## Quantization Architecture

Fully reproducible via `orinfer_build.models.deepseek_v41.calibrate_norot` and `orinfer_build.models.deepseek_v41.quantize_norot`:

- **Unrotated Native-Grid Basis**: Eliminates global QuaRot32, `ln_fuse`, and legacy AWQ transforms so weights stay in the official MXFP4 power-of-two grid. Legacy rotated Engram kernels are replaced by a 20-AIC persistent BF16→FP32 cast (`engram.cast_fp32`).
- **Execution-Graph-Exact Activation Calibration (262,144 Disjoint Multi-Domain Tokens)**:
  - **Zero-overhead SmoothQuant**: Calibrated `q_norm → wq_b + indexer.wq_b` (all 40 layers, up to −92% error), `attn_norm → wq_a + wkv + indexer.weights_proj` (36 non-KV-source W8A8 layers, up to −56%), `shared_experts` `w3/w2` (exact `8×288` TP8 A8 + clamp-exact `|u_max| <= 8.5`, up to −58%), `ffn_norm` W4A8-gated SmoothQuant (+4%–+15% across W4 layers), and per-expert clamp-exact `w3/w2` SwiGLU SmoothQuant.
  - **Act-Order Blocked Hessian-GPTQ (38 W4 Routed-Expert Layers)**: Routing-weight-exact (`wt^2`) shrinkage-blended UP Hessians + full-rank DOWN Hessians (~4,100 tok/expert) with exact closed-form H-optimal per-channel scales (plus Gauss-Seidel coordinate-descent refinement on `L00, L02–L05, L07, L37–L38`), cutting W4 weight H-NMSE 2–7× below data-free SSZ (`up = 0.00066–0.00209`, `down = 0.00164–0.00322`) in Transposed-NZ layout.
- **Sensitivity-Optimal Lossless W8 Layers (`L15, L39`)**: Selected by 40-layer out-of-sample residual-impact audit (`L15` is #1 sensitivity by 5×; `L39` protects the final layer) while keeping ready HBM at `53.8 GiB/card` so on-device prefix cache stays at full `4.64 GiB` capacity for high-concurrency serving.
- **Shared KV-Source Protection (`L02, L08, L14, L20`)**: BF16 `attn.wq_a` and `attn.wkv` with tuned small-M `row_selections`.
- **DSpark Draft (`mtp.0/1/2`)**: Exact-grid lossless full W8 routed experts (`up NMSE = 0.000e+00`).

## Verified Results (8× Ascend 910B3 TP8, CANN 9.2)

| Metric | Previous Rotated `v0.4.5` | Data-Free `v2` (`SSZ-W4 + L38–39 W8`) | **This Revision (`0a6c09f4...`)** | Delta vs `v0.4.5` |
| --- | ---: | ---: | ---: | ---: |
| `mean KL(Official MXFP4+FP8 \|\| Model)` (1,556 pos) | `0.03075` | `0.02467` | **`0.01992`** | **−35.2%** (95% CI `[-0.0133, -0.0083]`) |
| `median KL` / `p95 KL` | `0.00846` / `0.130+` | `0.00621` / `0.105` | **`0.00533` / `0.07837`** | **−37.0% median** |
| `mean KL(Model \|\| Official)` | `0.03101` | `0.02384` | **`0.01918`** | **−38.1%** |
| Official Argmax Match / Top-3 Cover | `94.15%` / — | `95.69%` / — | **`95.76%` / `99.55%`** | **+1.61 pp** |
| DSpark 256-Window Acceptance (`committed/round`) | `67.11%` (`4.3555`) | `70.47%` (`4.5234`) | **`69.84%` (`4.4922`, test `81.56%`)** | **+2.73 pp (`+0.1367`)** |
| Frozen 240s Mini-Weka (`c=16`, `seed=20261002`) | `129` reqs (`0` err) | `130` reqs (`0` err) | **`130` reqs (`0` err)** | **+0.8%** (`22,656` decode toks) |
| Mini-Weka `prefill` / `verify` / `draft` (ms/iter) | `124.16` / `90.83` / `4.78` | `123.01` / `88.08` / `4.85` | **`122.49` / `88.62` / `4.77`** | Faster prefill, verify & draft |
| Min Free HBM / Card (peak serving) | `4.12 GiB` | `7.34 GiB` | **`7.35 GiB`** | Clean `SIGTERM`, 8/8 idle |

Full machine-readable metrics and operator validation logs are recorded in [benchmarks/a2-norot-gptq-sq-20261126.json](benchmarks/a2-norot-gptq-sq-20261126.json).

## Usage

```bash
orinfer package install orinfer-exec-deepseek-v41-flash-1017-w4a8-ascend910b-norot-gptq-sq-20261126.tar.xz
hf download immiq/DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 --local-dir ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8
orinfer validate-model ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8

source /usr/local/Ascend/cann/set_env.sh
export ORINFER_A2_HOST_TABLE_MIB=212000
export ORINFER_VISION_CACHE_MIB=1024
export HCCL_OP_EXPANSION_MODE=AIV HCCL_NPU_SOCKET_PORT_RANGE=auto
orinfer serve ./DeepSeek-V4.1-Flash-1017-Orinfer-A2-W4A8 \
  --model deepseek-v4.1 --listen 127.0.0.1:8000 \
  --mtp-drafts auto --max-speculative-tokens 0 --graph full \
  --prefix-cache-mib 8192 --kv-offload-mib 262144 \
  --request-body-mib 64 \
  --queue-timeout-ms 60000 --output-timeout-ms 180000 \
  --drain-timeout-ms 120000
```

## Attribution

Model weights retain the upstream MIT license in `LICENSE`; upstream model documentation is preserved in `SOURCE-README.md`. Orinfer is licensed under LGPL-3.0-or-later.
