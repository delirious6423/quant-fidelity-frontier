---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE
pipeline_tag: image-text-to-text
language: [zh, en]
tags: [orinfer, jetson, e8p, int8, mixed-precision, vision-language, safetensors]
---

# Qwen3.8-Flash-Next · Orinfer E8P Q2A8

[中文](README_zh.md) · [Hugging Face](https://huggingface.co/immiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8) · [ModelScope](https://modelscope.cn/models/iammiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8)

A high-quality quantized version (**`E8P-HQ v3`** on `main`, drop-in compatible with `v0.4.1`) of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) for **Jetson AGX Orin 64GB / CUDA SM87**, deployed with [Orinfer v0.4.1](https://github.com/iMMIQ/orinfer/releases/tag/v0.4.1). It supports text, images and multiple images, streaming Chat Completions, native MTP, prompt lookup, prefix caching, continuous batching and resumable JSONL offline inference. Translation requests use `enable_thinking: false`; thinking follows the original model's chat template.

## Quantization and files

Routed experts use integer E8P near-2-bit weights with INT8 computation. GDN QKV/Z, QSA Q/gate, all 48 `ssm_out` / `attn_output` (`2560 × 6144`) output projections, and the shared output head use single-resident signed-block128 Q4A8. Shared FFN and sensitive MTP projections retain W8; HC retains BF16 and vision FP16. CPU embedding/PLE tables use bounded row caches.

### `E8P-HQ v3` Quality Upgrade (`main`, **2026-11-25**, 100% `v0.4.1` Drop-In)

All quantized weight banks on `main` have been upgraded directly from BF16 `Qwen/Qwen3.8-Flash-Next` using a **65,536-token prose-balanced calibration set** (37.5% EN, 25.0% ZH, 18.75% Code/Agent, 18.75% `<think>` Math), keeping the exact physical `v0.4.1` tensor layouts and SM87 execution-package digest (`zero` kernel rebuild, `zero` speed overhead):

- **Direct BF16 → Hessian-GPTQ Row-Q4** (`Q4_output`, all 48 `ssm_out`/`attn_output`, 36 `attn_qkv`/`attn_gate`, 12 `attn_q`): eliminates baseline `v0.4.1`'s weight-only RTN / W8→Q4 double-quantization via 64-trial A8 sign search + Multi-Start Blocked-Cholesky Hessian-GPTQ (`damping=0.02` attn, `0.03` `lm_head`), cutting `ssm_out`/`attn_output` NMSE by **66.3%–92.0%** and `Q4_output` KL by **54.2%**.
- **48L MoE Routed Experts + Real-Activation MTP Layer 48 (`E8P-HQ v3`)**: 64-trial online-A8 sign search + per-expert regularized Mahalanobis Block-8 LDLQ (`damping_gu=0.25, damping_dn=0.30, tau_gu=2560, tau_dn=1280`) + 1 coordinate-descent refinement sweep (`cd_sweeps=1`) + SwiGLU cubic-attenuation compensation (`gu_boost=1.035` + calibration gain `a_cal`). MTP Layer 48 is calibrated on real post-trunk Layer-48 activations.
- **Dense W8 & 320M CPU PLE**: Hessian-aware optimal clipping + GPTQ for all W8 dense projections; 32-trial Paley20×Walsh8 sign search + 4-start MS-ALS + 3-ULP FP16 polish for all 320M PLE rows.

Deployed read-only GPU data, including scales, derived constants and precision exceptions: **38,487,903,152 bytes** (`-377.2 MB` vs. v0.2.0–v0.4.0), approximately **2.391 bits per recorded GPU weight parameter**. Stored safetensors: **52.56 GB** across **35 shards** (27 GPU shards, `38.49 GB`; 8 CPU shards, `14.08 GB`). Context capacity: **262,144 tokens**; KV cache is group-64 INT8 with FP16 scales and persistent GDN state is FP32. Quantization seed: `20261002`.

The repository contains HF configuration/tokenizer assets, the native chat template, standard safetensors containers and `cache/model.json`. The descriptor fixes physical tensor layouts and the Schema 2 execution-package digest. The matching statically linked `orinfer` binary and pure-data Schema 2 execution package (`package.json`, `bindings.json`, `assets.safetensors` with cubins and source) are distributed through GitHub Release.

## Deploy

Download `main` (recommended `E8P-HQ v3` upgraded weights; the original weight-only baseline remains pinned on tag [`v0.4.1`](https://huggingface.co/immiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8/tree/v0.4.1)) of this model repository into `./model`. Prebuilt deployment uses the aarch64 Rust CLI, NVIDIA CUDA Driver and the matching SM87 Schema 2 execution package.

Execution-package digest: `28040d50317572c45552b2da77517fbffb49da0547ca3ea8882c1cff2db9664a` (verified with `orinfer validate-model`: `tensor_count = 2787`, `shard_count = 27`, `weight_bytes = 38487903152`).

For interactive multi-turn and concurrent agent workloads, start with `--mtp-drafts auto --prompt-lookup --prefix-cache-mib 4096`. Add `--policy throughput` when prioritizing aggregate throughput over strict TPOT bounds.

```bash
curl -fLO https://github.com/iMMIQ/orinfer/releases/download/v0.4.1/orinfer-v0.4.1-linux-aarch64.tar.xz
curl -fLO https://github.com/iMMIQ/orinfer/releases/download/v0.4.1/orinfer-exec-flash-next-a8-sm87-v0.4.1.tar.xz
curl -fLO https://github.com/iMMIQ/orinfer/releases/download/v0.4.1/SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
tar -xf orinfer-v0.4.1-linux-aarch64.tar.xz
./orinfer-v0.4.1-linux-aarch64/bin/orinfer package install ./orinfer-exec-flash-next-a8-sm87-v0.4.1.tar.xz
./orinfer-v0.4.1-linux-aarch64/bin/orinfer validate-model ./model
./orinfer-v0.4.1-linux-aarch64/bin/orinfer serve ./model \
  --model qwen3.8-flash-next --cuda-graph decode_only \
  --mtp-drafts auto --prompt-lookup --prefix-cache-mib 4096
```

The listener is `[::]:8088`. OpenAI Chat Completions uses `/v1/chat/completions`; set `stream: true` for SSE. Images use `image_url`, and thinking output uses `reasoning_content`. The CLI also provides Ollama-compatible APIs and `orinfer batch` for JSONL jobs. See the [serving guide](https://github.com/iMMIQ/orinfer/blob/v0.4.1/docs/serving.md).

## Measured performance

### 240s Mini-Weka 512-trajectory benchmark (`v0.4.1`, **2026-11-23**)

Jetson AGX Orin 64GB (R36.4.3, CUDA SM87), `--cuda-graph decode_only --mtp-drafts auto --prompt-lookup --prefix-cache-mib 4096 --max-active-requests 4`, compact 512 multi-turn agent trajectories (cold first turns 2/3/4/6 Ki tokens, concurrency 4, 240 s wall limit, `temperature=1.0`, `ignore_eos=true`):

| Policy | Completed Reqs | Full Trajectories | Aggregate Output TPS | TTFT P50 / P95 (s) | Prefix Cache Hit | Errors / Graph Evictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `latency` | **102** | 12 | **46.50** | 1.00 / 4.08 | 79.3% | 0 / 0 |
| `throughput` | **102–105** | 13 | **46.31–48.15** | 0.80 / 3.29 | 79.2%–79.4% | 0 / 0 |

Output TPS includes queuing, prefill, generation and HTTP/SSE over the 240 s profiling window (`latency` `max_decode_gap_s = 0.401 s`, `prefill_suspend_s + prefill_resume_s = 0.387 s`).

### Fixed-length diagnostic snapshot (`v0.4.0`, **2026-11-20**)

Historical single/multi-request fixed-length diagnostics from v0.4.0 (`fc87e2ca`, greedy, seed `20261002`, `decode_only`, thinking off, prefix cache off; median of 3 warm runs). See the [performance reference](https://github.com/iMMIQ/orinfer/blob/v0.4.1/docs/performance.md) for methodology.

| Prefill 512 | Prefill 8K | Single decode, MTP (512 / 8K) | 8-request decode | Max requests / decode | Serve → API (s) | Serve → first output (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 979.3 | 1,081.8 | 62.9 / 31.5 | 163.3 | 32 / 284.8 | 15.10 | 15.27 |

## Quality scope

### End-to-End Evaluation vs. Official BF16 (`E8P-HQ v3` on `main` vs. baseline `v0.4.1`, **2026-11-25**)

Evaluated under the exact deployed precision (`Q4_output` + Row-Q4 attention + W8A8 dense + online-A8 E8P-Q2 experts + quantized CPU W8 embedding & PLE) across 5,015 trunk and 4,988 MTP strictly held-out positions (raw JSON logs archived in [`docs/`](./docs/)):

#### 48-Layer Trunk (Live `Q4_output` Head)

| Suite | Positions | `v0.4.1` Top-32 KL | **`E8P-HQ v3` Top-32 KL** | **KL Change** | `v0.4.1` Top-1 | **`E8P-HQ v3` Top-1** | `v0.4.1` ΔNLL | **`E8P-HQ v3` ΔNLL** | High-Conf Flips (`p≥0.5`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Orinfer 8 Scenes | 64 | 0.11216 | **0.02549** | **-77.3%** | 93.75% | **96.88%** | -0.09692 | **-0.04489** | 3/63 → **1/63** |
| Held-Out EN Prose | 1,344 | 0.86168 | **0.59349** | **-31.1%** | 69.12% | **76.93%** | +0.58107 | **+0.33563** | 277/1138 → **199/1138** |
| Held-Out ZH Prose | 896 | 0.96738 | **0.57007** | **-41.1%** | 59.04% | **70.20%** | +0.68181 | **+0.30537** | 213/678 → **132/678** |
| Held-Out Agent/Code | 166 | 0.41419 | **0.09766** | **-76.4%** | 71.69% | **86.14%** | -0.06583 | **+0.02235** | 22/124 → **4/124** |
| Held-Out `<think>` Math | 497 | 0.41262 | **0.04411** | **-89.3%** | 84.71% | **95.37%** | -1.11209 | **+0.18972** | 68/485 → **18/485** |
| Long Context (2K) | 2,048 | 0.94474 | **0.46443** | **-50.8%** | 67.48% | **79.00%** | +0.73710 | **+0.28152** | 442/1720 → **252/1720** |

#### MTP Layer 48 (Speculative Decoding Head)

| Suite | Positions | `v0.4.1` Top-32 KL | **`E8P-HQ v3` Top-32 KL** | **KL Change** | `v0.4.1` Top-1 | **`E8P-HQ v3` Top-1** | `v0.4.1` ΔNLL | **`E8P-HQ v3` ΔNLL** | High-Conf Flips (`p≥0.5`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Orinfer 8 Scenes | 56 | 0.30719 | **0.03376** | **-89.0%** | 85.71% | **96.43%** | -0.07857 | **-0.08547** | 8/56 → **2/56** |
| Held-Out EN Prose | 1,338 | 2.43440 | **0.75080** | **-69.2%** | 40.66% | **70.55%** | +1.86442 | **+0.21571** | 572/1060 → **225/1060** |
| Held-Out ZH Prose | 892 | 2.51259 | **0.69257** | **-72.4%** | 33.86% | **66.59%** | +2.04705 | **+0.26575** | 388/637 → **135/637** |
| Held-Out Agent/Code | 163 | 1.46996 | **0.16924** | **-88.5%** | 50.92% | **81.60%** | +1.07900 | **+0.02948** | 55/122 → **8/122** |
| Held-Out `<think>` Math | 493 | 0.62609 | **0.10544** | **-83.2%** | 80.12% | **92.49%** | +0.34713 | **+0.06816** | 81/467 → **26/467** |
| Long Context (2K) | 2,046 | 2.32441 | **0.65155** | **-72.0%** | 43.40% | **72.09%** | +1.95898 | **+0.33082** | 832/1653 → **329/1653** |

### Historical `v0.4.1` Baseline Acceptance (`2026-11-23`)

- **vs. Official CPU BF16 (64 positions)**: `mean_target_nll_delta = -0.0322` (vs. v0.4.0 `-0.1148`), `max_abs_target_logprob_delta = 2.187`, `0 / 60` high-confidence top-1 changes.
- **vs. Frozen v0.4.0 512 / 2K / 8K long context (176 positions)**: `mean_target_nll_delta = +0.0197`, `0 / 160` high-confidence top-1 changes.
- **GPU contracts**: `validate_sampled_lookup_requests`, `validate_layered_prefill` (`kl = 0.0`, `state_equal = true` across all 4 interruption regimes), and `validate_flash_prefix_cache` (503 / 2033 / 8187 tokens exact state and token match) all passed.

## License

Model weights retain [Qwen Community License 1.0](LICENSE). Orinfer software and its separate execution package use LGPL-3.0-or-later; source and third-party notices ship with the release.
