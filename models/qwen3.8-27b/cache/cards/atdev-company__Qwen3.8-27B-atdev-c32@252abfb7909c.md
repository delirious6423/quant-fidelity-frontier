---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - RedHatAI/Qwen3.8-27B-INT4
base_model_relation: quantized
library_name: vllm
datasets:
  - HuggingFaceH4/ultrachat_200k
  - openai/gsm8k
  - HuggingFaceFW/fineweb-2
  - HuggingFaceFW/fineweb-edu
tags:
  - qwen3.8
  - 4-bit
  - gptq
  - hadamard
  - w4a4
  - atdev-serve
---

# Qwen3.8-27B atdev C32 — Hadamard W4A4 sidecar

**c=32: 32 concurrent clients (32 requests in flight).** C32 refers to client concurrency; its speed figures report aggregate output throughput across those clients.

> **Easy setup on vLLM — let your AI coding agent handle it.**
> atdev-serve installs alongside unmodified **vLLM 0.31.0** and serves the familiar OpenAI-compatible API.
> **[Open the AI installation & setup guide](https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md)** — give this link to your AI coding agent and ask it to install, launch your chosen C1/C32 profile, and verify the server on your GPU machine.
> You can also follow the guide yourself. **Requires a supported NVIDIA GPU and CUDA environment**; the published profiles target an A100 80GB.


Model-side files for the **C32 Hadamard W4A4** profile of [atdev-serve](https://github.com/atdev-company/atdev-serve), on one A100 80GB (sm80), vLLM 0.31.0. C32 means 32 concurrent clients. This repo is **not a standalone model**: use it with the RedHatAI INT4 base and the matching atdev Hadamard activation kernels.

**No external drafter is added.** C32 retains the target model's built-in MTP head. This release provides only the selected Hadamard W4A4 path, not unrotated W4A4 or down-projection W4A8 restoration. The unchanged **up to 14.2x** maximum-speed headline on the project belongs to the separate **C1 fast W3** profile; it does not apply to this C32 sidecar.

## Files and integrity

These are the sidecar files for **c=32: 32 concurrent clients (32 requests in flight)**, using the Hadamard W4A4 profile.

| File | Bytes | Contents |
|---|---:|---|
| `hadamard/qwen38_27b_hadamard_gptq_d10.safetensors` | 12,543,701,944 | 400 INT4 projections / 800 packed-weight and BF16-scale tensors, group128, Hadamard128 block-diagonal K rotation, GPTQ act-order damp0.1 |
| `SHA256SUMS.txt` | — | Weight checksum |
| `accuracy_results.json` | — | Released-profile measured scores and limitations |
| `DATA_AND_LICENSES.md` | — | Exact separate C1/C32 calibration provenance and dataset attributions |

Weight SHA256: `608aef5bf2569e183845cd61ecb783385f61b491603912002d753edcca14728b`.

The source is the BF16 Qwen model in the AWQ-smoothed space needed by the RedHatAI checkpoint. The artifact is a matching rotated INT4 overlay, not another full 27B checkpoint. Embeddings, norms, heads and other unchanged parameters are read from the INT4 base.

## Measured results

**c=32: 32 concurrent clients (32 requests in flight).** Throughput below is aggregate output tok/s across all clients, not single-user per-request decode speed.

One saved 96-prompt run at concurrency32 measured **2,688.128833 aggregate output tok/s**: **1.53x** the historical best-vLLM 1,757.1 tok/s and about **4.0x** the historical bf16-default 671.8 tok/s. This is a single run, not a repeated mean or a simultaneous remeasurement of every baseline. C32 aggregate throughput is not C1 per-request decode speed. The old unrotated C32 result of 2,730.9 tok/s is not this shipped profile.

| Metric | Stock INT4 reference | c32_hadamard_w4a4 |
|---|---:|---:|
| MMLU-Pro | 71.71% | 72.79% |
| KMMLU Korean | 66.84% | 65.07% |
| IFEval prompt strict | 82.90% | 82.44% |
| IFEval prompt loose | 86.60% | 86.51% |
| IFEval instruction strict | 87.89% | 88.01% |
| IFEval instruction loose | 90.89% | 91.01% |
| HumanEval | 95.73% | 95.12% |
| HumanEval+ | 90.85% | 89.63% |
| MBPP | 91.27% | 92.06% |
| MBPP+ | 76.46% | 78.04% |
| WikiText-2 PPL (lower is better) | 7.298721 | 7.209287 |

GSM8K full1319: **1249/1319 (94.69%)**; stock reference 1248/1319. Long-context: **36/36**. No separate GSM300 result is claimed. Broad reference accuracy is the mean of two saved stock INT4 runs. Final selected KMMLU Korean is the repeated **65.07%** result (initial66.04%); it remains about1.78 percentage points below the reference mean. These are measured task scores, not statistical equivalence or universal lossless accuracy. See [full profile comparison](https://github.com/atdev-company/atdev-serve/blob/main/docs/ACCURACY.md).

## Context length

**Up to 262,144 total context tokens (input + output)** are supported by the C1 Fast W3 and C32 Hadamard W4A4 profiles. Each was tested on a **single A100 80GB** with **261,583 input tokens + 21 output tokens = 261,604 total tokens**, correctly retrieving all three codes placed at the beginning, middle and end of a synthetic long input.

Pass `--max-model-len 262144` to the selected launch script. The default remains 8,192 total context tokens; the published speed benchmarks use an output cap of 1,024 tokens. This record is a single long-request functional test for each profile.

## Usage

**c=32: 32 concurrent clients (32 requests in flight).** The command below launches the C32 serving preset; the client or benchmark must issue concurrent requests to reproduce this workload.

```bash
git clone https://github.com/atdev-company/atdev-serve
cd atdev-serve
./install.sh && source atdev-env.sh
./serve_c32.sh
# Docker: docker compose --profile c32 up --build
```

The script selects Hadamard W4A4 and automatically resolves `atdev-company/Qwen3.8-27B-atdev-c32:hadamard/qwen38_27b_hadamard_gptq_d10.safetensors`. Override with a local file using `HADW=/models/atdev-c32/hadamard/qwen38_27b_hadamard_gptq_d10.safetensors ./serve_c32.sh`. It never needs the C1 W3 overlay or C1 DFlash drafter. Use `MODEL=<local INT4 directory>` and `HF_HUB_OFFLINE=1` with pre-cached files when offline.

The sidecar and matching Hadamard activation transform are a pair. Do not use these rotated weights with the old unrotated kernels or plain stock vLLM; do not silently substitute W4A8 after a load error. [AI setup manual](https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md).

## Calibration data and licenses

This is offline GPTQ calibration, not new language-model fine-tuning. The 1,024-row capture pool submitted512 target-generated clean rows from UltraChat/GSM8K train sources plus author wrappers,256 [FineWeb-2 Korean](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) rows, and256 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) rows. Prefixes were capped at1,024tokens (780,107 tokens submitted), seed0, with at most16,384 sampled activation rows per linear layer. Sampled row IDs were not saved; not every submitted row is guaranteed to contribute to every projection. This is a different pool from the original C1 W3 clean400 recipe.

The UltraChat/GSM8K prompt sources are MIT. FineWeb-2 and FineWeb-Edu are ODC-By1.0 datasets with their upstream attributions and Common Crawl/source-content terms. The model derivatives and code are Apache-2.0, separately from those dataset terms. Raw calibration text and benchmark labels are not redistributed here. Read [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md) for source links, exact recipe and limitations. No research-only recovery candidate is included.
