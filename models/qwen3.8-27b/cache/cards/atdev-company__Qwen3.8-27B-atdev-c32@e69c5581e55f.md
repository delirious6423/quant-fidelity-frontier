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

# Qwen3.8-27B atdev C32: Hadamard W4A4 sidecar for the c32m and c32s profiles: up to 4.0x single-turn, up to 5.4x multi-turn

> **Easy setup on vLLM: let your AI coding agent handle it.**
> atdev-serve installs alongside unmodified **vLLM 0.31.0** and serves the familiar OpenAI-compatible API.
> **[Open the AI installation & setup guide](https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md)**: give this link to your AI coding agent and ask it to install, launch your chosen profile, and verify the server on your GPU machine.
> You can also follow the guide yourself. **Requires a supported NVIDIA GPU and CUDA environment**; the published profiles target an A100 80GB.

**Four profiles, each tuned for its use case.** Run the build made for your workload: **c32m** for a team assistant or agent fleet, **c32s** for batch jobs or an API of independent requests.

**Naming.** `c` = concurrent clients, the number is how many the profile is tuned for; the letter is the workload: `m` = multi-turn (growing conversations, prefix cache), `s` = single-turn (independent requests).
**`c32m`** = clients 32, multi-turn · **`c32s`** = clients 32, single-turn

This repo serves **c32m** (32 clients, multi-turn) and **c32s** (32 clients, single-turn); its sibling [atdev-company/Qwen3.8-27B-atdev-c1](https://huggingface.co/atdev-company/Qwen3.8-27B-atdev-c1) serves c1m and c1s.
Launch with `./serve.sh c32m` or `./serve.sh c32s` ([atdev-serve](https://github.com/atdev-company/atdev-serve)).

- **5.44x faster than the original bf16 model on stock vLLM, multi-turn (c32m):** 32 sessions x 7 turns (2k to 10.5k tokens) reach **386.3 vs 71.0 aggregate output tokens/s** at `max_tokens` 128 (4.9x at 512, 3.7x at 2048; wall time 73.4 s vs 399.8 s at 128). Condition: bf16 is a single run whose smaller KV cache (185,586 tokens) gave a 0% prefix-cache hit rate, which is part of why the gap is large; c32m is the median of 3 cold runs; full stack (this repo's rotated weights).
- **1.58x faster for 32 concurrent multi-turn sessions than the fastest stock setting (c32m):** **857.3 vs 543.5 aggregate output tokens/s** at `max_tokens` 512 (1.57x at 128, 1.52x at 2048; wall time 117.6 s vs 184.3 s at 512). Same trace and GPU, median of 3 cold runs, full stack (this repo's rotated weights).
- **Less energy per token:** c32m used 76.5 kJ against 121.4 kJ for stock vLLM at `max_tokens` 2048 (0.35 vs 0.54 J per output token).
- **c32s, single-turn:** **2,688.1 aggregate output tok/s** at 32 concurrent requests (96 prompts, up to 1,024 output tokens, single measured run): **1.53x** the best vLLM settings and about **4.0x** bf16 on stock vLLM.
- Accuracy class of stock INT4 on the broad benchmarks (see the tables below).

## Install

Manual (Linux, one A100 80GB, CUDA 13.0 toolkit with `nvcc`, Hugging Face access):

```bash
git clone https://github.com/atdev-company/atdev-serve && cd atdev-serve
./install.sh && source atdev-env.sh
./serve.sh c32m                           # or c32s
```

Or paste a prompt into your AI coding agent on the GPU machine:

*c32m*

```text
Install atdev-serve on this machine and launch the c32m profile (32 clients, multi-turn). Follow https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md and use STACK=c32m.
Verify that the server answers a chat request, then tell me the endpoint URL and any problems you ran into.
```

*c32s*

```text
Install atdev-serve on this machine and launch the c32s profile (32 clients, single-turn). Follow https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md and use STACK=c32s.
Verify that the server answers a chat request, then tell me the endpoint URL and any problems you ran into.
```

Everything else is in [AI_SETUP.md](https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md).

![c32m: aggregate output tokens per second over the whole trace at max_tokens 2048: bf16 stock vLLM 293.5, fastest stock INT4 719.2, atdev-serve c32m 1096.1 (higher is better)](assets/speed_c32m.png)

*Chart: output tokens per second over the whole run, prefill included; multiples are tok/s ratios (c32m 1.52x vs the fastest stock setting and 3.7x vs bf16 at `max_tokens` 2048); each arm emitted slightly different numbers of tokens, so the wall-time ratios in the table below differ a little.*

![c32s: aggregate output tok/s: bf16 default 671.8, INT4 default 1264.7, best vLLM 1757.1, atdev engine only 2135.7, atdev c32s full stack 2688.1](assets/speed_c32s.png)

Model-side files for the **Hadamard W4A4** profiles of [atdev-serve](https://github.com/atdev-company/atdev-serve), on one A100 80GB (sm80), vLLM 0.31.0. This repo is **not a standalone model**: it holds one rotated INT4 sidecar that atdev-serve loads next to the INT4 target.

**No external drafter is added.** c32m and c32s retain the target model's built-in MTP head. This release provides only the selected Hadamard W4A4 path, not unrotated W4A4 or down-projection W4A8 restoration. The **up to 14.2x** single-user headline belongs to c1s (the sibling repo), not to this repo.

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

## c32s: measured results (single-turn)

**c32s: 32 concurrent clients (32 requests in flight).** Throughput below is aggregate output tok/s across all clients, not single-user per-request decode speed.

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

## c32m: multi-turn profile (the default of `./serve_c32.sh`)

`./serve.sh c32m` is the Hadamard W4A4 stack of this repo with prefix caching for 32 concurrent multi-turn sessions, chunked-prefill kernels (FlashAttention-2 / FlashInfer, a rotation-aware 4-bit prefill GEMM), a 16,384-token context and a start-up warmup. The model files are unchanged (the same sidecar). `./serve.sh c32s` is the previous launch.

Aggregate output tokens per second over the whole trace (wall-clock seconds and time ratios alongside), cold start, median of 3 runs, same A100 (32 sessions x 7 turns, 2k to 10.5k tokens; stock = fastest stock vLLM setting found: INT4, int8 activations, MTP k=3, prefix caching on):

| `max_tokens` | c32m tok/s | stock tok/s | c32m vs stock | bf16 tok/s | c32m vs bf16 | wall time: c32m / stock / bf16 | time ratio vs stock / bf16 |
|---|---:|---:|---:|---:|---:|---|---|
| 128 | **386.3** | 245.5 | **1.57x** | 71.0 | **5.44x** | 73.4 s / 115.1 s / 399.8 s | 1.57x / 5.45x |
| 512 | **857.3** | 543.5 | **1.58x** | 175.6 | **4.88x** | 117.6 s / 184.3 s / 569.7 s | 1.57x / 4.84x |
| 2048 | **1,096.1** | 719.2 | **1.52x** | 293.5 | **3.73x** | 199.1 s / 314.0 s / 771.8 s | 1.58x / 3.88x |

The bf16 column is the original Qwen3.8-27B bf16 model on stock vLLM 0.31.0 (MTP k=3, prefix caching on): **a single run** on the same GPU as the other arms. Its weights leave a KV cache of only 185,586 tokens, smaller than the working set of 32 sessions, so its prefix cache thrashed: **0% hit rate and 6 / 88 / 126 preemptions** at caps 128 / 512 / 2048. That is part of why the gap to bf16 is large (largest at `max_tokens` 128: 5.44x); against the fastest stock INT4 setting the multiples are 1.57x / 1.58x / 1.52x. The speed multiples are ratios of output tokens per second over the whole run (prefill included); the time ratios can differ because the runs emit different numbers of tokens (c32m at `max_tokens` 2048: 218,240 tokens for c32m, 225,850 for stock, 226,530 for bf16).

The tok/s ratio is flat across output lengths (1.57x / 1.58x / 1.52x). With 32 sessions in flight, prefill and decode share the same GPU steps, so there is no clean split, and we have not measured why it stays flat (unlike c1m, whose tok/s ratio grows from 2.37x to 3.40x). Source: the 3-repetition run on trace v5-c32 (medians; at `max_tokens` 2048 mean 197.2 s).
Energy (NVML): 32.6 / 48.6 / 76.5 kJ against 50.7 / 75.7 / 121.4 kJ for stock (0.64x / 0.64x / 0.63x); average power 387-444 W, within 1-2% of stock.

**Accuracy of c32m** (research runtime = the frozen bundle; stock = one stock INT4 run of the same harness): GSM8K-1319 94.39% (stock 94.31%), long-context 36/36, MMLU-Pro 71.4% (stock 71.5%), KMMLU 67.6% (67.1%), HumanEval+ 89.0% (90.2%), MBPP+ 75.9% (76.7%), WikiText-2 PPL 7.365 (7.299), IFEval prompt strict 81.4% (mean of 3 runs 80.2 / 82.3 / 81.7; stock 83.4%, **2.0 points lower**, run-to-run spread up to about 4 points).
**Provenance:** the broad scores were run on our research runtime; the release tree reproduced speed (73.5 / 118.0 s at `max_tokens` 128 / 512), perplexity 7.365 and GSM8K-300 0.963; the single-turn c32s launch reproduced perplexity 7.209 and GSM8K-300 0.957.
c32m is a full-stack result (rotated W4A4 weights); no engine-only multi-turn measurement exists. An optional end-of-reply checkpoint (`ATDEV_EOR_CKPT=1`, off by default) speeds up live conversations (measured on an earlier bundle: 3-12%) and costs 1.3-2.7% where replies cannot be reused.
Details: [docs/MULTITURN.md](https://github.com/atdev-company/atdev-serve/blob/main/docs/MULTITURN.md) and [docs/TECHNIQUES.md](https://github.com/atdev-company/atdev-serve/blob/main/docs/TECHNIQUES.md).

## Context length

**Up to 262,144 total context tokens (input + output)** are supported by c1s and c32s. Each was tested on a **single A100 80GB** with **261,583 input tokens + 21 output tokens = 261,604 total tokens**, correctly retrieving all three codes placed at the beginning, middle and end of a synthetic long input.

Pass `--max-model-len 262144` to the launcher (`./serve.sh c32s --max-model-len 262144`; c32m defaults to 16,384 and was not tested beyond). The default of c32s remains 8,192 total context tokens; the published speed benchmarks use an output cap of 1,024 tokens. This record is a single long-request functional test for each profile.

## Usage

**c=32: 32 concurrent clients (32 requests in flight).** The command below launches the C32 serving preset; the client or benchmark must issue concurrent requests to reproduce this workload.

```bash
git clone https://github.com/atdev-company/atdev-serve
cd atdev-serve
./install.sh && source atdev-env.sh
./serve.sh c32m                           # multi-turn (default); ./serve.sh c32s for the single-turn launch
# Docker: docker compose --profile c32 up --build   (c32m)   |   --profile c32-single (c32s)
```

The script selects Hadamard W4A4 and automatically resolves `atdev-company/Qwen3.8-27B-atdev-c32:hadamard/qwen38_27b_hadamard_gptq_d10.safetensors`. Override with a local file using `HADW=/models/atdev-c32/hadamard/qwen38_27b_hadamard_gptq_d10.safetensors ./serve_c32.sh`. It never needs the C1 W3 overlay or C1 DFlash drafter. Use `MODEL=<local INT4 directory>` and `HF_HUB_OFFLINE=1` with pre-cached files when offline.

The sidecar and matching Hadamard activation transform are a pair. Do not use these rotated weights with the old unrotated kernels or plain stock vLLM; do not silently substitute W4A8 after a load error. [AI setup manual](https://github.com/atdev-company/atdev-serve/blob/main/AI_SETUP.md).

## Calibration data and licenses

This is offline GPTQ calibration, not new language-model fine-tuning. The 1,024-row capture pool submitted512 target-generated clean rows from UltraChat/GSM8K train sources plus author wrappers,256 [FineWeb-2 Korean](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) rows, and256 [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) rows. Prefixes were capped at1,024tokens (780,107 tokens submitted), seed0, with at most16,384 sampled activation rows per linear layer. Sampled row IDs were not saved; not every submitted row is guaranteed to contribute to every projection. This is a different pool from the original C1 W3 clean400 recipe.

The UltraChat/GSM8K prompt sources are MIT. FineWeb-2 and FineWeb-Edu are ODC-By1.0 datasets with their upstream attributions and Common Crawl/source-content terms. The model derivatives and code are Apache-2.0, separately from those dataset terms. Raw calibration text and benchmark labels are not redistributed here. Read [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md) for source links, exact recipe and limitations. No research-only recovery candidate is included.
