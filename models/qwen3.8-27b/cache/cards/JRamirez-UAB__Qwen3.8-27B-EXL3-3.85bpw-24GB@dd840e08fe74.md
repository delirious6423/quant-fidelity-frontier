---
language:
- es
- en
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- sysadmin
- opencode
- linux
- vllm
- exl3
- exllamav3
- qtip
- qwen3.8
- mtp
- speculative-decoding
- long-context
- vision
- rtx3090
- ampere
- endpoints_compatible
---

# Qwen3.8-27B — EXL3 3.85 bpw, for 24 GB GPUs (vision + MTP, served with vLLM)

> **Install from this card, not from generic snippets.** This is an EXL3 checkpoint: vLLM loads it only together with exllamav3 and our plugins. Follow [Requirements & install](#requirements--install-vllm-031--exllamav3-151--vllm-quantpack) and [Quick start](#quick-start-single-rtx-3090-24-gb); a plain `pip install vllm`, Transformers or SGLang cannot load it.

An EXL3 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) made with [exllamav3](https://github.com/turboderp-org/exllamav3)'s
per-tensor bit-allocation ("SC") pipeline with our own calibration, a 5-bit `lm_head`, a 6-bit vision tower, a 4-bit MTP head and INT8 token embeddings.
It is served with **vLLM** (continuous batching) through our `vllm-quantpack` plugin and runs a **215k-token context with vision and MTP speculative decoding
on a single 24 GB RTX 3090**, with a **KL divergence of 0.0073 against BF16** (≈5× lower than our W4A16 GPTQ version, and lower than turboderp's
4.00 bpw at a smaller size). The `3.70bpw` branch trades a little fidelity (KLD 0.0089) for a 226k context.

> **Tested setup:** RTX 3090 (24 GB, Ampere, power limit 350 W), vLLM 0.31.0 + exllamav3 1.5.1 + the OrcaSAQ2 plugin + our `qwen38_exl3_vision` 0.3.1 plugin, weights downloaded from this repo, the Quick start configuration (2026-10-14). Only this setup has been tested.

**Highlights (one RTX 3090 24 GB, measured on 2026-10-14; details in *Family comparison*):**
- **KLD 0.0073 against BF16** on our evaluation set (0.0067 on turboderp's public trace).
- **215,000 with vision** on one 24 GB GPU with MTP (3.03 tokens per step); needle 8/8 at 197,991.
- **Tool calls with the real opencode request 60/60**, opencode agent tasks **29/30** (10 sysadmin tasks × 3 runs), images **15/15**.
- **Reasoning 48/48 at `xhigh`** with 22,474 reasoning tokens in total.
- Decode 87.9 / 137.4 / 149.5 tok/s with 1 / 2 / 8 requests; 300 MiB of VRAM really free at peak.

> **Our focus.** We quantize and evaluate models for one use case: a **local Linux system-administration agent** (opencode with tool calling)
> for a university IT department, **used in Spanish**. Our calibration traces include real sysadmin/agent conversations in Spanish, and our quality
> checks (tool calls with the real opencode request, agent tasks with real command execution, long-context retrieval, long answers in Spanish) reflect
> that use. KL divergence is also reported on turboderp's public trace, so it can be compared and reproduced. Numbers for other domains or languages may differ.

## What is in this repository

### Which one to use

All Qwen3.8-27B variants we publish, measured on 2026-10-14 under the same conditions (see *Family comparison* for every number):

| Variant | Weights | Context on one RTX 3090 | KLD vs BF16 | Agent tasks | Decode, 1 request | Pick it for |
|---|---|---|---|---|---|---|
| [GPTQ INT4](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB/tree/mtp-int8) | 14.21 GiB | 222,000 (text) · 176,128 with vision | 0.0400 | 28/30 | 104.0 tok/s | the longest context on stock vLLM (+ a 4-line patch): 222,000 text-only, or 176,128 with vision |
| [GPTQ INT8 g64](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB) | 15.75 GiB | 176,000 (text) · 131,072 with vision | 0.0175 | 29/30 | 101.4 tok/s | a more faithful GPTQ (KLD 0.0175) on stock vLLM (+ the same patch), with less context |
| **[EXL3 3.85](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB) (this repo)** | 13.62 GiB | 215,000 with vision | 0.0073 | 29/30 | 87.9 tok/s | the closest to BF16 (KLD 0.0073) with vision at 215,000; needs exllamav3 + our plugin |
| [EXL3 3.70](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB/tree/3.70bpw) (this repo) | 13.19 GiB | 226,000 with vision | 0.0089 | 29/30 | 79.3 tok/s | more context (226,000) and 4 sessions with vision, a little less faithful than 3.85 |

### Variants / revisions

| Revision | Linear layers (achieved) | `lm_head` | Vision | MTP head | Embeddings | Size | KLD (ours / pub.) |
|---|---|---|---|---|---|---|---|
| **`main`** | **3.85 bpw (3.866)**, SC allocation | 5 bits | 6 bits | 4 bits | INT8 per row | **13.62 GiB** | **0.0073 / 0.0067** |
| `3.70bpw` | 3.70 bpw (3.716), SC allocation | 5 bits | 6 bits | 4 bits | INT8 per row | 13.19 GiB | 0.0089 / 0.0085 |

Unlike our Swift EXL3 quantizations, these were made **without YAQA** (on Swift, YAQA changed the KLD by 0–3 %).

### Files and reproducibility

- `model-*.safetensors`, `config.json` (`quantization_config`: `exl3`, 3.85 bits, head 5, vision 6, MTP 4), `quantization_config.json` (per-tensor allocation).
- `plugin/qwen38_exl3_vision/`: our vLLM plugin (vision tensors and padding). `scripts/embed_int8.py`: re-encodes `embed_tokens` to INT8 per row.
- The calibration and evaluation sets are internal and not published; the public KLD figures use turboderp's `qbench_prompts_gen.json`, so they can be reproduced
  with exllamav3's `qbench` against the BF16 model.

## Results
### Family comparison (Qwen3.8-27B, same conditions)

Measured on 2026-10-14 for every variant in the same way: weights downloaded from Hugging Face, one RTX 3090 24 GB (power limit 350 W, no other use), vLLM 0.31.0 (GPTQ: + the embedding patch / EXL3: + exllamav3 1.5.1 + plugins), each variant with its own Quick start configuration. KLD and perplexity come from exllamav3's `qbench` (see *How to read it*).

| Variant | Weights | KLD mean / median (internal · public) | Perplexity (BF16 1.5860 · pub. 1.3332) | Needle (8 facts) near max. context | Tools: real opencode request | Agent tasks (10 × 3) | Reasoning `xhigh`: correct · tokens | Vision |
|---|---|---|---|---|---|---|---|---|
| [GPTQ W4A16 + INT4 embeddings](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB/tree/mtp-int8) (`mtp-int8`) | 14.21 GiB | 0.0400 / 0.0127 · pub. 0.0312 / 0.0017 | 1.6631 · pub. 1.3779 | 8/8 at 204,551 | 56/60 | 28/30 | 47/48 · 32,190 | 15/15 |
| [GPTQ W4A16 g64 + INT8 embeddings](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB) (`main`) | 15.75 GiB | 0.0175 / 0.0049 · pub. 0.0171 / 0.0009 | 1.6203 · pub. 1.3568 | 8/8 at 162,208 | 59/60 | 29/30 | 48/48 · 20,414 | 15/15 |
| **[EXL3 3.85 bpw](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB) (`main`, this repo)** | **13.62 GiB** | **0.0073 / 0.0018 · pub. 0.0067 / 0.0003** | **1.5976 · pub. 1.3430** | **8/8 at 197,991** | **60/60** | **29/30** | **48/48 · 22,474** | **15/15** |
| [EXL3 3.70 bpw](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB/tree/3.70bpw) (`3.70bpw`, this repo) | 13.19 GiB | 0.0089 / 0.0023 · pub. 0.0085 / 0.0004 | 1.5999 · pub. 1.3466 | 8/8 at 208,158 | 60/60 | 29/30 | 48/48 · 20,873 | 15/15 |

| Variant | Context on one RTX 3090 | Sessions | Prefill near max. context | Decode, 1 / 2 / 8 requests (tok/s) | MTP tokens per step | VRAM really free at peak |
|---|---|---|---|---|---|---|
| GPTQ INT4 | 222,000 (text) · 176,128 with vision | 2 | 605 tok/s | 104.0 / 182.4 / 192.4 | 2.95 | 282 MiB · vision config: 930 MiB |
| GPTQ INT8 g64 | 176,000 (text) · 131,072 with vision | 2 | 680 tok/s | 101.4 / 164.6 / 170.2 | 3.07 | 196 MiB · vision config: 840 MiB |
| EXL3 3.85 | 215,000 with vision | 2 | 556 tok/s | 87.9 / 137.4 / 149.5 | 3.03 | 300 MiB |
| EXL3 3.70 | 226,000 with vision | 4 | 545 tok/s | 79.3 / 142.0 / 264.4 | 3.03 | 148 MiB |

How to read it:
- **KLD / perplexity**: next-token KL divergence against the BF16 model with exllamav3's `qbench` (lower is better). *Internal*: our sysadmin/agent evaluation set (not published, separate from the calibration data); *public*: turboderp's `qbench_prompts_gen.json`, reproducible by anyone. GPTQ checkpoints were evaluated as a BF16 copy holding the dequantized weights. These two columns come from `qbench`, not from this campaign.
- **Needle**: 8 facts spread through one prompt at about 92 % of the served context; all 8 must come back. **Tools**: the real request of the opencode agent (system prompt + 11 tools), 3 cases × 20 tries, at the model's default sampling (`generation_config.json`); a try counts when it calls a valid tool with valid arguments and the expected command. **Agent**: 10 sysadmin tasks run with `opencode run` and real command execution, 3 runs, at the recommended temperature; with sampling on, the same checkpoint varies by about ±2 of 30 between runs, so other measurements in this card may differ by that much. **Reasoning**: 6 checkable problems × 8 at `xhigh`; total reasoning tokens over the 48. **Vision**: our 4-image test.
- **Prefill / decode**: prefill from the needle prompt; decode is aggregate throughput with 1 / 2 / 8 simultaneous requests (temperature 0; with more requests than *Sessions*, the extra ones wait). **MTP tokens per step**: vLLM's speculative-decoding metrics over the whole run.
- **VRAM really free**: lowest `memory.free` while serving (nvidia-smi every 0.5 s, from the end of start-up to the end of the run); we require ≥100 MiB.
- Older sections below keep their own date and setup; use this table to compare variants.

**How many sessions fit** with each Quick start configuration (KV size in tokens as reported by vLLM at start-up; a session uses the KV of its own context, and vLLM never runs more than *max sessions* at once):

| Variant | KV pool (tokens) | Max sessions | 32k each | 64k each | 128k each | Full context each |
|---|---|---|---|---|---|---|
| GPTQ INT4 | 223,441 | 2 | 2 | 2 | 1 | 1 × 222,000 |
| GPTQ INT4 (vision config) | 178,923 | 2 | 2 | 2 | 1 | 1 × 176,128 |
| GPTQ INT8 g64 | 178,816 | 2 | 2 | 2 | 1 | 1 × 176,000 |
| GPTQ INT8 g64 (vision config) | 132,423 | 2 | 2 | 2 | 1 | 1 × 131,072 |
| EXL3 3.85 | 216,433 | 2 | 2 | 2 | 1 | 1 × 215,000 |
| EXL3 3.70 | 231,757 | 4 | 4 | 3 | 1 | 1 × 226,000 |

**Few users or many users.** The Quick start configuration is tuned for 1–4 simultaneous users (fastest single request). For more users, the *many users* option below (measured the same day, same GPU) gives more total throughput; each request gets slower when many run at once. Same model, same quality: only the serving settings change.

| Variant | Configuration | Decode by simultaneous requests (aggregate tok/s) | N sessions at once, needle 8/8 | Tools (5 × 3) | VRAM really free |
|---|---|---|---|---|---|
| GPTQ INT4 | Quick start (2 sessions, MTP 3) | 1: 104 · 2: 182 · 8: 192 | — | 56/60 | 282 MiB |
| GPTQ INT4 | many users (8 sessions of up to 222,000 tokens, MTP 3) | 1: 105 · 2: 185 · 4: 341 · 8: 475 | 8/8 of 22k | 15/15 | 176 MiB |
| GPTQ INT8 g64 | Quick start (2 sessions, MTP 3) | 1: 101 · 2: 165 · 8: 170 | — | 59/60 | 196 MiB |
| GPTQ INT8 g64 | many users (8 sessions of up to 165,000 tokens, MTP 3) | 1: 94 · 2: 177 · 4: 316 · 8: 470 | 8/8 of 22k | 14/15 | 386 MiB |
| EXL3 3.85 | Quick start (2 sessions, MTP 3) | 1: 88 · 2: 137 · 8: 150 | — | 60/60 | 300 MiB |
| EXL3 3.85 | many users (8 sessions of up to 215,000 tokens, MTP 3) | 1: 84 · 2: 140 · 4: 252 · 8: 296 | 8/8 of 22k | 15/15 | 138 MiB |
| EXL3 3.70 | Quick start (4 sessions, MTP 3) | 1: 79 · 2: 142 · 8: 264 | — | 60/60 | 148 MiB |
| EXL3 3.70 | many users (8 sessions of up to 220,000 tokens, MTP 3) | 1: 82 · 2: 141 · 4: 253 · 8: 298 | 8/8 of 22k | 15/15 | 338 MiB |

<!-- end of the controlled comparison -->

### Benchmarks and validated configurations (single RTX 3090 24 GB, 350 W)

_Earlier measurements, before the controlled comparison of 2026-10-14: other dates, vLLM versions or settings, as stated in each table. Use *Family comparison* above to compare variants._

#### Service configuration (quick start above)

| Test | `main` 3.85 bpw (215,000) | `3.70bpw` (226,000) |
|---|---|---|
| KV pool | 216,433 tokens (KV 8.42e9 B) | 227,439 tokens (KV 8.85e9 B) |
| Idle: used / really free | 23,406 / 719 MiB | 23,346 / 779 MiB |
| 8 facts in one long prompt (greedy) | **8/8** at 200,069 tokens, 371.2 s (539 tok/s) | **8/8** at 210,602 tokens, 400.4 s (526 tok/s) |
| 2 simultaneous sessions | 2 × ≈99k, 8/8 in both, 722 tok/s aggregate | 2 × ≈104k, 8/8 in both, 712 tok/s aggregate |
| Decode, 1 request, t=0 / 0.6 | 75.9 / 73.1 tok/s | 81.4 / 71.1 tok/s |
| Decode, 2 concurrent, t=0 / 0.6 | 135.8 / 131.7 tok/s | 139.2 / 130.0 tok/s |
| Images (4 synthetic tasks, 18 checks; 2026-10-02 battery) | **4/4 tasks, 18/18**; follow-up turn on the same image (prefix cache) 2.4 → 1.8 s | **4/4, 18/18**; 2.5 → 1.8 s |
| Real opencode request, tool calls (xhigh, 3 cases × 20; T=1.0 / T=0.3) | **59/60 / 60/60** (the miss: full multi-turn session at T=1.0, answered from the history) | **60/60 / 60/60** |
| Peak used / really free | 23,814 MiB / **311 MiB**, no OOM | 23,754 MiB / **371 MiB**, no OOM |

Images (EXL3 rows, vision enabled): **4/4 tasks, 18/18 checks** with our 2026-10-02 battery (failed-service screenshot, chart, dense firewall ruleset,
and a terminal where the port is held by another process while the journal shows a dead PID as a trap). This battery replaces the older 3-image one
(8/9), whose first image wrongly required that dead PID; the PID is still checked in a follow-up turn.

#### Reasoning and agent tasks

6 problems with checkable answers (arithmetic, word problem, counting, logic, subnetting, systemd) × 8 tries, temperature 0.6, total reasoning tokens over
48 requests; agent = 4 sysadmin tasks × 2 with `opencode run` (`xhigh`, temperature 0.3, our setting until 2026-10-13, real tool execution, reasoning counted by a proxy):

| Model | low | medium | xhigh | Correct (low · medium · xhigh) | Agent: ok · reasoning tokens · time |
|---|---|---|---|---|---|
| **EXL3 3.85 (`main`)** | 18,683 | 20,851 | **23,221** | 47 · 48 · 48 | 8/8 · 5,366 · 342 s |
| EXL3 3.70 (`3.70bpw`) | 17,879 | 20,068 | 21,912 | 48 · 48 · 48 | 8/8 · 3,681 · 327 s |
| W4A16 GPTQ (sibling repo, 2026-09-30) | 17,957 | 21,317 | 26,481 | 48 · 48 · 48 | 8/8 · 4,817 · 245 s |
| turboderp SC 3.00 bpw (reference, 2026-09-30) | 17,504 | 20,404 | 18,526 | 48 · 48 · 48 | not measured |

The EXL3 versions reason 12–17 % less at `xhigh` on these problems; on the short agent tasks all reason little and the differences (−24 % to +11 %) are within
the run-to-run spread. The level (`low` / `medium` / `xhigh`) does not regulate the amount of reasoning much in either. For a model that reasons markedly less, see
the Swift 1.5 family.

#### Agent turns and 8 sessions (`--max-num-seqs 8`, same KV pool)

| Revision | Turn: time to first token / total (median) | First turn (no cache) | 8 sessions: needle each, aggregate prefill | Decode with 8 sessions (t=0 / 0.6) | Really free (min) |
|---|---|---|---|---|---|
| `main` 3.85 | 3.2 / 4.3 s | 13.7 s | 8 × 23k, 8/8 in all, 972 tok/s | 289.6 / 281.1 tok/s | 149 MiB |

Agent turn = the real opencode request (≈13.5k → 14.7k tokens over 10 turns) with prefix caching. For comparison, the W4A16 sibling needs 2.7 / 3.6 s per turn
and reaches 472 tok/s with 8 sessions, but only up to a 200k pool and without vision.

#### MTP acceptance

On the mixed workload of the family table: 3.03 tokens per step for `main`, per-position acceptance 0.86 / 0.66 / 0.52. `3.70bpw` has no 8-session run yet;
on the validation run (needle, decode, images, tools) it gave 2.96 tokens per step, against 2.93 for `main` on the same kind of run.

## Sampling and repetition loops

Measured on 2026-10-14 with the same software and GPU as *Family comparison*, on our EXL3 3.85 build of Qwen3.8-27B (loops come from the model and the sampling, not from the quantization format): 5 long Spanish sysadmin prompts × 6 runs per temperature, up to 16,000 tokens, top_p 0.95 and top_k 20; a loop = more than 30 % repeated 8-grams. Serving: the Quick start with a 32k context and 8 sessions.

| Temperature | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| Loops / 30 | 12 | 12 | 7 | 8 | 1 | 1 | 1 | 0 |
| Answers that finished before 16,000 tokens / 30 | 6 | 4 | 5 | 5 | 4 | 4 | 2 | 0 |

From temperature **0.7** up, loops are rare (at most 1 of 30). This repo's default (`generation_config.json`) is **1.0**.

Other models, same test and setup (loops / 30):

| Model (build) | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B-pi · EXL3 3.85 | 9 | 4 | 9 | 10 | 4 | 2 | 1 | 0 |
| Qwopus3.8-27B-Flash-V2 · EXL3 3.85 | 3 | 1 | 2 | 1 | 2 | 0 | 0 | 0 |
| Swift 1.5 (Qwen3.8-27B) · EXL3 3.85 | 7 | 5 | 3 | 4 | 3 | 1 | 1 | 0 |
| Ornith-1.5-9B · EXL3 6.00 (16 GB) | 6 | 2 | 3 | 1 | 1 | 0 | 0 | 0 |
| NeoHorse-1-9B · EXL3 6.00 (16 GB) | 6 | 3 | 3 | 1 | 0 | 0 | 0 | 0 |
| Qwen3.6-35B-A3B · EXL3 4.00 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| Ornith-1.5-35B-A3B · EXL3 4.00 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwopus3.6-35B-A3B-Coder · EXL3 4.00 | 9 | 13 | 12 | 6 | 11 | 6 | 5 | 4 |
| Swift 1.5 Qwen3.8-Flash-Next (125B-A6B) · EXL3 2.25 · vLLM | 24 | 24 | 21 | 21 | 17 | 8 | 0 | 0 |
| Qwen3.8-Flash-Next (125B-A6B) · EXL3 2.25 · vLLM | — | — | 13 | 7 | 5 | 2 | 0 | 0 |

## Requirements & install (vLLM 0.31 + exllamav3 1.5.1 + vllm-quantpack)

**Before you install** (checked on a clean Ubuntu 24.04 host with an A40, 2026-10-28):
- An NVIDIA driver that supports CUDA 13.2 (tested with 595.84) and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- The Python 3.12 headers and a C compiler. `uv venv --python 3.12` uses the system Python 3.12 when there is one, and Triton
  compiles a small C helper every time vLLM starts; without the headers every `vllm serve` stops with
  `fatal error: Python.h: No such file or directory`. On Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y python3.12-dev build-essential`.
- No CUDA toolkit is needed: the FlashInfer line below installs prebuilt kernels (without it, vLLM looks for `nvcc` at start and fails with
  `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`).
- Disk: about 20 GB for the venv, as much again for uv's cache (`uv cache clean` frees it), plus the model.

EXL3 is not a native vLLM format. One plugin is loaded through vLLM's `vllm.general_plugins` entry point:
- **[vllm-quantpack](https://github.com/jramirez-uab/vllm-quantpack)** (ours, on PyPI): EXL3 linear, embedding and MoE layers through exllamav3's kernels, plus the vision tower and the MTP head.

Use **exllamav3 1.5.1** (with 1.6.x the KLD is the same, but tool calls without reasoning got worse in our tests) and Python **3.12** (the exllamav3 wheels
are cp312).

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python vllm==0.31.0 --torch-backend=auto        # torch 2.13.0
uv pip install --python venv/bin/python flashinfer-cubin==0.7.0.post1 flashinfer-jit-cache==0.7.0.post1+cu130 \
  --extra-index-url https://flashinfer.ai/whl/ --extra-index-url https://flashinfer.ai/whl/cu130/ --index-strategy unsafe-best-match   # FlashInfer kernels
W=exllamav3-1.5.1+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl
curl -L -o /tmp/$W "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.1/${W/+/%2B}"
uv pip install --python venv/bin/python /tmp/$W vllm-quantpack
venv/bin/python -c "import torch; from exllamav3.ext import exllamav3_ext; print(torch.__version__, torch.cuda.get_device_name(0))"
```
> **Plugin (2026-10-28):** [vllm-quantpack](https://github.com/jramirez-uab/vllm-quantpack) (`pip install vllm-quantpack`) replaces the OrcaSAQ2 + `qwen38_exl3_vision` plugins. The measurements in this card were made with the previous stack (OrcaSAQ2 + our plugins); this checkpoint was then served again with vllm-quantpack 0.1.0 following this card on a clean install (text, image and tool calls checked). The previous files stay in `plugin/` for reference.


## Quick start (single RTX 3090, 24 GB)

**`main` (3.85 bpw), vision + MTP 3, 215k context, 2 concurrent sequences, 1 image per request (≤1 MP)** — our validated configuration:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB \
  --served-model-name qwen3.8-27b \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1003520}' \
  --max-model-len 215000 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-memory-bytes 8420000000 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**`3.70bpw` branch, 226k context:** the same command with `--revision 3.70bpw --max-model-len 226000 --kv-cache-memory-bytes 9000000000 --max-num-seqs 4` (4 sessions; validated in *Family comparison*).

**8 concurrent sessions:** add `--max-num-seqs 8` and keep the same KV cache (the pool is shared: 8 × ≈23k tokens at once, or one long session).
Validated on `main` with 149 MiB really free at peak (not yet run on `3.70bpw`).

**For many users** (measured on 2026-10-14, same GPU): the same command with `--max-num-seqs 8` — 8 sessions of up to 215,000 tokens, MTP 3. Aggregate decode 296 tok/s with 8 requests (Quick start: 150 with 8); a single request runs at 84 tok/s (88 with the Quick start); 8 of 8 simultaneous sessions returned all 8 needle facts; 138 MiB of VRAM really free. Rule for CUDA graphs: they must cover sessions × (1 + MTP) tokens per step.

## Using it with opencode (how we tested)

This checkpoint is meant to be used as a **coding / sysadmin agent from [opencode](https://opencode.ai)** (tested with opencode 1.18.33). Tool calling was
measured with the real opencode request (its long system prompt and 11 tools, captured and replayed against vLLM) and with `opencode run` agent tasks;
context, speed, memory and image tests were run directly against the vLLM API.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local vLLM",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8-27B EXL3 (local)",
          "tool_call": true,
          "temperature": true,
          "attachment": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 215000, "output": 22000 },
          "variants": {
            "none":   { "chat_template_kwargs": { "enable_thinking": false } },
            "low":    { "chat_template_kwargs": { "reasoning_effort": "low" } },
            "medium": { "chat_template_kwargs": { "reasoning_effort": "medium" } },
            "xhigh":  { "chat_template_kwargs": { "reasoning_effort": "xhigh" } }
          }
        }
      }
    }
  },
  "agent": {
    "build": { "model": "local-vllm/qwen3.8-27b", "variant": "xhigh", "temperature": 1.0 },
    "plan":  { "model": "local-vllm/qwen3.8-27b", "variant": "xhigh", "temperature": 1.0 }
  }
}
```

- **Temperature 1.0** (since 2026-10-13; it was 0.3): our first agent tests of this family used 0.3, but at 0.3 the model falls into repetition loops on long answers (see *Sampling and repetition loops*), and in our agent tests (10 real sysadmin tasks × 3 runs with `opencode run`) 1.0 scored the same as 0.3 within one task: 29/30 at 1.0 and 30/30 at 0.3 for Qwen3.8-27B, -pi and Swift 1.5 27B; Qwopus3.8, served with its `chat_template-xml-tools.jinja`, 28/30 at 1.0 and 26/30 at 0.3.

- **`limit.context` = `--max-model-len`** (226000 for the `3.70bpw` branch); with `output: 22000` opencode compacts the history at ≈`context − 22k` tokens of input.
- **Declare `"temperature": true` in the model**, otherwise opencode sends no temperature to a custom model (checked by capturing the requests).
- Without `attachment` / `modalities` opencode will not send images; drop them if you serve with `--language-model-only`.
- Start opencode inside a project directory (it takes a git snapshot before each step); for `opencode run` from scripts, close stdin (`</dev/null`).

## Important notes

- **Memory.** The driver reserves 453 MiB of the 24,576 MiB that vLLM cannot use: read the real headroom with
  `nvidia-smi --query-gpu=memory.used,memory.free --format=csv`, not as 24,576 − used. Size the KV cache with `--kv-cache-memory-bytes`, never with
  `--gpu-memory-utilization` (the profiling of this hybrid GatedDeltaNet model under-estimates the activations). `--max-num-batched-tokens 1024` and
  `expandable_segments` lower the peak by ≈350 MiB at no measurable cost. With MTP, KV capacity grows in block-sized steps: check the `GPU KV cache size`
  line (8,420,000,000 B → 216,433 tokens; 8,850,000,000 B → 227,439) and keep `--max-model-len` below it.
- **Measured peaks** (2 sessions, vision): 3.85 bpw 23,814 MiB used, **311 MiB really free**; 3.70 bpw 23,754 MiB, 371 MiB free. Leave the GPU for the LLM: a desktop session on the same GPU would eat this margin.
- **Hugging Face page warnings:** the "Model size" shown by Hugging Face (≈8B params) counts the packed storage elements of the safetensors files
  (EXL3 packs ≈4 weights into each `int16`), not parameters: this is the 27B model at ≈3.85 bits per weight. The *Configuration Parsing Warning*
  "`quantization_config.bits` must be an integer" is also expected: for EXL3 `bits` is the average bits per weight of a per-tensor allocation
  (the per-tensor values are in `quantization_config.json`); exllamav3 and the OrcaSAQ2 plugin read it as a float.
- `VLLM_USE_FLASHINFER_SAMPLER=0` is recommended by the OrcaSAQ2 plugin (flashinfer's JIT sampler is fragile; torch's is equivalent).
- Use `--tool-call-parser qwen3_coder` (Qwen3.8 XML tool calls) and `--reasoning-parser qwen3`. `reasoning_effort` accepts only `xhigh`, `medium` and `low`
  (`high` returns HTTP 400); they are prompt instructions, not limits. For a hard cap, `thinking_token_budget` was validated with vLLM 0.29 on the W4A16 version
  (not re-checked with this stack).
- On Ampere (sm86) the fp8 KV cache works with the FlashInfer attention backend (the default).
- **Why it is slower than W4A16 on Ampere** (per-layer micro-benchmarks on the 3090): with 1 row the EXL3 kernel matches Marlin, but with the 4–8 rows of an
  MTP verification step decoding the trellis becomes compute-bound and costs ≈6 ms more per step; the plugin path (BF16↔FP16 conversions, fused layers run
  piecewise) adds ≈2.6 ms, and a higher-bit `lm_head` ≈1–2 ms. Ada / Blackwell GPUs should lose less. Prefill is ≈8 % slower.
- **Stability (our machine):** our RTX 3090 hung twice at its factory 370 W limit with very long prompts (once inside an exllamav3 kernel, once with an
  unrelated model); after setting a 350 W power limit and a more aggressive fan curve, none of the tests here (up to 245k-token prompts) hung again. If you
  see Xid errors in `dmesg`, try a lower power limit first.

## Quantization recipe

| Step | What we did |
|---|---|
| Calibration trace | Generated by **Qwen3.8-27B itself** (`reasoning_effort: xhigh`, MTP enabled) with exllamav3's `sc_trace`: 196 prompts: 157 from exllamav3's own `sc_trace` set (logic, maths, code, science; English and Spanish) and 39 of our own generic sysadmin/agent conversations (invented examples, no real system data; not published). The trace is the model's own `xhigh` answers to them, ≈535k generated tokens; 196 sequences × 2048 tokens used for conversion |
| Reference and sensitivity | `sc_rfn_probe` against a 2.00 bpw reference, `sc_measure --shaped --max-sys 24` (per-tensor sensitivity on the trace) |
| Bit allocation | `sc_optimize` at the target rate with a 5-bit `lm_head` (`-hb 5`, `-al 2.0`) → per-tensor recipe (achieved 3.866 / 3.716 bpw) |
| Conversion | exllamav3 1.5.1 `convert.py` with the recipe, the calibration trace, vision 6 bits (`-vb 6`), MTP 4 bits (`-mb 4`); codebook `mul1`, output scales always |
| Embeddings | `embed_tokens` re-encoded to **INT8 per row** (absmax / 127, BF16 scales) in the OrcaSAQ2 plugin's format (≈1.2 GiB less VRAM); script included |

- Tokenizer, chat template and generation config unchanged (XML tool calls, `qwen3_coder`).
- Hardware: one RTX PRO 6000 Blackwell / L40S per job; conversion peaked at ≈33 GiB of RAM, `sc_measure` at ≈57 GiB.
- The measured KLDs were ≈0.55× of `sc_optimize`'s prediction (3.85: predicted 0.0132, measured 0.0073; 3.70: 0.0162 → 0.0089).

## Integrity and validation

What was checked for this checkpoint, how, and where the numbers are (measured on 2026-10-14; the method and every figure are in *Family comparison*).

**Artifact integrity**
- Every weight file is checked against our local copy with SHA-256 before publication (`SHA256SUMS` in this repo).
- The measurements used the weights downloaded from this repo at commit `d517bc252aaf`; later commits change only metadata, documentation or the plugin, never the `.safetensors` files.
- vLLM loads every tensor listed in `model.safetensors.index.json` on start-up (a missing tensor or a wrong shape would stop it): checked from a fresh download.
- After the metadata update of 2026-10-14 (`quantization_config.json`, plugin 0.3.1, `embed_int8.py`): downloaded again from this repo and re-tested: loads; vision 15/15 needle 64k 8/8; tool calls 15/15; engine errors 0.

**Quantization**
- Per-tensor bit allocation is recorded by exllamav3 in `quantization_config.json`; its `tensor_storage` entry for `embed_tokens` describes the INT8 `qweight` + BF16 `scales` actually stored (updated on 2026-10-14; the plugin only reads the `.trellis` entries).
- The INT8 embedding conversion (`scripts/embed_int8.py`) is reproducible: re-running it on the source checkpoint gives the published `qweight` and `scales` bit for bit (checked on Ornith-1.5-9B 6.00 bpw); its reported error now uses the stored BF16 scales (0.00806 vs 0.00803 with FP32 scales).
- Fidelity: KL divergence against the BF16 Qwen3.8-27B with exllamav3's `qbench` on two traces (ours and turboderp's public one).

**Chat template**
- The published template renders correctly in 40 edge cases (empty and very long system prompt, no tools, two tool calls in one message, nested JSON and multi-line arguments, images with tools, two images, earlier reasoning, multi-turn with several tool results) × the reasoning settings (`enable_thinking: false`, `reasoning_effort` low / xhigh): all pass (`prueba_plantillas.py`). With Transformers directly, pass tool-call arguments as dicts (vLLM converts JSON strings itself).

**Runtime**
- Load, long-context needle, vision, tool calls with the real opencode request, 10 agent tasks × 3 runs, concurrency and VRAM: see *Family comparison*; engine errors during the run: 0.
- On a desktop RTX 3090 that also drives the display (300 W): Quick start at 215,000 tokens: 295 MiB really free.

**Scope**: results apply to the revision, software stack (vLLM 0.31.0 + exllamav3 1.5.1 + plugins), hardware and settings stated above; other setups are not tested.


## Limitations

- **Review the commands before they run.** As an agent, the model proposes shell commands and opencode can execute them: review them before running anything as root or with `sudo`, and give the agent a non-privileged account and a test machine where you can.
- Quality was not evaluated with standard benchmarks; the checks are KL divergence against BF16, long-context retrieval, tool calling, vision on synthetic images,
  short reasoning problems and agent tasks.
- The full 262,144 context does not fit at this rate on 24 GB with MTP and vision (turboderp's 3.00 bpw does, at a higher KLD).
- Decode is ≈20 % slower than our W4A16 GPTQ version on Ampere; the OrcaSAQ2 plugin is young and describes itself as a correctness path.
- The fp8 KV cache uses the default scale 1.0. Vision was validated with synthetic images only (terminal screenshot, chart, dense text), not photos or video.
- All numbers come from one machine and one software stack.

## Feedback welcome

I would be grateful for reports of installations of this checkpoint, **both working and failed ones**, especially on GPUs other than the RTX 3090.
Please open a discussion in the [Community](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB/discussions) tab with your
GPU(s), driver, vLLM / exllamav3 / plugin versions, the full command line and the result (speed, context, or the error message).

## Credits & license

### Credits & acknowledgements

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Qwen team).
- [exllamav3](https://github.com/turboderp-org/exllamav3) and the EXL3 format by **turboderp** (MIT), including the SC pipeline, `qbench` and the public
  evaluation trace; EXL3 is based on [QTIP](https://arxiv.org/abs/2406.11235) (Tseng et al.). Reference EXL3 quantizations used in the table:
  [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3).
- vLLM integration: the [OrcaSAQ2 plugin](https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel) by Continuum AI Corp; [vLLM](https://github.com/vllm-project/vllm).
- Our W4A16 sibling builds on the INT4-embedding work of [Marcin116](https://huggingface.co/Marcin116), [born2bewild](https://huggingface.co/born2bewild) and
  [syv-ai](https://github.com/syv-ai/qwen38-27b-rtx3090).

### License

Apache-2.0, same as the base model. This is a community quantization, not an official Qwen release. The OrcaSAQ2 plugin and exllamav3 are not
redistributed here; install them from their repositories under their own licenses.
