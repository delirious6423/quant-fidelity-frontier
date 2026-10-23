---
language:
- es
- en
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
datasets:
- HuggingFaceH4/ultrachat_200k
tags:
- sysadmin
- opencode
- linux
- vllm
- qwen3.8
- w4a16
- int4
- int8
- gptq
- compressed-tensors
- mtp
- speculative-decoding
- long-context
- rtx3090
- ampere
- endpoints_compatible
---

# Qwen3.8-27B — GPTQ W4A16 g64 + INT8 embeddings ("v4"), for 24 GB GPUs (MTP included)

> **Install from this card, not from generic snippets.** This checkpoint keeps its token embeddings quantized, which stock vLLM cannot load without a small patch. Follow [Requirements & install](#requirements--install-vllm-with-a-small-patch) and [Quick start](#quick-start-single-rtx-3090-24-gb); generic vLLM, SGLang or Transformers commands will fail.

A 4-bit quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for **vLLM on a single 24 GB Ampere GPU (RTX 3090)**, with the
model's **MTP head (INT8) for speculative decoding**. It is the **higher-fidelity successor** of our
[INT4-embedding checkpoint](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB): group size 64 instead of 128, **`embed_tokens` and `lm_head` in INT8** instead of INT4, and a
calibration set that mixes agent/sysadmin conversations with UltraChat. Measured against BF16, the **KL divergence drops from
0.0400 to 0.0175** (−56 %); the price is ≈1.5 GiB more weights, so the context on one 24 GB
GPU is **176,000 tokens** instead of 222,000.

> **License:** Apache-2.0, as the base model. This is not an official release of the model's authors or the Qwen team.

> **Tested setup:** RTX 3090 (24 GB, Ampere, power limit 350 W), vLLM 0.31.0 + the 4-line embedding patch in `patches/`, weights downloaded from this repo, the Quick start configuration (2026-10-14). Only this setup has been tested.

**Highlights (one RTX 3090 24 GB, measured on 2026-10-14; details in *Family comparison*):**
- **KLD 0.0175 against BF16** on our evaluation set (0.0171 on turboderp's public trace).
- **176,000 (text) · 131,072 with vision** on one 24 GB GPU with MTP (3.07 tokens per step); needle 8/8 at 162,208.
- **Tool calls with the real opencode request 59/60**, opencode agent tasks **29/30** (10 sysadmin tasks × 3 runs), images **15/15**.
- **Reasoning 48/48 at `xhigh`** with 20,414 reasoning tokens in total.
- Decode 101.4 / 164.6 / 170.2 tok/s with 1 / 2 / 8 requests; 196 MiB of VRAM really free at peak.

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
| **[GPTQ INT8 g64](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB) (this repo)** | 15.75 GiB | 176,000 (text) · 131,072 with vision | 0.0175 | 29/30 | 101.4 tok/s | a more faithful GPTQ (KLD 0.0175) on stock vLLM (+ the same patch), with less context |
| [EXL3 3.85](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB) | 13.62 GiB | 215,000 with vision | 0.0073 | 29/30 | 87.9 tok/s | the closest to BF16 (KLD 0.0073) with vision at 215,000; needs exllamav3 + our plugin |
| [EXL3 3.70](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB/tree/3.70bpw) | 13.19 GiB | 226,000 with vision | 0.0089 | 29/30 | 79.3 tok/s | more context (226,000) and 4 sessions with vision, a little less faithful than 3.85 |

### Files

- `model.safetensors` (+ index), `model_mtp.safetensors` (INT8 MTP head), `config.json` (compressed-tensors), `recipe.yaml` (as serialized by llm-compressor).
- `scripts/quantize_v4.py`, `scripts/mtp_quant.py`; `patches/` (vLLM patch and its Apache-2.0 license); `SHA256SUMS`.

## Results
### Family comparison (Qwen3.8-27B, same conditions)

Measured on 2026-10-14 for every variant in the same way: weights downloaded from Hugging Face, one RTX 3090 24 GB (power limit 350 W, no other use), vLLM 0.31.0 (GPTQ: + the embedding patch / EXL3: + exllamav3 1.5.1 + plugins), each variant with its own Quick start configuration. KLD and perplexity come from exllamav3's `qbench` (see *How to read it*).

| Variant | Weights | KLD mean / median (internal · public) | Perplexity (BF16 1.5860 · pub. 1.3332) | Needle (8 facts) near max. context | Tools: real opencode request | Agent tasks (10 × 3) | Reasoning `xhigh`: correct · tokens | Vision |
|---|---|---|---|---|---|---|---|---|
| [GPTQ W4A16 + INT4 embeddings](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB/tree/mtp-int8) (`mtp-int8`) | 14.21 GiB | 0.0400 / 0.0127 · pub. 0.0312 / 0.0017 | 1.6631 · pub. 1.3779 | 8/8 at 204,551 | 56/60 | 28/30 | 47/48 · 32,190 | 15/15 |
| **[GPTQ W4A16 g64 + INT8 embeddings](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB) (`main`, this repo)** | **15.75 GiB** | **0.0175 / 0.0049 · pub. 0.0171 / 0.0009** | **1.6203 · pub. 1.3568** | **8/8 at 162,208** | **59/60** | **29/30** | **48/48 · 20,414** | **15/15** |
| [EXL3 3.85 bpw](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB) (`main`) | 13.62 GiB | 0.0073 / 0.0018 · pub. 0.0067 / 0.0003 | 1.5976 · pub. 1.3430 | 8/8 at 197,991 | 60/60 | 29/30 | 48/48 · 22,474 | 15/15 |
| [EXL3 3.70 bpw](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-EXL3-3.85bpw-24GB/tree/3.70bpw) (`3.70bpw`) | 13.19 GiB | 0.0089 / 0.0023 · pub. 0.0085 / 0.0004 | 1.5999 · pub. 1.3466 | 8/8 at 208,158 | 60/60 | 29/30 | 48/48 · 20,873 | 15/15 |

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

### Earlier measurements

_Earlier measurements (single RTX 3090 24 GB, 300 W, 2026-10-06), before the controlled comparison of 2026-10-14: other dates, vLLM versions or settings, as stated in each table. Use *Family comparison* above to compare variants._


| Test | Result |
|---|---|
| KV pool | 178,816 tokens (`--max-model-len 176000`, KV 7.09e9 B) |
| 8 facts spread over 161,646 tokens (greedy) | **8/8**, 251 s (644 tok/s prefill) |
| 2 concurrent sessions of 79k tokens | 8/8 and 8/8, 841 tok/s aggregate prefill |
| Decode, 1 request, t=0 / t=0.6 | 81.2 / 93.2 tok/s |
| Decode, 2 concurrent, aggregate, t=0 / t=0.6 | 157.3 / 166.6 tok/s |
| MTP mean acceptance length (tokens per step, max 4) | 3.27 |
| Real opencode request, tool calls (`xhigh`, 3 cases × 5) | **15/15**, no parse errors |
| Reasoning tokens, 6 checkable problems × 8 (T 0.6), correct | low 19,100 (47/48) · medium 21,996 (48/48) · xhigh 23,269 (48/48) |
| `opencode run`, 4 sysadmin tasks × 2 (`xhigh`, T 0.3) | 8/8 tasks, 5,045 reasoning tokens, 287 s, 40 requests |
| Peak VRAM / really free | 23,854 MiB / 271 MiB, no OOM; fresh install from this repo: 177 MiB free |
| Installed from this repo following this card (2026-10-09) | 8/8 at 154,854 tokens, 2 × 76k 8/8, tool calls 14–15/15, no engine errors |

**Vision** (the quick-start command without `--language-model-only`, with `--max-model-len 131072 --kv-cache-memory-bytes 5500000000` `--limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"max_pixels":1003520}'`): KV pool 132,423 tokens; **4-image battery 15/15**, follow-up turn with prefix cache OK; needle 8/8 at 115,483 tokens; tool calls with the real opencode request 15/15; **869 MiB really free at peak**.

### Quality: KL divergence against BF16 (2026-10-05)

Measured with exllamav3's `qbench`, first on an **internal sysadmin/agent evaluation set** (not published; different conversations from the
calibration trace, but from the same domain), the W4A16 checkpoint evaluated as a BF16 copy holding the dequantized weights (Transformers):

| Checkpoint | KLD mean | median | p90 | Perplexity (BF16 1.5860) |
|---|---|---|---|---|
| **This repo (v4)** | **0.0175** | **0.0049** | 0.0476 | **1.6203** |
| INT4 embeddings, g128, UltraChat calibration | 0.0400 | 0.0127 | — | 1.6631 |

On turboderp's **public** evaluation trace (`qbench_prompts_gen.json`, the one used in all our cards, unrelated to our calibration data) the gain
holds: **KLD 0.0171** (median 0.0009, perplexity 1.3568) against 0.0312 (median 0.0018) for the INT4-embedding checkpoint,
so it is not just an effect of calibrating and evaluating on the same domain.

On Swift 1.5 we isolated the steps (internal set): INT8 embeddings / `lm_head` 0.0467 → 0.0396, + mixed calibration 0.0251, + group size 64 0.0215.

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

## Requirements & install (vLLM with a small patch)

Stock vLLM (0.29 to 0.31) does not pass `quant_config` to the embedding layer of `qwen3_5` models, so a checkpoint with quantized
embeddings fails to load (`no parameter named 'embed_tokens.weight_packed'`). Apply the 4-line patch in
[`patches/qwen3_5-embed-quant.patch`](patches/qwen3_5-embed-quant.patch) (from [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090), Apache-2.0):

```bash
uv venv --python 3.12 venv && uv pip install --python venv/bin/python "vllm==0.31.0" --torch-backend=auto        # torch 2.13.0
VLLM_DIR=$(venv/bin/python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')
hf download JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB patches/qwen3_5-embed-quant.patch --local-dir .
patch -p1 -d "$VLLM_DIR" < patches/qwen3_5-embed-quant.patch
```

## Quick start (single RTX 3090, 24 GB)

**Text only, MTP 3, 176k context, 2 concurrent sequences** (validated configuration):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB \
  --served-model-name qwen3.8-27b \
  --language-model-only \
  --max-model-len 176000 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-memory-bytes 7090000000 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**For many users** (measured on 2026-10-14, same GPU): the same command with `--kv-cache-memory-bytes 6790000000 --max-model-len 165000 --max-num-seqs 8` — 8 sessions of up to 165,000 tokens, MTP 3. Aggregate decode 470 tok/s with 8 requests (Quick start: 170 with 8); a single request runs at 94 tok/s (101 with the Quick start); 8 of 8 simultaneous sessions returned all 8 needle facts; 386 MiB of VRAM really free. Rule for CUDA graphs: they must cover sessions × (1 + MTP) tokens per step.

## Using it with opencode

How we use it as a **coding / sysadmin agent from [opencode](https://opencode.ai)** (opencode 1.18.x). `limit.context` must match `--max-model-len` of your
`vllm serve` command (176000 in the Quick start above).

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local vLLM",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "qwen3-8-27b-gptq-w4a16-g64-embed-int8-24": {
          "name": "Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB (local)",
          "tool_call": true,
          "temperature": true,
          "limit": { "context": 176000, "output": 22000 },
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
    "build": { "model": "local-vllm/qwen3-8-27b-gptq-w4a16-g64-embed-int8-24", "variant": "xhigh", "temperature": 1.0 },
    "plan":  { "model": "local-vllm/qwen3-8-27b-gptq-w4a16-g64-embed-int8-24", "variant": "xhigh", "temperature": 1.0 }
  }
}
```

- **Declare `"temperature": true` in the model**, otherwise opencode sends no temperature to a custom model and vLLM falls back to its defaults.
- Temperature 1.0 is this repo's `generation_config.json` default. Our first agent tests of this family used 0.3, but at 0.3 the model falls into repetition loops on long answers (see *Sampling and repetition loops*), and in our agent tests (10 real sysadmin tasks × 3 runs with `opencode run`) 1.0 scored the same as 0.3 within one task: 29/30 at 1.0 and 30/30 at 0.3 for Qwen3.8-27B, -pi and Swift 1.5 27B; Qwopus3.8, served with its `chat_template-xml-tools.jinja`, 28/30 at 1.0 and 26/30 at 0.3.
- With `output: 22000` opencode compacts the history at ≈`context − 22k` tokens of input.
- Start opencode inside a project directory (it takes a git snapshot before each step); for `opencode run` from scripts, close stdin (`</dev/null`).

## Important notes

### Memory (read this before changing the numbers)

- **The driver reserves 453 MiB of the 24,576 MiB that vLLM cannot use.** "24,576 − used" over-estimates the headroom by that amount;
  check the real value with `nvidia-smi --query-gpu=memory.used,memory.free --format=csv`.
- The configuration above left **177 MiB really free at peak** when installed exactly as described here (fresh venv, model downloaded from this repo,
  155k-token prompt and 2 × 76k concurrent); our own build peaked at 23,854 MiB used / 271 MiB free (415 MiB free at rest).
  Both are above our 100 MiB safety margin, but there is little room: do not raise the KV cache.
  `--max-num-batched-tokens 1024` and `expandable_segments` are part of that budget: without them the prefill peak is several hundred MiB higher.
- Size the KV cache with `--kv-cache-memory-bytes`, not `--gpu-memory-utilization`: with this hybrid (GatedDeltaNet) model the memory profiling
  under-estimates the prefill activations and a percentage over-allocates the KV cache.
- With MTP the KV capacity grows in block-sized steps: check the `GPU KV cache size` line; `--max-model-len` must not exceed it.
- Our RTX 3090 hung once (driver `GSP RM heartbeat timed out`, Xid 154) during a 168k-token prefill of this checkpoint at a 350 W power limit;
  at 300 W with a more aggressive fan curve the same test passed and all the numbers here are from that setup. If your card is marginal, lower its power limit.

### Other notes

- Use `--tool-call-parser qwen3_coder`. On Ampere (sm86), fp8 KV cache only works with the **FlashInfer** attention backend (the default).
- `reasoning_effort` accepts only `xhigh`, `medium` and `low` (`high` returns HTTP 400); they are prompt instructions, not limits. For a hard
  cap send `thinking_token_budget` in the request (vLLM 0.29 enforces it with `--reasoning-parser qwen3`, also with MTP).
- For use from [opencode](https://opencode.ai), the configuration in the [INT4-embedding card](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-embed-int4-24GB) applies with
  `"limit": { "context": 176000, "output": 22000 }` (declare `"temperature": true` in the model, or opencode sends no temperature).

## Quantization recipe ("v4")

| Component | Treatment |
|---|---|
| Linear layers of the language model (attention, MLP, GatedDeltaNet `in_proj_qkv` / `in_proj_z` / `out_proj`) | **GPTQ W4A16**, symmetric INT4, **group size 64**, `actorder=static`, dampening 0.01 |
| `embed_tokens`, `lm_head` | **INT8** symmetric, group size 128, RTN (min-max) |
| GatedDeltaNet `in_proj_a`, `in_proj_b`, norms, `A_log`, `dt_bias`, `conv1d` | kept in BF16 |
| MTP head (`mtp.*`) | INT8 g128 RTN (`mtp.fc`, `mtp.layers.*`); shares the INT8 embeddings / `lm_head` |
| Vision tower | kept in BF16 |

- Source: `Qwen/Qwen3.8-27B`.
- Tool: llm-compressor 0.14.0, compressed-tensors 0.19.0, transformers 5.17.0, torch 2.14.0+cu130.
- **Calibration: 512 samples × 2048 tokens**: 196 conversations from an internal Linux sysadmin / agent trace (multi-turn, with tool calls and reasoning;
  not published) plus 316 samples of `HuggingFaceH4/ultrachat_200k` (`train_sft`, seed 42), shuffled. Only GPTQ uses it; the INT8 parts are RTN.
- Tokenizer and chat template are identical to the base model.
- Scripts: [`scripts/quantize_v4.py`](scripts/quantize_v4.py) (with `CAL_TRACE` you can mix in your own trace; without it, UltraChat only) and
  [`scripts/mtp_quant.py`](scripts/mtp_quant.py) (INT8 MTP head).

## Integrity and validation

What was checked for this checkpoint, how, and where the numbers are (measured on 2026-10-14; the method and every figure are in *Family comparison*).

**Artifact integrity**
- Every weight file is checked against our local copy with SHA-256 before publication (`SHA256SUMS` in this repo).
- The measurements used the weights downloaded from this repo at commit `165f819ec30f`; later commits change only metadata, documentation or the plugin, never the `.safetensors` files.
- vLLM loads every tensor listed in `model.safetensors.index.json` on start-up (a missing tensor or a wrong shape would stop it): checked from a fresh download.

**Quantization**
- Quantization recipe and calibration in `recipe.yaml` and `scripts/` (llm-compressor GPTQ, `compressed-tensors` format).
- Fidelity: KL divergence against the BF16 Qwen3.8-27B with exllamav3's `qbench` on two traces (ours and turboderp's public one).

**Chat template**
- The published template renders correctly in 40 edge cases (empty and very long system prompt, no tools, two tool calls in one message, nested JSON and multi-line arguments, images with tools, two images, earlier reasoning, multi-turn with several tool results) × the reasoning settings (`enable_thinking: false`, `reasoning_effort` low / xhigh): all pass (`prueba_plantillas.py`). With Transformers directly, pass tool-call arguments as dicts (vLLM converts JSON strings itself).

**Runtime**
- Load, long-context needle, vision, tool calls with the real opencode request, 10 agent tasks × 3 runs, concurrency and VRAM: see *Family comparison*; engine errors during the run: 0.
- On a desktop RTX 3090 that also drives the display (300 W): Quick start at 176,000 tokens: 193 MiB really free; vision config at 131,072 tokens: 891 MiB really free.

**Scope**: results apply to the revision, software stack (vLLM 0.31.0 + the embedding patch), hardware and settings stated above; other setups are not tested.


## Limitations

- **Review the commands before they run.** As an agent, the model proposes shell commands and opencode can execute them: review them before running anything as root or with `sudo`, and give the agent a non-privileged account and a test machine where you can.
- Quality was not evaluated with standard benchmarks; the checks are KL divergence (above), long-context retrieval, tool calling, short reasoning
  problems and agent tasks.
- 176k is less than the 262k the model supports; with 24 GB, use the INT4-embedding or EXL3 checkpoints for longer contexts.
- The fp8 KV cache uses the default scale 1.0. All numbers come from one machine and one software stack.

## Feedback welcome

Reports of installations, **both working and failed ones**, are very welcome in the
[Community](https://huggingface.co/JRamirez-UAB/Qwen3.8-27B-GPTQ-W4A16-g64-embed-int8-24GB/discussions) tab: GPU(s), driver, vLLM version, the full command line and the result (speed, context, or the error).

## Credits & license

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- vLLM patch: [syv-ai](https://github.com/syv-ai/qwen38-27b-rtx3090); quantized-embedding layout first shown by [Marcin116](https://huggingface.co/Marcin116).
- Tools: [llm-compressor](https://github.com/vllm-project/llm-compressor), [compressed-tensors](https://github.com/neuralmagic/compressed-tensors),
  [vLLM](https://github.com/vllm-project/vllm); KL divergence with `qbench` from [exllamav3](https://github.com/turboderp-org/exllamav3).
