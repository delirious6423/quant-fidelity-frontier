---
language:
- es
- en
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
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
- moe
- qwen3_8
- mtp
- speculative-decoding
- long-context
- vision
- 48gb
- rtx-pro-5000
- endpoints_compatible
---

# Qwen3.8-Flash-Next — EXL3 2.25 bpw for vLLM on a 48 GB GPU (262k context, 4 sessions, vision, MTP)

> **Install from this card, not from generic snippets.** This is an EXL3 checkpoint: vLLM loads it only together with exllamav3 and our plugins. Follow [Requirements & install](#requirements--install-vllm-031--exllamav3-151--plugins) and [Quick start](#quick-start-one-48-gb-gpu-eg-rtx-pro-5000) (on Ampere GPUs, TabbyAPI as described below); a plain `pip install vllm`, Transformers or SGLang cannot load it.

EXL3 quantization (2.25 bits per weight on average) of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (MoE, 125B total / 6B active,
plus a 51B n-gram "PLE" table, MTP head and vision), made to serve the **whole model on one 48 GB GPU** with **vLLM** (continuous batching):
**262,144 tokens per session without YaRN, 4 concurrent sessions sharing a 420k-token KV cache, vision and MTP**.
Same layout and memory footprint as our [Swift 1.5 Flash-Next 2.25 bpw](https://huggingface.co/JRamirez-UAB/Swift-1.5-Qwen3.8-Flash-Next-EXL3-2.25bpw-48GB),
which was validated on a real **NVIDIA RTX PRO 5000 Blackwell (48 GB)**.

> **Our focus.** We quantize and evaluate models for one use case: a **local Linux system-administration agent** (opencode with tool calling)
> for a university IT department, **used in Spanish**. Our calibration traces include real sysadmin/agent conversations in Spanish, and our quality
> checks (tool calls with the real opencode request, agent tasks with real command execution, long-context retrieval, long answers in Spanish) reflect
> that use. KL divergence is also reported on turboderp's public trace, so it can be compared and reproduced. Numbers for other domains or languages may differ.

> ⚠️ **Use temperature 1.0** (Qwen's recommended sampling for this model: temperature 1.0, top_p 0.95, top_k 20 — this repo's `generation_config.json`).
> At lower temperatures Qwen3.8-Flash-Next models **fall into repetition loops** on long answers (see [Sampling and repetition loops](#sampling-and-repetition-loops)).

## What is in this repository (and what is not)

### Which one to use

All Qwen3.8-Flash-Next (125B-A6B) variants we publish, measured on 2026-10-14 under the same conditions (see *Family comparison* for every number):

| Variant | Weights | Context on the test GPU | KLD vs BF16 | Agent tasks | Decode, 1 request | Pick it for |
|---|---|---|---|---|---|---|
| **[EXL3 2.25 · vLLM](https://huggingface.co/JRamirez-UAB/Qwen3.8-Flash-Next-EXL3-2.25bpw-48GB) (this repo)** | 37.68 GiB | 262,144 with vision | 0.0532 | 30/30 | 118.5 tok/s | one 48 GB Ada or Blackwell GPU with vLLM: 262,144 tokens with vision and 4 sessions, fp8 KV cache |
| **[EXL3 2.25 · TabbyAPI](https://huggingface.co/JRamirez-UAB/Qwen3.8-Flash-Next-EXL3-2.25bpw-48GB) (this repo)** | 37.68 GiB | 262,144 with vision | 0.0532 | 30/30 | 93.5 tok/s | a 48 GB Ampere GPU (A40): TabbyAPI with a Q8 cache (vLLM would need a BF16 KV cache there) and ≈6 GB of host RAM instead of ≈90 GB |

| File(s) | What | Used by |
|---|---|---|
| `model-*.safetensors`, `config.json`, index | Qwen3.8-Flash-Next weights in EXL3 (2.25 bpw average) | vLLM and exllamav3 |
| `ngram_embedding.safetensors` (26 GB) | the n-gram (PLE) table in EXL3 | exllamav3 only (vLLM cannot read it) |
| `scripts/assemble.py` | builds the vLLM directory (downloads the two files below) | vLLM |
| `plugin/qwen38_exl3_vision/` | our vLLM plugin 0.3.1: EXL3 MoE experts, vision tower, Qwen3.8-Flash-Next (`Qwen4Exp`) | vLLM |

Download size: **66.7 GB** (40.5 GB of EXL3 weights + 26.2 GB EXL3 n-gram table). vLLM users can skip the n-gram table
(`--exclude ngram_embedding.safetensors`, 40.5 GB).

**Not included, because they are not ours to redistribute** — `scripts/assemble.py` downloads them from their owners:
- **NVIDIA's FP8 PLE table** (`model-fp8-mtp-ple.safetensors` from [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4),
  NVIDIA Open Model License; 54 GB download, 51 GB once extracted). vLLM keeps it in **pinned host RAM**, not on the GPU.
- **turboderp's MTP mixer patch** (`mtp_hyper_connection_mixer_patch.safetensors`, 13 MB, from
  [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3), branch `3.05bpw_h5_ng5`): without it the MTP head does not load.

## Results
### Family comparison (Qwen3.8-Flash-Next (125B-A6B), same conditions)

Measured on 2026-10-14 for every variant in the same way: weights downloaded from Hugging Face, one RTX PRO 6000 Blackwell Server Edition 96 GB (power limit 300 W, 2,270 MHz) with vLLM 0.31.0 + exllamav3 1.5.1 + plugins, and, for the 2.25 bpw build, one A40 48 GB with TabbyAPI (exllamav3 1.6.1, Q8 cache) as described for Ampere GPUs, each variant with its own Quick start configuration. KLD and perplexity come from exllamav3's `qbench` (see *How to read it*).

| Variant | Weights | KLD mean / median (internal · public) | Perplexity (BF16 2.1510 · pub. 1.6917) | Needle (8 facts) near max. context | Tools: real opencode request | Agent tasks (10 × 3) | Reasoning `xhigh`: correct · tokens | Vision |
|---|---|---|---|---|---|---|---|---|
| **[EXL3 2.25 bpw (vLLM)](https://huggingface.co/JRamirez-UAB/Qwen3.8-Flash-Next-EXL3-2.25bpw-48GB) (`main`, this repo)** | **37.68 GiB** | **0.0532 / 0.0215 · pub. 0.0481 / 0.0064** | **2.1944 · pub. 1.7135** | **8/8 at 241,437** | **58/60** | **30/30** | **48/48 · 19,264** | **15/15** |
| **[EXL3 2.25 bpw (TabbyAPI, Ampere)](https://huggingface.co/JRamirez-UAB/Qwen3.8-Flash-Next-EXL3-2.25bpw-48GB) (`main`, this repo)** | **37.68 GiB** | **0.0532 / 0.0215 · pub. 0.0481 / 0.0064** | **2.1944 · pub. 1.7135** | **8/8 at 241,437** | **57/60** | **30/30** | **48/48 · 18,780** | **15/15** |

| Variant | Context on the test GPU | Sessions | Prefill near max. context | Decode, 1 / 2 / 8 requests (tok/s) | MTP tokens per step | VRAM used at peak |
|---|---|---|---|---|---|---|
| EXL3 2.25 · vLLM | 262,144 with vision | 4 | 3,438 tok/s | 118.5 / 184.0 / 279.6 | 2.74 | 48,417 MiB |
| EXL3 2.25 · TabbyAPI | 262,144 with vision | 4 | 1,171 tok/s | 93.5 / 69.5 / 93.8 | — | 45,125 MiB (366 MiB free on the A40 48 GB) |

How to read it:
- **KLD / perplexity**: next-token KL divergence against the BF16 model with exllamav3's `qbench` (lower is better). *Internal*: our sysadmin/agent evaluation set (not published, separate from the calibration data); *public*: turboderp's `qbench_prompts_gen.json`, reproducible by anyone. GPTQ checkpoints were evaluated as a BF16 copy holding the dequantized weights. These two columns come from `qbench`, not from this campaign.
- **Needle**: 8 facts spread through one prompt at about 92 % of the served context; all 8 must come back. **Tools**: the real request of the opencode agent (system prompt + 11 tools), 3 cases × 20 tries, at the model's default sampling (`generation_config.json`); a try counts when it calls a valid tool with valid arguments and the expected command. **Agent**: 10 sysadmin tasks run with `opencode run` and real command execution, 3 runs, at the recommended temperature; with sampling on, the same checkpoint varies by about ±2 of 30 between runs, so other measurements in this card may differ by that much. **Reasoning**: 6 checkable problems × 8 at `xhigh`; total reasoning tokens over the 48. **Vision**: our 4-image test.
- **Prefill / decode**: prefill from the needle prompt; decode is aggregate throughput with 1 / 2 / 8 simultaneous requests (temperature 0; with more requests than *Sessions*, the extra ones wait). **MTP tokens per step**: vLLM's speculative-decoding metrics over the whole run.
- **VRAM really free**: lowest `memory.free` while serving (nvidia-smi every 0.5 s, from the end of start-up to the end of the run); we require ≥100 MiB.
- Older sections below keep their own date and setup; use this table to compare variants.

**How many sessions fit** with each Quick start configuration (KV size in tokens as reported by vLLM at start-up; a session uses the KV of its own context, and vLLM never runs more than *max sessions* at once):

| Variant | KV pool (tokens) | Max sessions | 32k each | 64k each | 128k each | Full context each |
|---|---|---|---|---|---|---|
| EXL3 2.25 · vLLM | 419,939 | 4 | 4 | 4 | 3 | 1 × 262,144 |

**Few users or many users.** The Quick start configuration is tuned for 1–4 simultaneous users (fastest single request). For more users, the *many users* option below (measured the same day, same GPU) gives more total throughput; each request gets slower when many run at once. Same model, same quality: only the serving settings change.

| Variant | Configuration | Decode by simultaneous requests (aggregate tok/s) | N sessions at once, needle 8/8 | Tools (5 × 3) | VRAM used at peak |
|---|---|---|---|---|---|
| EXL3 2.25 · vLLM | Quick start (4 sessions, MTP 3) | 1: 118 · 2: 184 · 8: 280 | — | 58/60 | 48,417 MiB |
| EXL3 2.25 · vLLM | many users (16 sessions of up to 262,144 tokens, MTP 1, CUDA graphs up to 64 tokens) | 1: 90 · 4: 235 · 8: 379 · 16: 555 | 16/16 of 16k | 15/15 | 48,471 MiB |

<!-- end of the controlled comparison -->

### Earlier measurements

_Earlier measurements, before the controlled comparison of 2026-10-14: other dates, vLLM versions or settings, as stated in each table. Use *Family comparison* above to compare variants._


| Check | Result |
|---|---|
| KL divergence vs Qwen3.8-Flash-Next **BF16** (public trace / our sysadmin trace) | **0.048 / 0.053** (turboderp's 2.05 bpw EXL3, same reference: 0.060 / 0.069) |
| Tool calls with the real opencode request (reasoning xhigh · no reasoning) | 29/30 · 30/30 |
| Agent tasks (`opencode run`, real command execution, 20 tasks × 2), **file names as written** | **39/40** |
| Hard sysadmin tasks (8 × 3: Slurm GRES/QoS, FreeIPA, systemd, nginx, nftables, ZFS, logs) | **20/24** (best of all the models we tested; Swift 1.5 Flash-Next 19/24, 27B models 16–18/24) |
| Memory on a 48 GB GPU (same tensor sizes as our Swift build, validated on a real RTX PRO 5000) | KV cache 7.5e9 bytes fp8 = 419,939 tokens, 4 sessions × 262,144, vision + MTP |
| A40 (48 GB, Ampere) | vLLM only with a BF16 KV cache (≈98k tokens); **TabbyAPI: 278,528 tokens with vision** (see *Quick start*); our hard-task runs were done on an A40 |

Our recipe keeps every expert at ≥2 bits (`sc_optimize -mink 2`); calibrating with the model's own answers (see *Quantization recipe*) gave a lower KLD than
exllamav3's default calibration data (0.051 / 0.058) and the best agent result without help in file names.

## Sampling and repetition loops

Measured on 2026-10-14 with the same software and GPU as *Family comparison*, on our EXL3 2.25 · vLLM build of Qwen3.8-Flash-Next (125B-A6B) (loops come from the model and the sampling, not from the quantization format): 5 long Spanish sysadmin prompts × 6 runs per temperature, up to 16,000 tokens, top_p 0.95 and top_k 20; a loop = more than 30 % repeated 8-grams. Serving: the Quick start with a 32k context and 8 sessions.

| Temperature | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| Loops / 30 | — | — | 13 | 7 | 5 | 2 | 0 | 0 |
| Answers that finished before 16,000 tokens / 30 | — | — | 1 | 5 | 0 | 0 | 4 | 1 |

From temperature **0.9** up, loops are rare (at most 1 of 30). This repo's default (`generation_config.json`) is **1.0**.

**With TabbyAPI** (A40, Q8 cache, same test, measured on Qwen3.8-Flash-Next 2.25): loops / 30 at T 0.3–1.0 = 8 · 11 · 9 · 7 · 8 · **0 · 0 · 0**, the same picture as vLLM. TabbyAPI has its own loop detector: it stops a generation it sees repeating (100 of 232 answers, almost all at T ≤ 0.7), so a loop ends early as a cut-off answer instead of running to the token limit. At T 0.9–1.0, 9 of 60 requests ended with a server error in our run (8 long generations at a time): keep T 1.0 and retry failed requests.

Other models, same test and setup (loops / 30):

| Model (build) | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B · EXL3 3.85 | 12 | 12 | 7 | 8 | 1 | 1 | 1 | 0 |
| Qwen3.8-27B-pi · EXL3 3.85 | 9 | 4 | 9 | 10 | 4 | 2 | 1 | 0 |
| Qwopus3.8-27B-Flash-V2 · EXL3 3.85 | 3 | 1 | 2 | 1 | 2 | 0 | 0 | 0 |
| Swift 1.5 (Qwen3.8-27B) · EXL3 3.85 | 7 | 5 | 3 | 4 | 3 | 1 | 1 | 0 |
| Ornith-1.5-9B · EXL3 6.00 (16 GB) | 6 | 2 | 3 | 1 | 1 | 0 | 0 | 0 |
| NeoHorse-1-9B · EXL3 6.00 (16 GB) | 6 | 3 | 3 | 1 | 0 | 0 | 0 | 0 |
| Qwen3.6-35B-A3B · EXL3 4.00 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| Ornith-1.5-35B-A3B · EXL3 4.00 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Qwopus3.6-35B-A3B-Coder · EXL3 4.00 | 9 | 13 | 12 | 6 | 11 | 6 | 5 | 4 |
| Swift 1.5 Qwen3.8-Flash-Next (125B-A6B) · EXL3 2.25 · vLLM | 24 | 24 | 21 | 21 | 17 | 8 | 0 | 0 |

## Requirements & install (vLLM 0.31 + exllamav3 1.5.1 + vllm-quantpack)

EXL3 is not a native vLLM format. One plugin is loaded through vLLM's `vllm.general_plugins` entry point:
- **[vllm-quantpack](https://github.com/jramirez-uab/vllm-quantpack)** (ours, on PyPI): EXL3 linear, embedding and MoE layers through exllamav3's kernels, plus the vision tower, the MTP head and the Qwen3.8-Flash-Next model (per-layer embeddings).

Use **exllamav3 1.5.1**: with 1.6.x the KLD is the same but tool calls without reasoning got worse in our tests. Python 3.12; a CUDA 13.0 toolkit for flashinfer's JIT
kernels (point `CUDA_HOME` at it if the `nvcc` from pip does not match the CUDA headers).

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python vllm==0.31.0 --torch-backend=auto          # torch 2.13.0
W=exllamav3-1.5.1+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl
curl -L -o /tmp/$W "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.1/${W/+/%2B}"
hf download JRamirez-UAB/Qwen3.8-Flash-Next-EXL3-2.25bpw-48GB --local-dir qwen-fn-225 --exclude ngram_embedding.safetensors
uv pip install --python venv/bin/python /tmp/$W vllm-quantpack
venv/bin/python qwen-fn-225/scripts/assemble.py --src qwen-fn-225 --dst qwen-fn-225-vllm
```
> **Plugin (2026-10-28):** [vllm-quantpack](https://github.com/jramirez-uab/vllm-quantpack) (`pip install vllm-quantpack`) replaces the OrcaSAQ2 + `qwen38_exl3_vision` plugins. The measurements in this card were made with the previous stack (OrcaSAQ2 + our plugins); this checkpoint was then served again with vllm-quantpack 0.1.0 following this card on a clean install (text, image and tool calls checked). The previous files stay in `plugin/` for reference.


## Quick start (one 48 GB GPU, e.g. RTX PRO 5000)

The host needs **about 90 GB of RAM** for this process (the PLE table is pinned in RAM: 51 GB); with a 70 GB limit our Swift build was killed for lack of RAM.

```bash
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve qwen-fn-225-vllm --served-model-name qwen-fn \
  --max-model-len 262144 --kv-cache-dtype fp8 --kv-cache-memory-bytes 7500000000 \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --enable-prefix-caching --mamba-cache-mode align \
  --limit-mm-per-prompt '{"image":4,"video":0}' --mm-processor-kwargs '{"max_pixels":1003520}' \
  --compilation-config '{"max_cudagraph_capture_size":16}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

### On a 48 GB Ampere GPU (A40): TabbyAPI instead of vLLM

On Ampere, vLLM can only give this model a **BF16** KV cache (its attention only accepts BF16 or FP8-e4m3, and Ampere has no FP8-e4m3), which leaves ≈98–131k
tokens on an A40. **[TabbyAPI](https://github.com/theroyallab/tabbyAPI)** (exllamav3 native) reads this repository **as is** — no `assemble.py`, no NVIDIA PLE
table, the EXL3 n-gram table stays in host RAM (**6 GB of RAM** for the process instead of ≈90 GB) — and uses its own quantized cache:

Install (how we did it; TabbyAPI commit `2636ef602b55b5343cf3740dd55a77f21e0a8003`, Python 3.12, torch 2.13.0 + CUDA 13.2, exllamav3 1.6.1):

```bash
git clone https://github.com/theroyallab/tabbyAPI && cd tabbyAPI && git checkout 2636ef602b55b5343cf3740dd55a77f21e0a8003
uv venv --python 3.12 venv
uv pip install --python venv/bin/python torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132
W=exllamav3-1.6.1+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl
curl -L -o /tmp/$W "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.1/${W/+/%2B}"
uv pip install --python venv/bin/python /tmp/$W -e .
# the model: this repo as downloaded, including ngram_embedding.safetensors (no assemble.py, no NVIDIA PLE)
hf download JRamirez-UAB/Qwen3.8-Flash-Next-EXL3-2.25bpw-48GB --local-dir models/qwen-fn-225
venv/bin/python main.py          # reads config.yml (below) and serves an OpenAI-compatible API
```

```yaml
# config.yml (TabbyAPI, exllamav3 1.6.1) — measured on a real A40 (48 GB)
network: {host: 127.0.0.1, port: 5000, disable_auth: true}
model: {model_dir: models, model_name: qwen-fn-225, max_seq_len: 262144, cache_size: 270336, cache_mode: Q8,
        chunk_size: 2048, max_batch_size: 4, vision: true}
draft_model: {draft_mode: mtp, draft_num_tokens: 3}
```

| A40, vision on | Result |
|---|---|
| Shared cache (all sessions together) | **270,336 tokens** (≤262,144 per session, up to 4 sessions); 366 MiB really free at peak (measured 2026-10-14; with 278,528 the Qwen3.8 build kept only 92 MiB free, below our 100 MiB margin) |
| Decode (1 · 2 requests, MTP 3) | at the recommended **T 1.0: 59–78 · 82–93 tok/s**; at T 0: 75–76 · 107–113 tok/s (vLLM on the same A40 at T 0, BF16 KV: 92 · 139) |
| Prefill | ≈900–960 tok/s (vLLM: ≈1,200) |
| Needle 8/8 · vision · tool calls · agent tasks | 8/8 at 191k · 15/15 · 29/30 · 19/20 (Swift build) |

Note that TabbyAPI's cache is one pool shared by all sessions (vLLM reserves per session): 278k is one 262k session plus a little, or four ≈70k sessions.

**For many users** (measured on 2026-10-14, same GPU): the same command with `--max-num-seqs 16 --speculative-config '{"method":"mtp","num_speculative_tokens":1}' --compilation-config '{"max_cudagraph_capture_size":64}'` — 16 sessions of up to 262,144 tokens, MTP 1, CUDA graphs up to 64 tokens. Aggregate decode 555 tok/s with 16 requests (Quick start: 280 with 8); a single request runs at 90 tok/s (118 with the Quick start); 16 of 16 simultaneous sessions returned all 8 needle facts; 48,471 MiB of VRAM used at peak. Rule for CUDA graphs: they must cover sessions × (1 + MTP) tokens per step.

## Using it with opencode (how we tested)

This checkpoint is meant to be used as a **coding / sysadmin agent from [opencode](https://opencode.ai)** (tested with opencode 1.18.35). Tool calling was
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
        "qwen-fn": {
          "name": "Qwen3.8-Flash-Next EXL3 2.25 (local)",
          "tool_call": true,
          "temperature": true,
          "attachment": true,
          "modalities": { "input": ["text", "image"], "output": ["text"] },
          "limit": { "context": 262144, "output": 32000 },
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
    "build": { "model": "local-vllm/qwen-fn", "variant": "xhigh", "temperature": 1.0 },
    "plan":  { "model": "local-vllm/qwen-fn", "variant": "xhigh", "temperature": 1.0 }
  }
}
```

- **`limit.context` = `--max-model-len`** (262144); with `output: 32000` opencode compacts the history at ≈`context − 32k` tokens of input.
- **Declare `"temperature": true` in the model**, otherwise opencode sends no temperature to a custom model (checked by capturing the requests), and set it to
  **1.0** (see [Sampling and repetition loops](#sampling-and-repetition-loops)). Our agent tasks ran at 0.6 (short tool-driven turns, no loops seen there); long answers need 1.0.
- Without `attachment` / `modalities` opencode will not send images; drop them if you serve with `--language-model-only`.
- Start opencode inside a project directory (it takes a git snapshot before each step); for `opencode run` from scripts, close stdin (`</dev/null`).

## PLE table outside the vLLM process (experimental)

[service-ple](https://github.com/jramirez-uab/service-ple) is an experimental project that keeps this model's PLE table out of the vLLM process: a per-machine service with a shared cache that reads the rows from the local file or forwards them to a central server. See its repository for details and current status.

## Important notes

- **Plugin 0.3.1 and CUDA graphs** (2026-10-14): set `max_cudagraph_capture_size` ≥ `max-num-seqs` × (1 + MTP tokens) so that every decode step is
  replayed from a graph (the Quick start's 16 is enough for its 4 sessions with MTP 3). 0.3.1 runs every step of up to 256 tokens in exllamav3's fused MoE
  kernel; 0.3.0 only did so up to 25 tokens (10 experts per token × tokens ≤ 256) and fails at start with a larger capture size. RTX PRO 6000 (300 W),
  Swift 2.25 bpw, 16 sequences, short prompts, total tok/s with 1 / 4 / 8 / 16 requests at once: 0.3.0 with the Quick start's graphs ≤ 16 and MTP 3 →
  107 / 223 / 158 / 237; 0.3.1 with graphs ≤ 64 and MTP 1 → 83 / 225 / 351 / **496**; 0.3.1, graphs ≤ 64, MTP 3 → 91 / 233 / 334 / 347. So: MTP 3 for one
  user, **MTP 1 with graphs ≥ 2 × sequences for many**. KL divergence 0.3.1 vs 0.3.0: 0.0193 (two 0.3.0 runs differ by 0.0200).
- **Memory.** Size the KV cache with `--kv-cache-memory-bytes`, never with `--gpu-memory-utilization`, and read the real headroom with `nvidia-smi`
  (the driver reserves ≈450 MiB that vLLM cannot use). On the RTX PRO 5000, 7.5e9 left 457 MiB free at peak with 4 long sessions, vision and MTP; 7.75e9 did not fit.
- `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` and CUDA-graph captures ≤16 are required for the EXL3 MoE path.
- **File names in non-English prompts.** Qwen3.x models sometimes write an output file under the conventional English name when its Spanish name is a system
  concept (`procesos.csv` → `processes.csv`). This build got 39/40 agent tasks as written; writing file names and paths in backticks avoids it entirely.
- **Invented command options.** With uncommon CLIs (Slurm `sacctmgr`, FreeIPA `ipa`) it may guess option names, like every model we tested; giving the agent the
  tool's real `--help`/`man` output is the practical fix.

## Quantization recipe

exllamav3 1.5.1 from the official Qwen3.8-Flash-Next BF16: per-tensor bit allocation with `sc_optimize` (target 2.25 bpw, `-mink 2`: no expert below 2 bits —
83 % of the experts at 2 bits, 17 % at 3), head 6 bits, vision 6, MTP 4, a shared 4-bit n-gram table. **Calibration on the model's own answers**: 500 rows
(200 agent turns with the real opencode tools, 100 general Spanish sysadmin prompts and 200 exllamav3 default rows) generated by NVIDIA's NVFP4 build of the same model.

## Integrity and validation

What was checked for this checkpoint, how, and where the numbers are (measured on 2026-10-14; the method and every figure are in *Family comparison*).

**Artifact integrity**
- Every weight file is checked against our local copy with SHA-256 before publication; Hugging Face shows each file's SHA-256 in *Files and versions*.
- The measurements used the weights downloaded from this repo at commit `cfe253f16dd8`; later commits change only metadata, documentation or the plugin, never the `.safetensors` files.
- vLLM loads every tensor listed in `model.safetensors.index.json` on start-up (a missing tensor or a wrong shape would stop it): checked from a fresh download.

**Quantization**
- Per-tensor bit allocation is recorded by exllamav3 in `quantization_config.json`; its `tensor_storage` entry for `embed_tokens` describes the INT8 `qweight` + BF16 `scales` actually stored (updated on 2026-10-14; the plugin only reads the `.trellis` entries).
- The INT8 embedding conversion (`scripts/embed_int8.py`) is reproducible: re-running it on the source checkpoint gives the published `qweight` and `scales` bit for bit (checked on Ornith-1.5-9B 6.00 bpw); its reported error now uses the stored BF16 scales (0.00806 vs 0.00803 with FP32 scales).
- Fidelity: KL divergence against the BF16 Qwen3.8-Flash-Next (125B-A6B) with exllamav3's `qbench` on two traces (ours and turboderp's public one).

**Chat template**
- The published template renders correctly in 40 edge cases (empty and very long system prompt, no tools, two tool calls in one message, nested JSON and multi-line arguments, images with tools, two images, earlier reasoning, multi-turn with several tool results) × the reasoning settings (`enable_thinking: false`, `reasoning_effort` low / xhigh): all pass (`prueba_plantillas.py`). With Transformers directly, pass tool-call arguments as dicts (vLLM converts JSON strings itself).

**Runtime**
- Load, long-context needle, vision, tool calls with the real opencode request, 10 agent tasks × 3 runs, concurrency and VRAM: see *Family comparison*; engine errors during the run: 0.

**Scope**: results apply to the revision, software stack (vLLM 0.31.0 + exllamav3 1.5.1 + plugins), hardware and settings stated above; other setups are not tested.


## Limitations

- **n-gram speculative decoding does not start with this model** in vLLM 0.31.0 (`--speculative-config '{"method":"ngram",...}'` stops at start-up with `RuntimeError: PLE inputs were not prepared`, tested on 2026-10-14): use its MTP head, as in the Quick start.
- **Review the commands before they run.** As an agent, the model proposes shell commands and opencode can execute them: review them before running anything as root or with `sudo`, and give the agent a non-privileged account and a test machine where you can.
- Quality was not measured with standard benchmarks (see Qwen's card for those); the checks are KL divergence against BF16, long-context retrieval,
  tool calling with a real agent request, reasoning problems and agent tasks. All numbers come from our machines and one software stack.
- The 48 GB memory figures were measured with our Swift 1.5 build of the same architecture and identical tensor sizes, not with this exact checkpoint.
- The EXL3 MoE path in vLLM is our own plugin code (not upstream).
- This repository has not yet been checked by downloading it on a fresh machine.

## Feedback welcome

Reports of installations, **both working and failed ones**, are very welcome in the [Community](https://huggingface.co/JRamirez-UAB/Qwen3.8-Flash-Next-EXL3-2.25bpw-48GB/discussions)
tab: GPU(s), driver, vLLM version, the full command line and the result (speed, context, or the error).

## Credits & license

- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen, under the **[Qwen Community License 1.0](LICENSE)**: these quantized weights
  are a derivative work and keep that license (Model-as-a-Service / AI Work Assistant businesses need a separate license from Qwen; very large products must
  display the model name). See [NOTICE](NOTICE) for the list of changes.
- [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (quantization, kernels, `qbench` and the public trace; the MTP mixer patch);
  NVIDIA (FP8 PLE table and the NVFP4 build used to generate the calibration trace; not redistributed here); [OrcaSAQ2-kernel](https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel); [vLLM](https://github.com/vllm-project/vllm).
- Our plugin and `scripts/assemble.py` are Apache-2.0.
