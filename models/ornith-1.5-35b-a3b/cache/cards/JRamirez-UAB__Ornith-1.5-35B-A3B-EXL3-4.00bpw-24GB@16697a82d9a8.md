---
language:
- es
- en
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
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
- ornith
- mtp
- speculative-decoding
- long-context
- vision
- rtx3090
- ampere
- endpoints_compatible
---

# Ornith-1.5-35B-A3B — EXL3 MoE for vLLM, 262k context with vision and MTP on a 24 GB GPU

> **Install from this card, not from generic snippets.** This is an EXL3 checkpoint: vLLM loads it only together with exllamav3 and our plugins. Follow [Requirements & install](#requirements--install-vllm-031--exllamav3-151--plugins) and [Quick start](#quick-start-single-rtx-3090-24-gb); a plain `pip install vllm`, Transformers or SGLang cannot load it.

An EXL3 quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) made with [exllamav3](https://github.com/turboderp-org/exllamav3)'s per-tensor bit-allocation
("SC") pipeline, with a 6-bit `lm_head`, 6-bit vision tower, 4-bit MTP head and INT8 token embeddings. Ornith's agent-oriented 35B-A3B MoE (Terminal-Bench 2.1 67.8 per its card). Of the three MoEs it is the best agent in our tests (tool calling 30/30 in every mode, short reasoning) and the hardest to quantize, so this repo also has 4.50 and 5.00 bpw branches: full 262k on GPUs with more than 24 GB, and less context on a 24 GB card (see Quick start).

It is served with **vLLM** (continuous batching) through **our `vllm-quantpack` plugin, which includes EXL3 support for the MoE experts** (fused exllamav3 kernel). On a single **RTX 3090 (24 GB)** the `main` revision runs the **full 262,144-token context with vision, MTP and
2 concurrent sessions** with ≈1 GB of VRAM to spare.

> **Tested setup:** RTX 3090 (24 GB, Ampere, power limit 350 W), vLLM 0.31.0 + exllamav3 1.5.1 + the OrcaSAQ2 plugin + our `qwen38_exl3_vision` 0.3.1 plugin, weights downloaded from this repo, the Quick start configuration (2026-10-14). Only this setup has been tested.

> **Our focus.** We quantize and evaluate models for one use case: a **local Linux system-administration agent** (opencode with tool calling)
> for a university IT department, **used in Spanish**. Our calibration traces include real sysadmin/agent conversations in Spanish, and our quality
> checks (tool calls with the real opencode request, agent tasks with real command execution, long-context retrieval, long answers in Spanish) reflect
> that use. KL divergence is also reported on turboderp's public trace, so it can be compared and reproduced. Numbers for other domains or languages may differ.

## What is in this repository

### Which one to use

All Ornith-1.5-35B-A3B variants we publish, measured on 2026-10-14 under the same conditions (see *Family comparison* for every number):

| Variant | Weights | Context on one RTX 3090 | KLD vs BF16 | Agent tasks | Decode, 1 request | Pick it for |
|---|---|---|---|---|---|---|
| **[EXL3 4.00](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB) (this repo)** | 17.62 GiB | 262,144 with vision | 0.0163 | 24/30 | 113.8 tok/s | a 24 GB GPU with the full 262,144 context and vision |
| [EXL3 4.50](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB/tree/4.50bpw) (this repo) | 19.57 GiB | 171,008 with vision | 0.0126 | 25/30 | 112.1 tok/s | a 48 GB GPU (on 24 GB only 171,008 tokens) |
| [EXL3 5.00](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB/tree/5.00bpw) (this repo) | 21.53 GiB | 124,928 (text) | 0.0095 | 24/30 | 87.3 tok/s | a 48 GB GPU (on 24 GB only 124,928 tokens, text-only) |

| Branch | Size |
|---|---|
| `main` | 18.9 GB of weights |
| `5.00bpw` | 23.1 GB of weights |
| `4.50bpw` | 21.0 GB of weights |

Download a branch with `hf download JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB --revision <branch> --local-dir <dir>`.

## Results
### Family comparison (Ornith-1.5-35B-A3B, same conditions)

Measured on 2026-10-14 for every variant in the same way: weights downloaded from Hugging Face, one RTX 3090 24 GB (power limit 350 W, no other use), vLLM 0.31.0 (EXL3: + exllamav3 1.5.1 + plugins), each variant with its own Quick start configuration. KLD and perplexity come from exllamav3's `qbench` (see *How to read it*).

| Variant | Weights | KLD mean / median (internal · public) | Perplexity (BF16 3.2275 · pub. 2.1414) | Needle (8 facts) near max. context | Tools: real opencode request | Agent tasks (10 × 3) | Reasoning `xhigh`: correct · tokens | Vision |
|---|---|---|---|---|---|---|---|---|
| **[EXL3 4.00 bpw](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB) (`main`, this repo)** | **17.62 GiB** | **0.0163 / 0.0060 · pub. 0.0168 / 0.0028** | **3.2297 · pub. 2.1454** | **8/8 at 241,437** | **60/60** | **24/30** | **48/48 · 24,411** | **14/15** |
| [EXL3 4.50 bpw](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB/tree/4.50bpw) (`4.50bpw`, this repo) | 19.57 GiB | 0.0126 / 0.0047 · pub. 0.0128 / 0.0022 | 3.2317 · pub. 2.1463 | 8/8 at 157,632 | 60/60 | 25/30 | 48/48 · 25,976 | 15/15 |
| [EXL3 5.00 bpw](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB/tree/5.00bpw) (`5.00bpw`, this repo) | 21.53 GiB | 0.0095 / 0.0033 · pub. 0.0095 / 0.0015 | 3.2323 · pub. 2.1421 | 8/8 at 64,230 | 59/60 | 24/30 | 48/48 · 22,551 | — |

| Variant | Context on one RTX 3090 | Sessions | Prefill near max. context | Decode, 1 / 2 / 8 requests (tok/s) | MTP tokens per step | VRAM really free at peak |
|---|---|---|---|---|---|---|
| EXL3 4.00 | 262,144 with vision | 2 | 1,283 tok/s | 113.8 / 198.4 / 199.2 | 1.82 | 1,072 MiB |
| EXL3 4.50 | 171,008 with vision | 2 | 1,660 tok/s | 112.1 / 191.8 / 195.7 | 1.83 | 330 MiB |
| EXL3 5.00 | 124,928 (text) | 2 | 2,523 tok/s | 87.3 / 138.4 / 141.2 | no MTP | 402 MiB |

How to read it:
- **KLD / perplexity**: next-token KL divergence against the BF16 model with exllamav3's `qbench` (lower is better). *Internal*: our sysadmin/agent evaluation set (not published, separate from the calibration data); *public*: turboderp's `qbench_prompts_gen.json`, reproducible by anyone. GPTQ checkpoints were evaluated as a BF16 copy holding the dequantized weights. These two columns come from `qbench`, not from this campaign.
- **Needle**: 8 facts spread through one prompt at about 92 % of the served context; all 8 must come back. **Tools**: the real request of the opencode agent (system prompt + 11 tools), 3 cases × 20 tries, at the model's default sampling (`generation_config.json`); a try counts when it calls a valid tool with valid arguments and the expected command. **Agent**: 10 sysadmin tasks run with `opencode run` and real command execution, 3 runs, at the recommended temperature; with sampling on, the same checkpoint varies by about ±2 of 30 between runs, so other measurements in this card may differ by that much. **Reasoning**: 6 checkable problems × 8 at `xhigh`; total reasoning tokens over the 48. **Vision**: our 4-image test.
- **Prefill / decode**: prefill from the needle prompt; decode is aggregate throughput with 1 / 2 / 8 simultaneous requests (temperature 0; with more requests than *Sessions*, the extra ones wait). **MTP tokens per step**: vLLM's speculative-decoding metrics over the whole run.
- **VRAM really free**: lowest `memory.free` while serving (nvidia-smi every 0.5 s, from the end of start-up to the end of the run); we require ≥100 MiB.
- Older sections below keep their own date and setup; use this table to compare variants.

**How many sessions fit** with each Quick start configuration (KV size in tokens as reported by vLLM at start-up; a session uses the KV of its own context, and vLLM never runs more than *max sessions* at once):

| Variant | KV pool (tokens) | Max sessions | 32k each | 64k each | 128k each | Full context each |
|---|---|---|---|---|---|---|
| EXL3 4.00 | 287,575 | 2 | 2 | 2 | 2 | 1 × 262,144 |
| EXL3 4.50 | 172,908 | 2 | 2 | 2 | 1 | 1 × 171,008 |
| EXL3 5.00 | 126,820 | 2 | 2 | 1 | 0 | 1 × 124,928 |

**Few users or many users.** The Quick start configuration is tuned for 1–2 simultaneous users (fastest single request). For more users, the *many users* option below (measured the same day, same GPU) gives more total throughput; each request gets slower when many run at once. Same model, same quality: only the serving settings change.

| Variant | Configuration | Decode by simultaneous requests (aggregate tok/s) | N sessions at once, needle 8/8 | Tools (5 × 3) | VRAM really free |
|---|---|---|---|---|---|
| EXL3 4.00 | Quick start (2 sessions, MTP 1) | 1: 114 · 2: 198 · 8: 199 | — | 60/60 | 1,072 MiB |
| EXL3 4.00 | many users (32 sessions of up to 262,144 tokens, MTP 1, CUDA graphs up to 64 tokens) | 1: 114 · 4: 328 · 8: 510 · 16: 672 · 32: 636 | 16/16 of 12k | 15/15 | 794 MiB |
| EXL3 4.50 | Quick start (2 sessions, MTP 1) | 1: 112 · 2: 192 · 8: 196 | — | 60/60 | 330 MiB |
| EXL3 4.50 | many users (32 sessions of up to 171,008 tokens, MTP 1, CUDA graphs up to 64 tokens) | 1: 115 · 4: 320 · 8: 500 · 16: 469 · 32: 526 | 16/16 of 12k | 14/15 | 154 MiB |

<!-- end of the controlled comparison -->

### Earlier measurements

_Earlier measurements, before the controlled comparison of 2026-10-14: other dates, vLLM versions or settings, as stated in each table. Use *Family comparison* above to compare variants._


| Revision | bpw | Weights | KLD internal · public | GPU | Context | Decode with MTP, 1 / 2 sessions (tok/s) | Prefill 62k → 237k (tok/s) | Needle | Vision | Tools A · B · C (of 30) | Reasoning `xhigh` (48 problems) | Really free at peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`main`** | 4.00 | 17.62 GiB | 0.0163 · 0.0168 | RTX 3090 | 262,144 | 113.8 / 195.7 | 2,441 → 1,255 | 8/8 at 237k | 15/15 | **30 · 30 · 30** | 24,289 (47/48) | 1,063 MiB |
| `4.50bpw` | 4.50 | 19.57 GiB | 0.0126 · 0.0128 | L40S 48 GB | 262,144 | 119.2 / 233.2 | 3,524 → 2,649 | 8/8 at 237k | 15/15 | 30 · 30 · 29 | 25,651 (45/48) | 13,988 MiB (KV 10e9) |
| `5.00bpw` | 5.00 | 21.53 GiB | 0.0095 · 0.0095 | L40S 48 GB | 262,144 | 118.9 / 220.9 | 3,457 → 2,657 | 8/8 at 237k | 14/15 | **30 · 30 · 30** | 26,417 (48/48) | 12,006 MiB (KV 10e9) |

How to read it:
- **KLD**: mean KL divergence of the next-token distribution against the BF16 model, measured with exllamav3's `qbench` on an internal sysadmin/agent evaluation
  set (not published) and on turboderp's public trace (`qbench_prompts_gen.json`). Lower is better.
- **Needle**: 8 facts spread through one prompt, all 8 returned; 2 concurrent sessions of 116k tokens also returned 8/8 each in every row.
- **Tools**: the real request of the opencode agent (system prompt + 11 tools), 3 cases × 10 tries; A = `xhigh` at the model's default temperature, B = `xhigh` at T=0.3,
  C = thinking off at T=0.3. **Reasoning**: total reasoning tokens over 6 checkable problems × 8 tries (T=0.6), and how many were correct.
- **Agent** (`opencode run`, 4 sysadmin tasks × 2 with real tool execution): 6/8 (4.00).
- Configuration of every row: `--max-model-len 262144`, fp8 KV cache, MTP 1, 2 sequences, prefix caching, 1 image per request (≤1 MP). RTX 3090 rows: KV 3.5e9 B (287,575-token pool).

## Sampling and repetition loops

Measured on 2026-10-14 with the same software and GPU as *Family comparison*, on our EXL3 4.00 build of Ornith-1.5-35B-A3B (loops come from the model and the sampling, not from the quantization format): 5 long Spanish sysadmin prompts × 6 runs per temperature, up to 16,000 tokens, top_p 0.95 and top_k 20; a loop = more than 30 % repeated 8-grams. Serving: the Quick start with a 32k context and 8 sessions.

| Temperature | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| Loops / 30 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Answers that finished before 16,000 tokens / 30 | 25 | 27 | 25 | 22 | 28 | 25 | 25 | 23 |

From temperature **0.3** up, loops are rare (at most 1 of 30). This repo's default (`generation_config.json`) is **1.0**.

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
| Qwopus3.6-35B-A3B-Coder · EXL3 4.00 | 9 | 13 | 12 | 6 | 11 | 6 | 5 | 4 |
| Swift 1.5 Qwen3.8-Flash-Next (125B-A6B) · EXL3 2.25 · vLLM | 24 | 24 | 21 | 21 | 17 | 8 | 0 | 0 |
| Qwen3.8-Flash-Next (125B-A6B) · EXL3 2.25 · vLLM | — | — | 13 | 7 | 5 | 2 | 0 | 0 |

## Requirements & install (vLLM 0.31 + exllamav3 1.5.1 + vllm-quantpack)

EXL3 is not a native vLLM format. One plugin is loaded through vLLM's `vllm.general_plugins` entry point:
- **[vllm-quantpack](https://github.com/jramirez-uab/vllm-quantpack)** (ours, on PyPI): EXL3 linear, embedding and MoE layers through exllamav3's kernels, plus the vision tower and the MTP head.

Use **exllamav3 1.5.1** (with 1.6.x the KLD is the same, but tool calls without reasoning got worse in our tests) and Python **3.12** (the exllamav3 wheels
are cp312). flashinfer compiles some kernels on first start with the CUDA toolkit that comes with the vLLM wheel; if your system `nvcc` does not match,
point `CUDA_HOME` at a CUDA 13.0 toolkit.

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python vllm==0.31.0 --torch-backend=auto        # torch 2.13.0
W=exllamav3-1.5.1+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl
curl -L -o /tmp/$W "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.1/${W/+/%2B}"
uv pip install --python venv/bin/python /tmp/$W vllm-quantpack
venv/bin/python -c "import torch; from exllamav3.ext import exllamav3_ext; print(torch.__version__, torch.cuda.get_device_name(0))"
```
> **Plugin (2026-10-28):** [vllm-quantpack](https://github.com/jramirez-uab/vllm-quantpack) (`pip install vllm-quantpack`) replaces the OrcaSAQ2 + `qwen38_exl3_vision` plugins. The measurements in this card were made with the previous stack (OrcaSAQ2 + our plugins); this checkpoint was then served again with vllm-quantpack 0.1.0 following this card on a clean install (text, image and tool calls checked). The previous files stay in `plugin/` for reference.


## Quick start (single RTX 3090, 24 GB)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
vllm serve JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB \
  --served-model-name ornith-1.5-35b-a3b \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1003520}' \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-memory-bytes 3500000000 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 2 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"max_cudagraph_capture_size":16}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

- **`4.50bpw` branch (21.1 GB of weights): does not fit a 24 GB GPU with 262k + vision + MTP.** Validated at 262,144 on an L40S 48 GB with `--revision 4.50bpw` and `--kv-cache-memory-bytes 10000000000` (other flags the same). On a 24 GB RTX 3090 it fits with **171,008 tokens with vision + MTP** (`--kv-cache-memory-bytes 2173912576 --max-model-len 171008`; needle 8/8 at 154k, vision 15/15, 323 MiB really free at peak), 215,040 text-only with MTP (`--language-model-only`, KV 2685617664) or the full 262,144 with vision and no MTP (KV 3063329408).
- **`5.00bpw` branch (23.2 GB of weights): does not fit a 24 GB GPU with 262k + vision + MTP.** Validated at 262,144 on an L40S 48 GB with `--revision 5.00bpw` and `--kv-cache-memory-bytes 10000000000` (other flags the same). On a 24 GB RTX 3090 it does **not** fit with vision + MTP; text-only (`--language-model-only`) it fits with **26,624 tokens with MTP** (KV 565396992) or **124,928 without MTP** (KV 1456910976, no `--speculative-config`); needle 8/8 at 90 % of each.

On 48 GB GPUs raise `--kv-cache-memory-bytes` (10e9 B gives an 821k-token pool: room for 8 sessions; add `--max-num-seqs 8`). For more sessions, raise `max_cudagraph_capture_size` with them
(see *Important notes*: ≥ `max-num-seqs` × 2 with MTP 1).

**For many users** (measured on 2026-10-14, same GPU): the same command with `--max-num-seqs 32 --compilation-config '{"max_cudagraph_capture_size":64}'` — 32 sessions of up to 262,144 tokens, MTP 1, CUDA graphs up to 64 tokens. Aggregate decode 636 tok/s with 32 requests (Quick start: 199 with 8); a single request runs at 114 tok/s (114 with the Quick start); 16 of 16 simultaneous sessions returned all 8 needle facts; 794 MiB of VRAM really free. Rule for CUDA graphs: they must cover sessions × (1 + MTP) tokens per step.

## Using it with opencode

How we use it as a **coding / sysadmin agent from [opencode](https://opencode.ai)** (opencode 1.18.x). `limit.context` must match `--max-model-len` of your
`vllm serve` command (262144 in the Quick start above).

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local vLLM",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "ornith-1-5-35b-a3b-exl3-4-00bpw-24gb": {
          "name": "Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB (local)",
          "tool_call": true,
          "temperature": true,
          "limit": { "context": 262144, "output": 22000 }
        }
      }
    }
  },
  "agent": {
    "build": { "model": "local-vllm/ornith-1-5-35b-a3b-exl3-4-00bpw-24gb", "temperature": 1.0 },
    "plan":  { "model": "local-vllm/ornith-1-5-35b-a3b-exl3-4-00bpw-24gb", "temperature": 1.0 }
  }
}
```

- **Declare `"temperature": true` in the model**, otherwise opencode sends no temperature to a custom model and vLLM falls back to its defaults.
- Temperature 1.0 is this repo's `generation_config.json` default.
- With `output: 22000` opencode compacts the history at ≈`context − 22k` tokens of input.
- Start opencode inside a project directory (it takes a git snapshot before each step); for `opencode run` from scripts, close stdin (`</dev/null`).

## Important notes

### Model-specific notes

**File names:** like other Ornith models it sometimes translates file names written in Spanish in the request (e.g. writes `errors.txt` when asked for `errores.txt`); both failed agent tasks here were that, with the right content. On Ornith-1.5-9B, writing the names between backticks fixed it (16/16). The model is not translating: it renames *output* files whose Spanish name is a system concept (processes, users, errors) to the conventional English name; names unrelated to the system (e.g. `tortuga.txt`) are kept (34/34). Backticks fix it and did not hurt when the user describes a system file in words or misspells its path (70/70). The 4.50/5.00 branches do not fit a 24 GB GPU with 262k + vision + MTP (see the branch notes in Quick start for what does fit on an RTX 3090). On a 24 GB card, measured on the 4.50 branch: **the vision tower costs ≈45k tokens of context and the MTP head ≈112k** (its weights are only 0.4 GB, but it adds its own KV cache, ≈16 % more per token, and ≈0.9 GB of buffers).

### Important notes

- **`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` is required.** The fused exllamav3 MoE kernel synchronises its blocks through global counters and needs the whole GPU;
  vLLM (0.30 and 0.31) otherwise runs the shared expert on a parallel CUDA stream and decoding hangs after a few seconds. The plugin sets it, but export it yourself too.
- **CUDA graphs: `max_cudagraph_capture_size` ≥ `max-num-seqs` × (1 + MTP tokens)**, so that every decode step is replayed from a graph (16 is enough for the
  2 sessions of the Quick start). Plugin **0.2.1** and later (this repo ships **0.3.2**) runs every step of up to 256 tokens in exllamav3's fused MoE kernel; 0.2.0 only did so up to
  32 tokens (8 experts per token × tokens ≤ 256) and fails at start with a larger capture size. RTX 3090, vLLM 0.30, 32 sequences, MTP 1, short prompts:
  0.2.0 (graphs ≤ 32) and 0.2.1 (≤ 128) give the same 106 / 378 / 615 tok/s in total with 1 / 8 / 16 requests at once; with 32 at once, 499 → 587 tok/s.
  Graphs plus MTP 1 matter more than the plugin: with MTP 3 and graphs ≤ 16, 8-32 requests ran 1.7-1.8× slower. KL divergence 0.2.1 vs 0.2.0: 0.0155,
  the same as between two runs of 0.2.0 (0.0152). On 24 GB use at most 128 (256 does not fit while capturing).
- **MTP 1, not 3.** In a MoE every drafted token activates more experts during verification: on a public Ornith EXL3 we measured 84.6 tok/s without MTP,
  93.7 with MTP 1, 70.8 with MTP 2 and 60.9 with MTP 3. MTP 1 accepts 78-89 % of drafts on these checkpoints.
- **Memory.** The driver reserves 453 MiB of a 24 GB card that vLLM cannot use: read the real headroom with `nvidia-smi --query-gpu=memory.free`. Size the KV cache
  with `--kv-cache-memory-bytes`, not `--gpu-memory-utilization`. The KV cache of these models is small (≈13 KB/token in fp8: only 10 of the 40 layers are
  full attention, with 2 KV heads), which is why 262k fits next to the 19 GB of weights.
- **Hugging Face page warnings:** the "Model size" shown by Hugging Face counts packed storage elements, not parameters; the *Configuration Parsing Warning*
  about `quantization_config.bits` not being an integer is expected (EXL3 `bits` is an average).
- No tensor or expert parallelism for the EXL3 MoE path: one GPU per instance.
- Use `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3`. `reasoning_effort` accepts `xhigh`, `medium` and `low`; they are prompt instructions, not limits.

## Quantization recipe

- exllamav3 1.5.1, SC pipeline: 2.00 bpw reference, per-tensor noise measurement (`sc_measure --shaped`, `--max-sys 24`) on a 196 × 2048-token calibration trace,
  bit allocation with `sc_optimize`, conversion with the same trace. The trace was generated by Qwen3.8-27B (same tokenizer) on prompts from exllamav3's SC set
  plus internal Linux sysadmin/agent conversations (not published); with a trace generated by the model itself the KLD was not better (Ornith 4.00: 0.0171 vs 0.0163).
- `lm_head` 6 bits, vision tower 6 bits, MTP head 4 bits, codebook `mul1`, `out_scales: always`.
- Token embeddings: INT8 per row (OrcaSAQ2 format), made with [`scripts/embed_int8.py`](scripts/embed_int8.py) from the converted checkpoint (≈−0.5 GB of VRAM, no measurable change).
- Source: `ornith-ai/Ornith-1.5-35B-A3B` revision `10fbf86f`. Chat template and tokenizer are the base model's.

## Integrity and validation

What was checked for this checkpoint, how, and where the numbers are (measured on 2026-10-14; the method and every figure are in *Family comparison*).

**Artifact integrity**
- Every weight file is checked against our local copy with SHA-256 before publication (`SHA256SUMS` in this repo).
- The measurements used the weights downloaded from this repo at commit `d39c3384e360`; later commits change only metadata, documentation or the plugin, never the `.safetensors` files.
- vLLM loads every tensor listed in `model.safetensors.index.json` on start-up (a missing tensor or a wrong shape would stop it): checked from a fresh download.
- After the metadata update of 2026-10-14 (`quantization_config.json`, plugin 0.3.1, `embed_int8.py`): downloaded again from this repo and re-tested: loads; vision 14/15 needle 64k 8/8; tool calls 15/15; engine errors 0.

**Quantization**
- Per-tensor bit allocation is recorded by exllamav3 in `quantization_config.json`; its `tensor_storage` entry for `embed_tokens` describes the INT8 `qweight` + BF16 `scales` actually stored (updated on 2026-10-14; the plugin only reads the `.trellis` entries).
- The INT8 embedding conversion (`scripts/embed_int8.py`) is reproducible: re-running it on the source checkpoint gives the published `qweight` and `scales` bit for bit (checked on Ornith-1.5-9B 6.00 bpw); its reported error now uses the stored BF16 scales (0.00806 vs 0.00803 with FP32 scales).
- Fidelity: KL divergence against the BF16 Ornith-1.5-35B-A3B with exllamav3's `qbench` on two traces (ours and turboderp's public one).

**Chat template**
- The published template renders correctly in 40 edge cases (empty and very long system prompt, no tools, two tool calls in one message, nested JSON and multi-line arguments, images with tools, two images, earlier reasoning, multi-turn with several tool results) × the reasoning settings (`enable_thinking: false`, `reasoning_effort` low / xhigh): all pass (`prueba_plantillas.py`). With Transformers directly, pass tool-call arguments as dicts (vLLM converts JSON strings itself).

**Runtime**
- Load, long-context needle, vision, tool calls with the real opencode request, 10 agent tasks × 3 runs, concurrency and VRAM: see *Family comparison*; engine errors during the run: 0.
- On a desktop RTX 3090 that also drives the display (300 W): Quick start at 262,144 tokens: 1,067 MiB really free.

**Scope**: results apply to the revision, software stack (vLLM 0.31.0 + exllamav3 1.5.1 + plugins), hardware and settings stated above; other setups are not tested.


## Limitations

- **Review the commands before they run.** As an agent, the model proposes shell commands and opencode can execute them: review them before running anything as root or with `sudo`, and give the agent a non-privileged account and a test machine where you can.
- Quality was not measured with standard benchmarks; the checks are KL divergence against BF16, long-context retrieval, vision, tool calling with a real agent
  request, short reasoning problems and agent tasks. All numbers come from our machines and one software stack.
- The EXL3 MoE path in vLLM is our own plugin code (not upstream); it has been tested only on Ampere and Ada (RTX 3090, A40, L40S) and Blackwell only for the KLD measurements (exllamav3, not vLLM).

## Feedback welcome

Reports of installations, **both working and failed ones**, are very welcome in the [Community](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB/discussions) tab: GPU(s), driver, vLLM version,
the full command line and the result (speed, context, or the error).

## Credits & license

- Base model: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MIT).
- [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (quantization, kernels and the `qbench` evaluation, public trace from turboderp's repos);
  [OrcaSAQ2-kernel](https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel) (EXL3 in vLLM); [vLLM](https://github.com/vllm-project/vllm).
- This quantization, the plugin and the scripts are released under the same license as the base model.
