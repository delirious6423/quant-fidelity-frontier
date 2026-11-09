---
license: mit
library_name: transformers
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
- es
- en
tags:
- sysadmin
- opencode
- linux
- vllm
- endpoints_compatible
- exl3
- exllamav3
- qtip
- moe
- ornith
- mtp
- long-context
- vision
- int4-kv-cache
- multi-user
- rtx3090
- ampere
---

# Ornith-1.5-35B-A3B — EXL3 3.00 bpw for 24 GB GPUs, 262k context for many users (int4 KV, served with vLLM)

> **Install from this card, not from generic snippets.** This is an EXL3 checkpoint with INT4 token embeddings: vLLM loads it only together with exllamav3 and our `vllm-quantpack` plugin. Follow [Requirements & install](#requirements--install-vllm-031--exllamav3-165--vllm-quantpack-014) and [Quick start](#quick-start-single-rtx-3090-24-gb); a plain `pip install vllm`, Transformers or SGLang cannot load it.

An **unofficial community** EXL3 quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), the agent-oriented 35B MoE with 3B active parameters, made with [exllamav3](https://github.com/turboderp-org/exllamav3) 1.6.5: **3.00 bpw** on the routed experts, 4 bits on attention, linear attention and the shared experts (`-hq 1`), a 6-bit `lm_head`, a 6-bit vision tower, a 4-bit MTP head and **INT4 token embeddings**. The calibration data includes agent steps written by the BF16 model itself on our sysadmin prompts.

It is the **multi-user** sibling of our [EXL3 4.00](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB): 4.1 GB fewer weight bytes leave a much bigger KV cache on the same card. On one **RTX 3090 (24 GB)** with vLLM's int4 KV cache it holds a **1.54 M-token KV pool at the full 262,144-token context**: 6 simultaneous sessions of 236k tokens were measured, and 16 agents at once completed 28 of 32 sysadmin tasks. Its KL divergence against BF16 is about twice that of the 4.00 (0.033 against 0.015), but in our agent and tool tests at 262k it scored the same or better.

> **License:** MIT, inherited from the base model. This is **not** an official release of the Ornith team.

> **Tested setup:** RTX 3090 (24 GB, Ampere, power limit 350 W, driver 595.84), vLLM 0.31.0 + exllamav3 1.6.5 + vllm-quantpack 0.1.3, `--max-model-len 262144`, int4 KV cache (2026-10-29/30). The model was read from our local copy of these files (same bytes, `SHA256SUMS`). Only this setup has been tested.

### Highlights (one RTX 3090 24 GB, full 262,144 context, int4 KV)
- **KLD against BF16:** 0.0332 on an opencode agent trace (ours) and 0.0366 on turboderp's public trace (EXL3 4.00: 0.0145 / 0.0168).
- **Many users** (text, no MTP): KV pool 1,543,326 tokens, 6 real concurrent sessions of 236k tokens with no preemption, 372 MiB free at peak; 4 / 8 / 16 agents at once: 7/8, 16/16, 28/32 tasks.
- **One user with vision and MTP 3:** KV pool 1,191,261 tokens, vision 15/15, decode 148-156 tok/s with one request.
- **Tool calls with the real opencode request:** 59/60 (3 cases × 20) and 28/30 in the battery; **agent tasks** 27/30 (10/10, 9/10, 8/10); **programming tasks** 36/36 (12/12 × 3).
- **Needle at 262k:** 9 neutral facts at 64k / 128k / 255k: 9/9, 9/9, 9/9.
- **Loops:** 0/30 at temperature 0.6, 0.8 and 1.0 on long Spanish answers.
- Decode with 1 / 2 / 4 / 8 requests: 93.7 / 142.6 / 240.9 / 379.3 tok/s (no MTP).

> **Our focus.** We quantize and evaluate models for one use case: a **local Linux system-administration agent** (opencode with tool calling)
> for a university IT department, **used in Spanish**. Our calibration traces include real sysadmin/agent conversations in Spanish, and our quality
> checks (tool calls with the real opencode request, agent tasks with real command execution, long-context retrieval, long answers in Spanish) reflect
> that use. KL divergence is also reported on turboderp's public trace, so it can be compared and reproduced. Numbers for other domains or languages may differ.

## What is in this repository

### Which one to use

| GPU | Repo | Context per session | Real concurrent sessions | KV cache | Speculative decoding | Pick it for |
|---|---|---|---|---|---|---|
| 24 GB (RTX 3090) | **this repo**, *many users* | 262,144 (text) | 6 × 236k measured, `--max-num-seqs 16` | int4, 1.54 M tokens | none | a shared server: many agents on one card |
| 24 GB (RTX 3090) | **this repo**, *one user* | 262,144 with vision | 4 × 262k by vLLM's count (not load-tested) | int4, 1.19 M tokens | MTP 3 | one person, fastest single request, images |
| 24 GB (RTX 3090) | [EXL3 4.00](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB) | 262,144 with vision | 2 | fp8, 287,575 tokens | MTP 1 | the lowest KL divergence on 24 GB |
| 16 GB | — | — | — | — | — | does not fit: 14.82 GB of weights |

### Files and reproducibility

| File | What it is |
|---|---|
| `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, `model.safetensors.index.json` | EXL3 weights (14.82 GB) and their index |
| `quantization_config.json` | exllamav3's per-tensor storage record (bits per tensor, INT4 embedding layout; calibration size only, no text) |
| `config.json`, `generation_config.json` | model config; sampling defaults (temperature 1.0, top_k 20, top_p 0.95) |
| `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | chat template and tokenizer of the base model |
| `preprocessor_config.json`, `processor_config.json`, `video_preprocessor_config.json` | image/video processor of the base model |
| `SHA256SUMS` | SHA-256 of every file above |

**Not included:** the BF16 reference, the calibration data (internal), the plugin (install it from PyPI, see below).

## Results

All tables: one RTX 3090 24 GB (power limit 350 W, no other use), vLLM 0.31.0 + exllamav3 1.6.5 + vllm-quantpack 0.1.3, `--max-model-len 262144`, `--kv-cache-dtype int4_per_token_head`, measured on 2026-10-29/30 with the *many users* Quick start unless stated otherwise.

### Quantization quality (exllamav3 `qbench`, against BF16)

| Trace | KLD mean | KLD median | KLD p90 | Perplexity (BF16) | EXL3 4.00 on the same trace: KLD mean · perplexity |
|---|---|---|---|---|---|
| opencode agent trace (ours, 24 conversations) | 0.0332 | 0.0116 | 0.0798 | 3.4871 (3.5005) | 0.0145 · 3.5054 |
| public (`qbench_prompts_gen.json`, turboderp) | 0.0366 | 0.0061 | 0.0976 | 2.1463 (2.1414) | 0.0168 · 2.1454 |

The INT4 embedding costs little: the same conversion with BF16 embeddings measured 0.0315 / 0.0349.

### Capacity on one RTX 3090 (24 GB)

| Configuration | `--kv-cache-memory-bytes` | KV pool reported by vLLM | Real concurrent sessions | Peak used / limit |
|---|---|---|---|---|
| many users: text, no MTP, 16 sequences, 8192-token chunks | 9015908106 | 1,543,326 tokens | 6 sessions of 235,929 tokens at once, 0 preemptions | 23,744 MiB, 372 MiB free |
| one user: vision, MTP 3, 4 sequences | 8680091156 | 1,191,261 tokens (4.54 × 262,144) | not load-tested | 23,586 MiB (limit 23,826) |

With 9,405,978,378 bytes the card ran out of memory (8 MiB short), so the *many users* value is the largest that kept ≥300 MiB free.

### Test battery (many users configuration)

| Test | Result |
|---|---|
| Tool calls, real opencode request, 3 cases × 20 (`xhigh`) | 59/60 (20/20, 20/20, 19/20), 0 invalid calls |
| Tool calls, battery | 28/30 valid and correct, 0 errors |
| Agent tasks, `opencode run`, 10 sysadmin tasks × 3 runs | 10/10, 9/10, 8/10 = 27/30 |
| Programming tasks (12 × 3) | 12/12, 12/12, 12/12 (589-790 s per run); repeat run 33/36 |
| Needle, 9 neutral facts | 64k 9/9 · 128k 9/9 · 255k 9/9 (fp8 KV: 9/9, 8/9, 9/9) |
| Needle, 8 facts in one prompt | 64k 8/8 · 128k 8/8 · 241k 8/8 |
| Decode, 1 / 2 / 4 / 8 requests | 93.7 / 142.6 / 240.9 / 379.3 tok/s |
| Prefill / decode near 15k · 64k · 128k · 260k | 4,395 / 91.0 · 3,730 / 79.9 · 2,843 / 69.1 · 1,906 / 54.2 tok/s |
| Agents at once (tasks passed, median time per task) | 4: 7/8, 62 s · 8: 16/16, 118 s · 16: 28/32, 242 s (1 timeout) |
| Time to first token of a new 32k prompt while 16 agents run | median 8.1 s |

**Request fields with opencode** (36 real agent conversations, 286 requests, 244 tool calls): 36/36 tasks, 0 lost request fields, 0 malformed calls (JSON, names, required arguments, types, enums), 0 protocol tags in the content, reasoning always separated.

### One user with vision and MTP (one-user configuration, two runs)

| Test | Run 1 | Run 2 |
|---|---|---|
| Vision (our 4-image test) | 15/15 | 15/15 |
| Tool calls, battery | 27/30 | 29/30 |
| Programming tasks | 11/12 | 12/12 |
| Decode, 1 / 2 / 4 / 8 requests | 148.0 / 214.2 / 357.2 / 354.8 tok/s | 156.1 / 205.7 / 344.3 / 345.6 tok/s |
| MTP tokens accepted per draft (short / 258k prompt) | 2.08 / 2.41 | 1.82 / 2.40 |
| Peak VRAM used | 23,586 MiB | 23,566 MiB |

### Real-world deployment: 30-hour university hackathon (8 × RTX 3090)

These exact weights (same SHA-256) served the LLM API of a university hackathon in November 2026: 30.5 hours, 43 of 88 teams used it, mostly
from coding agents (opencode, pi and custom harnesses) with long prompts.

**Goals:** one OpenAI-compatible endpoint for every team, full 262,144-token context, no failed requests, and enough sessions per GPU for
agents with long prompts, on consumer GPUs only.

**Setup:** 8 × RTX 3090 (24 GB), one vLLM 0.31 + vllm-quantpack 0.1.4 server per GPU (`--max-model-len 262144`, `--kv-cache-dtype
int4_per_token_head`, `--max-num-seqs 16`, `--max-num-batched-tokens 8192`, prefix caching, `--reasoning-parser qwen3`, tool calls on),
behind an nginx/Lua load balancer that keeps a conversation on the GPU that holds its prefix cache.

| Result | Value |
|---|---|
| Requests | 30,845 chat requests; 699 M prompt tokens (82 % from the prefix cache), 34.3 M generated |
| Errors | **0 server errors (5xx)**; 187 requests cut by the client, 4 retries after a backend restart, invisible to the user |
| Time to first token (streaming requests, real traffic) | median 2.1 s, p90 5.3 s, p99 31 s |
| Generation speed per request | median 79 tok/s, p95 128 tok/s |
| Prompt / answer size | median 3,291 / 376 tokens; p90 94k / 2,066 tokens |
| Load | peak 585 requests in 5 minutes, 34 concurrent requests, ~1,480 tok/s generated across the 8 GPUs; KV cache at most 18 % used |
| Efficiency | 109k generated tokens per GPU-hour; mean GPU utilisation 30 %, ~120 W per GPU |
| VRAM | 238–260 MiB free at the real-traffic peak |

**MTP measured on the live service** (tok/s per request, the hackathon configuration):

| Speculation | 1 request | 4 at once | Draft acceptance | Decision |
|---|---|---|---|---|
| None | 93 | 61 | – | first day |
| MTP 1 | 104–122 | 72–85 | 77–82 % | tested |
| **MTP 2** | **143–145** | **95** | 68 % | **kept** (KV pool 8,135,104,010 bytes = 1,169,300 tokens) |
| MTP 3 | +3–5 % over MTP 2 | – | 57 % | discarded |

Prefill with MTP 2: 32k tokens in 7–8 s, 128k tokens in 45.9 s.

**Decisions and lessons:**
- **MTP 2 on all GPUs** halfway through the event (+55 % per request over no MTP), with the KV pool trimmed so that ≥200 MiB stayed free at
  the real peak; the larger pool (+1,100 MiB) ran one GPU out of memory under real traffic.
- **Cache-aware routing:** new conversations go to the GPU with fewest requests in flight, later turns stay on the GPU holding their
  prefix; a turn moves only when the extra prefill is cheaper than waiting. A replay of the real trace cut the mean latency from 9.9 s to
  9.2 s (p99 110 → 102 s).
- **Draining before restarts:** taking a GPU out of the balancer before restarting it avoided cut requests; the few cuts happened when we
  restarted without draining.
- **Oversized KV cache** for this traffic (18 % peak): next time, a smaller pool or more sequences per GPU.
- **Reasoning counts towards `max_tokens`:** clients with a small `max_tokens` got empty answers; disable reasoning with
  `chat_template_kwargs: {"enable_thinking": false}` (the template ignores `/no_think`) or allow ≥ 2,000 tokens.

### KV cache type (same card, same 262k pool budget)

| KV cache | Programming tasks (12 × 3) | Prefill 258k | NLL against fp8 up to 262k |
|---|---|---|---|
| int4 (`int4_per_token_head`) | 35/36 and 33/36 | 1,684 tok/s | within the fp8 run-to-run noise |
| fp8 (`fp8_e4m3`) | 33/36 and 34/36 | 1,530 tok/s | — |

int4 halves the KV bytes of fp8 at no measurable cost for this model, which is why both Quick starts use it.

## Sampling and repetition loops

Many users configuration, 5 long Spanish sysadmin questions × 6 runs per temperature, up to 16,000 tokens, top_p 0.95 and top_k 20; a loop = more than 30 % repeated word 8-grams.

| Temperature | 0.6 | 0.8 | 1.0 |
|---|---|---|---|
| Loops / 30 | 0 | 0 | 0 |
| Answers that finished before 16,000 tokens / 30 | 28 | 29 | 28 |
| Empty answers / 30 | 1 | 0 | 0 |

No loops at any temperature tested. This repo's default (`generation_config.json`) is **1.0**, the value we recommend.

## Known model behaviour and discarded options

- **Model or quantization?** The KL divergence is about twice that of our 4.00 build, but the agent results do not show it: on the same card at 262k the 4.00 scored 35/36 and 33/36 in the programming tasks and 34/36 in the request-field check, against 36/36 and 36/36 here. With sampling on, the same checkpoint varies by about ±2 of 36 between runs.
- **File names:** like other Ornith models it sometimes renames *output* files whose Spanish name is a system concept (writes `errors.txt` when asked for `errores.txt`). Writing the names between backticks fixes it.
- **Runaway write:** in one of 801 tool calls across our captures (fp8 KV, another card) a file write degenerated into repetition until the 32,000-token output limit. A lower `max_tokens` bounds the cost.
- **Discarded options:**
  - **MTP for many users:** at 262k with MTP 1 the single request is faster (109 against 91 tok/s), but under load it did not help on the 3090, and it costs KV (1.24 M tokens instead of 1.54 M). It is used only in the one-user configuration.
  - **More than 16 sequences:** with 24 the engine died under load (`EngineDeadError`).
  - **`turboquant_k8v4` KV:** 24/36 programming tasks. `turboquant_4bit_nc` and `turboquant_k3v4_nc` matched int4 (35/36, 34/36); `k3v4` gives slightly more tokens (1.61 M) and 31/32 with 16 agents in a single run; it was not repeated, so int4 stays the default.
  - **2.50 bpw:** a bigger pool (1.87 M tokens) and the same tasks, but KLD 0.059 / 0.065. It is below our 3-bit floor for routed experts, so it is not published.

## Requirements & install (vLLM 0.31 + exllamav3 1.6.5 + vllm-quantpack 0.1.4)

**Before you install** (the same stack as our other cards, checked on a clean Ubuntu 24.04 host on 2026-10-28):
- An NVIDIA driver that supports CUDA 13.2 (tested with 595.84) and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- The Python 3.12 headers and a C compiler: Triton compiles a small C helper every time vLLM starts; without them every `vllm serve` stops with
  `fatal error: Python.h: No such file or directory`. On Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y python3.12-dev build-essential`.
- No CUDA toolkit is needed: the FlashInfer line below installs prebuilt kernels.
- Disk: about 20 GB for the venv, as much again for uv's cache (`uv cache clean` frees it), plus 15 GB for the model.

EXL3 is not a native vLLM format. **[vllm-quantpack](https://github.com/jramirez-uab/vllm-quantpack)** (ours, on PyPI) loads it through vLLM's plugin entry point: EXL3 linear, embedding (INT4/INT8) and MoE layers through exllamav3's kernels, the vision tower, the MTP head and a fixed, faster int4 KV cache. **Version 0.1.3 or later is required** (INT4 embeddings and the int4 KV fixes).

```bash
uv venv --python 3.12 venv
uv pip install --python venv/bin/python vllm==0.31.0 --torch-backend=auto        # torch 2.13.0
uv pip install --python venv/bin/python flashinfer-cubin==0.7.0.post1 flashinfer-jit-cache==0.7.0.post1+cu130 \
  --extra-index-url https://flashinfer.ai/whl/ --extra-index-url https://flashinfer.ai/whl/cu130/ --index-strategy unsafe-best-match   # FlashInfer kernels
W=exllamav3-1.6.5+cu132.torch2.13.0-cp312-cp312-linux_x86_64.whl
curl -L -o /tmp/$W "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.5/${W/+/%2B}"
uv pip install --python venv/bin/python /tmp/$W "vllm-quantpack>=0.1.4"
venv/bin/python -c "import torch; from exllamav3.ext import exllamav3_ext; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## Quick start (single RTX 3090, 24 GB)

**Many users** (the configuration of every table above except the one-user block; text only, no MTP):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
vllm serve JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-3.00bpw-24GB \
  --served-model-name ornith \
  --language-model-only \
  --max-model-len 262144 \
  --kv-cache-dtype int4_per_token_head \
  --kv-cache-memory-bytes 9015908106 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**One user with vision and MTP 3:**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
vllm serve JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-3.00bpw-24GB \
  --served-model-name ornith \
  --max-model-len 262144 \
  --kv-cache-dtype int4_per_token_head \
  --kv-cache-memory-bytes 8680091156 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

On a card that also drives a display, lower `--kv-cache-memory-bytes` by what the desktop uses. On 48 GB GPUs raise it: the KV of this model is ≈5.8 KB per token in int4 (only 10 of the 40 layers are full attention, with 2 KV heads).

## Using it with opencode (how we tested)

`limit.context` must match `--max-model-len` (262144 above).

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "local vLLM",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "ornith": {
          "name": "Ornith-1.5-35B-A3B EXL3 3.00 (local)",
          "tool_call": true,
          "temperature": true,
          "limit": { "context": 262144, "output": 32768 }
        }
      }
    }
  },
  "agent": {
    "build": { "model": "local-vllm/ornith", "temperature": 1.0 },
    "plan":  { "model": "local-vllm/ornith", "temperature": 1.0 }
  }
}
```

- **Declare `"temperature": true` in the model**, otherwise opencode sends no temperature to a custom model.
- With `output: 32768` opencode compacts the history at about `context − 32k` tokens of input, before vLLM rejects the request.
- The agent and tool tests in *Results* ran opencode 1.18.35 against the *many users* server with this configuration.

## Important notes

- **`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` is required.** The fused exllamav3 MoE kernel needs the whole GPU; otherwise vLLM runs the shared expert on a parallel CUDA stream and decoding hangs. The plugin sets it, but export it yourself too.
- **Memory.** The driver reserves about 453 MiB of a 24 GB card that vLLM cannot use: read the real headroom with `nvidia-smi --query-gpu=memory.free`. Size the KV cache with `--kv-cache-memory-bytes`, not `--gpu-memory-utilization`.
- **Prefix caching matters** for agents: opencode sends the same tools and system prompt (about 9k tokens) in every request, and each turn repeats the whole conversation, so keep each conversation on the same server when you run several.
- **Long prompts:** the first turn of a 258k-token prompt takes about 2.5 minutes to prefill on a 3090; later turns reuse the cache.
- **Hugging Face page warnings:** the "Model size" shown by Hugging Face counts packed storage elements, not parameters; the *Configuration Parsing Warning* about `quantization_config.bits` not being an integer is expected (EXL3 `bits` is an average).
- No tensor or expert parallelism for the EXL3 MoE path: one GPU per instance.
- Use `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3`.

## Quantization recipe

| Step | Setting |
|---|---|
| Source | `ornith-ai/Ornith-1.5-35B-A3B` revision `10fbf86f`; chat template and tokenizer unchanged |
| Tool | exllamav3 1.6.5 `convert.py`, RTX PRO 6000 Blackwell, about 2 h 10 min |
| Routed experts | 3.00 bpw (`-b 3.00`) |
| Attention, linear attention, shared experts | 4 bits (`-hq 1`) |
| `lm_head` / vision tower / MTP head | 6 / 6 / 4 bits |
| Calibration | 592 rows built from 594 agent steps written by the BF16 model on our internal sysadmin prompts (515 with tool calls) plus real conversation prefixes; not published, and no text is stored in the weights |
| Token embeddings | INT4 per row with BF16 scales: `python -m vllm_quantpack.tools.embed_quant SRC DST --bits 4` on the converted checkpoint |

## Integrity and validation

**Artifact integrity**
- `SHA256SUMS` lists every file; the verifier compares the repo with our local copy byte for byte before publication.
- The safetensors files carry no `__metadata__`; `quantization_config.json` holds only bit widths, tensor layout and the calibration size (rows and columns).

**Quantization**
- KL divergence against the BF16 model with exllamav3's `qbench` on two traces (*Results*).

**Chat template**
- The base model's template; the request-field check above rendered 664 real opencode requests and found every system, user, tool, reasoning and tool-call field in the prompt in order.

**Runtime**
- Load, 262k needle, capacity, tool calls with the real opencode request, agent and programming tasks, loops and the request-field check: see *Results*; engine errors during the runs: 0.

**Scope:** results apply to the stack (vLLM 0.31.0 + exllamav3 1.6.5 + vllm-quantpack 0.1.3), the RTX 3090 and the settings stated above; other setups are not tested.

## Limitations

- **Review the commands before they run.** As an agent, the model proposes shell commands and opencode can execute them: give it a non-privileged account and a test machine where you can.
- Will not load with stock vLLM, Transformers or SGLang: it needs exllamav3 and vllm-quantpack.
- Not measured: reasoning problems at `xhigh`, a detailed vision table (only our 4-image test), concurrency in the one-user configuration, GPUs other than the RTX 3090 for serving. The calibration and the opencode trace are internal, so those KLD numbers cannot be reproduced outside; the public-trace KLD can.
- The EXL3 MoE path and the int4 KV fixes are our own plugin code, not upstream vLLM.

## Feedback welcome

Reports of installations, **both working and failed ones**, are very welcome in the [Community](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-3.00bpw-24GB/discussions) tab: GPU(s), driver, vLLM version, the full command line and the result (speed, context, or the error).

## Credits & license

- Base model: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (MIT).
- [exllamav3](https://github.com/turboderp-org/exllamav3) by turboderp (quantization, kernels, `qbench` and the public trace); EXL3 is based on [QTIP](https://arxiv.org/abs/2406.11235) (Tseng et al.).
- [vLLM](https://github.com/vllm-project/vllm); [vllm-quantpack](https://github.com/jramirez-uab/vllm-quantpack) (ours, Apache-2.0).
- This quantization is released under the same license as the base model (MIT).
