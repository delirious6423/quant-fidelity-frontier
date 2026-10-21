---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - qwen3.8
  - nvfp4
  - quantized
  - speculative-decoding
  - dflash
  - rtx5090
---

# Qwen3.8-27B: NVFP4 + quantized `lm_head` + DFLASH2 on one RTX 5090

> Recipe to run Qwen3.8-27B on a single RTX 5090 (32 GB): NVFP4 model (Q6+ equivalent quality) using DFLASH2 speculative decoding and a quantized `lm_head`. Ready-to-download checkpoints, no build steps.

**Reasoning: xhigh, hard caps 16k think / 8k content.** Deep thinking every turn (froggeric `Qwen-Fixed-Chat-Templates` v22.4), bounded so an agent turn (Hermes, OpenCode) always produces an answer, never empty content.

Two artifacts (Files tab, or `hf download <repo>`): `Qwen3.8-27B-NVFP4-RTX5090-LMHead4` (main, ~18 GB) and `Qwen3.8-27B-DFlash2-NVFP4-modelopt` (draft, ~1.8 GB). The serve command expects them at `/models/...`.

## Results (RTX 5090 32 GB, sglang nightly `20621aa1`)

**Throughput** (450-token decode, no thinking):

| Workload | single | parallel x2 |
|---|---|---|
| prose | ~144 t/s | ~261 t/s |
| code | ~256 t/s | ~451 t/s |

**Prefill** (time to first token, through the admission proxy; mean of 3 cold runs):

| Prompt length | TTFT | avg t/s | final 1s t/s |
|---|---|---|---|
| 5k | 0.33 s | ~15.2k | (overhead-dominated) |
| 10k | 0.37 s | ~27.3k | ~26.0k |
| 20k | 0.91 s | ~22.1k | ~10.8k |
| 50k | 4.08 s | ~12.3k | ~7.3k |
| 100k | 11.85 s | ~8.4k | ~5.2k |
| 150k | 23.46 s | ~6.4k | ~4.3k |

`avg t/s` = prompt length ÷ TTFT (average prefill rate over the whole prompt, including admission + first-token overhead). `final 1s t/s` = the prefill rate in the final second at that prompt length (estimated from the local slope of the TTFT curve); small sizes are dominated by fixed overhead and not reliably resolvable.

**Notable features:**

- 175k KV pool
- Host-RAM KV tier: a ~100k conversation resumes in ~1 s, not ~20 s cold
- xhigh reasoning, hard caps 16k think / 8k content. Tested against higher caps with virtually no change in GPQA scoring.
- Uses the latest froggeric template to improve agentic use
- 4 simultaneous agent conversations (benchmark below)
- Quality: 9/10 objective correctness quiz, identical to the un-quantized head

**External evals** (lm-evaluation-harness, quantized checkpoint as served; `-` = Qwen publishes no 3.8-27B number):

| Benchmark | This stack | Qwen published |
|---|---|---|
| GPQA Diamond (xhigh thinking) | **84.8%** | **89.2%** |
| GSM8K (5-shot) | 96.8% | - |
| MATH-500 (math_verify) | 95.6% | - |
| AIME 2024 | 83.3% | - |
| HumanEval (pass@1) | 56.7% | - |
| MBPP (pass@1) | 75.0% | - |

GPQA was tested at xhigh with the thinking cap raised but the score hovered (85.4% vs 84.8%), so the 16k cap costs virtually nothing and keeps worst-case turns ~15-28 s tighter. Raise it if you wish.

**4-conversation switching** (4 multiturn agent conversations, identical except base size):

| Metric | 60k conversations | 100k conversations |
|---|---|---|
| Peak slots | 2 (parallel) | 1 (serial-jump) |
| Host-RAM restores | ~1.0-1.5 s | ~0.8-1.4 s |
| 20 turns total | ~129 s | ~240 s |

Two 60k conversations fit the pool and run in parallel; at 100k only one fits, so they take turns, each resuming from RAM in ~1 s. Every conversation looks like it has a dedicated 100k context.

## Serve

```bash
docker run -d --name sglang-qwen38 --restart unless-stopped \
  --gpus '"device=1"' --ipc host \
  -v /models:/models \
  -e SGLANG_MAX_THINK_TOKENS=16000 \
  -e NCCL_BUFFSIZE=2097152 \
  lmsysorg/sglang:nightly-dev-cu13-20260827-20621aa1 \
  python3 -m sglang.launch_server \
  --model-path /models/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090-LMHead4 \
  --host 0.0.0.0 --port 11403 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-strict-thinking \
  --chat-template /models/qwen_chat_template/chat_template.jinja \
  --quantization modelopt_fp4 --trust-remote-code \
  --mem-fraction-static 0.94 --max-total-tokens 175064 \
  --attention-backend flashinfer --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs-decode 2 --disable-prefill-cuda-graph \
  --max-mamba-cache-size 10 --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer --radix-eviction-policy slru \
  --disable-overlap-schedule --page-size 64 --schedule-policy lpm \
  --enable-hierarchical-cache --hicache-ratio 4.0 \
  --hicache-mem-layout page_first --hicache-write-policy write_through_selective \
  --hicache-io-backend kernel \
  --watchdog-timeout 1200 --chunked-prefill-size 4096 --max-running-requests 2 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /models/incoai/Qwen3.8-27B-DFlash2-NVFP4-modelopt \
  --speculative-draft-model-quantization modelopt_fp4 --speculative-dflash-block-size 6 \
  --served-model-name Qwen3.8-27B \
  --preferred-sampling-params '{"max_new_tokens": 24000}' \
  --language-only --warmups=prefill_shapes
```

Requires the sglang nightly image (v0.5.18 lacks DFLASH2-draft + quantized-head support): pin `lmsysorg/sglang:nightly-dev-cu13-20260827-20621aa1`.

## Tune to your liking

| Knob | Default | Tradeoff |
|---|---|---|
| `--speculative-dflash-block-size` | 6 | 6 is the measured throughput peak; lower = less draft VRAM, 9-27% slower |
| `--speculative-num-draft-tokens` | 5 | 12 eats ~75k of context pool; lower frees VRAM |
| `--max-total-tokens` | 175064 | KV pool; step down if your GPU has less headroom |
| `--mem-fraction-static` | 0.94 | 0.95/0.96 OOM on transient allocations |
| `--language-only` | on | Vision: drop it and set the pool to 150064 (~0.9 GB encoder) |

## Where the gains came from

- Quantized `lm_head` (-1.7 GB, paid for the bigger pool)
- DFLASH2 draft re-quantized to modelopt-NVFP4 (upstream doesn't load in sglang)
- Block 6, NCCL buffer 2 MiB, fp8 KV (more room for more KV)
- GPU-managed host-RAM KV tier (`--hicache-io-backend kernel`): the GPU does the RAM copies, so a spilled ~100k conversation restores in ~1 s, not ~20 s cold
- froggeric chat template + capping strict thinking (no runaway reasoning, no empty content)

## Bonus pro tip: put a request gate in front of sglang

This is the one thing I'd add on top of the serve command. The problem: a big request queued ahead of several small ones wastes parallel capacity. While the big one holds a slot, the small ones wait even when the budget has room for another. I proxy all `/v1/*` traffic through a small admission proxy that tokenizes each prompt and only admits it if it fits the KV pool. A small request that fits the leftover budget gets admitted in parallel instead of waiting behind the big one, and anything that can never complete gets a clean 400 up front. sglang only gates by request count (`--max-running-requests`), not KV budget; the gate fills that gap.

Even if you don't end up using it, an admission layer is worth it for any provider, sglang, llama.cpp, or vLLM; they all gate by count, not KV budget.

Paste this into your agent and it'll build you one:

```text
Build me a small FastAPI admission-control proxy to put in front of an sglang server.

Requirements:
1. Proxy every `/v1/*` request verbatim to the upstream sglang URL (configurable), streaming responses back.
2. For POST /v1/chat/completions, before forwarding, call the upstream POST /tokenize with {"messages": <the messages array>} to get the exact prompt token count (includes chat-template framing).
3. Fetch GET /server_info on startup and on a 30s timer for max_total_num_tokens and max_running_requests.
4. Admission: admit a request when an sglang slot is free AND its projected KV fits the pool. Projected in-use = sum over active requests of (prompt − shared radix prefix + output reservation), where a conversation continuation shares its prefix with the active request it extends.
5. Output reservation = min(client max_tokens or 24000, 4096); sglang's own scheduler only charges up to 4096 (SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION).
6. If prompt + the client's full output ceiling (24000 default) exceeds the pool, return HTTP 400 context_overflow up front instead of admitting.
7. If a request doesn't fit, queue it (asyncio.Condition). On every release and before each new admission, drain the queue FIFO: admit every queued request that now fits, bypassing those that can't. After a queued request has been bypassed 3 times, or has waited 20s, make it a barrier: nothing behind it may be admitted until it fits (prevents starvation by a stream of small requests).
8. Vision: if any message has an image_url, charge estimated image tokens from the pixel dimensions (Qwen2VL grid formula, 28px factor, ~2048 tokens at 1080p) on top of the /tokenize count.
9. Clean up reliably: if the client disconnects or the request is rejected while queued or admitted, release its slot (idempotent).
10. Expose /health, /status, and /metrics (Prometheus) with a gauge for current queue depth.

Config via env vars: upstream sglang URL, output reserve cap (default 4096), starve skips (default 3), starve seconds (default 20).
Write it as a single main.py using only fastapi, uvicorn, httpx, prometheus-client. Include a Dockerfile.
```

## Thanks

- [gittensor-model-hub](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) (NVFP4 base checkpoint)
- [incoai](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (DFlash2 draft)
- [calneymgp](https://huggingface.co/calneymgp/Qwen3.8-27B-NVFP4-lmhead4-recipe) (lm_head quantization recipe)
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B) (base model)
- [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (quantizer)
- [sglang](https://github.com/sgl-project/sglang) (serving engine)
- [froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (chat template)

Re-quantizations of open checkpoints. All Apache-2.0.