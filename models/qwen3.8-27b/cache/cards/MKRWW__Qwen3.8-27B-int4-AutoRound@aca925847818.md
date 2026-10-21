---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - auto-round
  - int4
  - gptq
  - vllm
  - quantized
  - qwen3_5
library_name: transformers
---

# Qwen3.8-27B-int4-AutoRound

**4-bit (AutoRound) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — the full Omni model in ~18 GB, so it runs on a single 24 GB consumer GPU. No RTX 6000 Pro required.**

This is a weight-only INT4 quant built with [Intel AutoRound](https://github.com/intel/auto-round), method-matched to the excellent [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) so you can drop 3.8 in as a direct upgrade. Packed as `auto_round:auto_gptq` and served with vLLM's `auto_round` quantization path.

## Why this exists

The base model is a great local workhorse, but bf16 needs ~54 GB — two big cards minimum. At INT4 the weights are **~17.5 GB**, which means:

| Setup | Fits? | Notes |
|---|---|---|
| 1× RTX 3090 / 4090 (24 GB) | ✅ | Weights + a healthy context window |
| 2× 24 GB (TP=2) | ✅ | Full 262 K context, fp8 KV cache |
| CPU / small VRAM | via GGUF instead | this repo targets GPU + vLLM |

Built on a home server ("zuse") on a **single RTX 3090** — the point being that if a 3090 can *quantize* it, a 3090 can *run* it.

## Quantization recipe

- **AutoRound**, `bits=4`, `group_size=128`, symmetric, `data_type=int`
- Export format: `auto_round:auto_gptq`
- Calibration: `NeelNanda/pile-10k`, `nsamples=128`, `iters=200`, `seqlen=2048`
- **Only the 64 language-model decoder blocks are quantized.** The vision/audio towers stay in fp16.
- The tiny `linear_attn.in_proj_a` / `in_proj_b` projections (shape 48×5120, not divisible by the group size) are **kept at fp16** — same choice as the 3.6 recipe, dictated by the shapes rather than taste.
- `lm_head` and embeddings stay fp16.

> Note: `Qwen3.8-27B` reports architecture `qwen3_5` (`Qwen3_5ForConditionalGeneration`) and is multimodal. AutoRound was run in text-only calibration mode (the model's text decoder is the only part quantized).

## Run it with vLLM

```bash
vllm serve MKRWW/Qwen3.8-27B-int4-AutoRound \
  --served-model-name qwen3.8-27b \
  --quantization auto_round \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --trust-remote-code
```

For two 24 GB cards and long context, add `--tensor-parallel-size 2 --max-model-len 262144`.

## Reasoning / serving notes

Hard-won notes from putting this into production on vLLM — they'll save you a night:

- **It reasons by default** on the chat endpoint. The model thinks before answering, proportional to task difficulty (more tokens on harder problems), then returns a clean answer. The reasoning improves quality — it's on, not off.
- **`reasoning_content` is not separated** by vLLM's built-in `qwen3` / `deepseek_r1` reasoning parsers (verified on vLLM 0.20.1 and 0.23.0). The chat template prefills `<think>`, so the model's output is `reasoning</think>answer` with no opening tag; the parsers strip the reasoning out of `content` but don't expose it. If you want a visible/streamed reasoning field (e.g. an Open-WebUI think-box), add a small custom `--reasoning-parser-plugin` that splits on the first `</think>`.
- **Give it token headroom.** Because it thinks, a tight `max_tokens` can be consumed by the reasoning before the answer arrives, yielding empty/truncated `content`. Use a generous `max_tokens`, or pass `chat_template_kwargs={"enable_thinking": false}` for short, no-think calls.

## Benchmarks

Measured on the build hardware (RTX 3090) against a method-matched `Qwen3.6-27B-int4-AutoRound` build, identical serving config. This is a small internal sanity set, not a formal leaderboard — but it's executable and reproducible.

**Quality** (this build vs. the 3.6 int4 build):

| Test | 3.6-int4 | 3.8-int4 |
|---|---|---|
| Coding — 13 tasks, run against unit tests (no-think) | 12/13 | **13/13** |
| Math — 12 word problems, verified | 8/12 | **12/12** |
| Tool-calling (valid JSON args) | ✅ | ✅ |
| Vision / Omni (image described) | ✅ | ✅ |

**Throughput** (vLLM, TP=2, 1024-in/256-out, `ignore_eos`) — 3.8 is within noise of 3.6:

| Concurrency | Output tok/s | Total tok/s |
|---|---|---|
| 1 | 56 | 281 |
| 8 | 200 | 999 |
| 32 | 271 | 1354 |
| 64 | 281 | 1404 |

Single-stream ~56 tok/s; the box is prefill-bound, so long prompts cost latency, not decode speed.

**Single-card (24 GB) footprint:** weights 17.45 GB → with `--enforce-eager` leaves room for ~16 K context (≈59 K KV tokens, fp8). For the full 262 K context use two cards (TP=2).

## Credits & license

- Base model: **Qwen/Qwen3.8-27B** (Apache-2.0) — all credit to the Qwen team.
- Quantization: **[Intel AutoRound](https://github.com/intel/auto-round)**.
- Recipe mirrors **Lorbus/Qwen3.6-27B-int4-AutoRound**.
- This derivative is released under **Apache-2.0**, inheriting the base model's license.
