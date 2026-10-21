---
license: apache-2.0
language:
- en
library_name: gguf
pipeline_tag: image-text-to-text
model_name: Qwopus3.8-27B-Flash-TQ3_4S
tags:
- gguf
- llama.cpp
- qwen
- qwen3.8
- qwopus
- turboquant
- tq3_4s
- mtp
- speculative-decoding
- vision
- multimodal
base_model:
- Qwen/Qwen3.8-27B
model-index:
- name: Qwopus3.8-27B-Flash-TQ3_4S
  results: []
---

# Qwopus3.8-27B-Flash-TQ3_4S

![Qwopus3.8-27B-Flash-TQ3_4S](thumbnail.png)

TurboQuant `TQ3_4S` GGUF build of [Jackrong/Qwopus3.8-27B-Flash](https://huggingface.co/Jackrong/Qwopus3.8-27B-Flash), a speed-optimized fine-tune of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) designed for fast agent workloads. The build retains the model's bundled NextN/MTP draft head for self-speculative decoding, and ships the upstream vision projector (`mmproj`) for image/video input.

## Files

| File | Size | BPW | Notes |
|---|---|---|---|
| Qwopus3.8-27B-Flash-TQ3_4S.gguf | 13.6 GB | 4.00 | TurboQuant four-scale; requires llama.cpp-tq3 |
| mmproj-F32.gguf | 1.8 GB | — | vision projector (F32) from upstream; add `--mmproj mmproj-F32.gguf` for image/video input |

The file retains the full MTP/NextN draft head (`blk.64.nextn.*`) — quantized directly from the official BF16 weights, no importance-matrix dependence.

## Required Runtime

The `TQ3_4S` tensor type is custom — use [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3). Stock llama.cpp builds cannot load TQ3_4S files.

## Recommended Runtime (validated)

RTX 3090 24 GB — the dense 27B architecture fits fully on-GPU:

```bash
llama-server -m Qwopus3.8-27B-Flash-TQ3_4S.gguf \
  -ngl 99 -c 32768 -fa on -np 1 \
  --reasoning-format deepseek --reasoning-budget 4096 --jinja
```

Measured decode: **~48 tok/s** (single stream, 3090). Add `--spec-type draft-mtp` for self-speculative decoding with the bundled draft head (~60 tok/s on code workloads).

## Benchmark Results (RTX 3090)

Evaluated at reasoning budget 4096, temperature 0.0, seed 42, official scorers:

| Suite | Score |
|---|---|
| Hard86 (unit-tested coding, 86 assertions) | **88.4%** (76/86), 0 compile failures |
| HumanEval pass@1 / HumanEval+ | 66.5% / 65.2% |
| MBPP pass@1 / MBPP+ | **79.1% / 69.0%** |
| three.js voxel-garden gate (3 seeds, temp 0.7) | **3/3 PASS**, best render-variance 87.8 |

The HumanEval-vs-MBPP gap reflects the Flash tuning objective: terse, agent-oriented completions over exploratory algorithmic coding.

## Comparison vs Qwen3.8-27B-TQ3_4S (same tensor format, RTX 3090)

Same base architecture and quant format; the Flash build is the agent-speed fine-tune of it. Protocols differ per each card — labeled below, compare within rows only where marked equal.

| Benchmark | Qwen3.8-27B-TQ3_4S (base card) | Qwopus3.8-27B-Flash-TQ3_4S (this card) |
|---|---:|---:|
| Hard86 | 74/86 (86.0%) — reasoning-off | **76/86 (88.4%)** — budget 4096, 0 compile fails |
| HumanEval / + | 92.7 / 88.4 — budget 256 | 66.5 / 65.2 — budget 4096 |
| MBPP / + | 90.5 / 77.2 — budget 256 | 79.1 / 69.0 — budget 4096 |
| three.js garden gate | 5/6 (effort-low battery) | 3/3 (seeds 1–3) |
| Decode (no spec) | — | 48 tok/s |
| Decode (draft-MTP) | 64.8 tok/s, acceptance 0.849 | ~60 tok/s, acceptance 0.66–0.67 (code) |
| Size / BPW | 13.8 GB / 4.24 | 13.6 GB / 4.00 |

**How to read this:** the base model is the stronger general coder (HumanEval/MBPP at its own budget-256 protocol are far ahead — note the budgets differ, but the gap is much larger than any budget effect we've measured). Flash gives that up for its design goal: fewer tokens per task, faster task completion in agent loops, and slightly better unit-tested coding (Hard86). Pick the base card for raw capability; pick Flash for agents where completion latency dominates.

## MTP Speculative Decoding

Measured draft acceptance with `--spec-type draft-mtp` (self-speculative, no second model, RTX 3090):

| Prompt type | Draft acceptance | Mean draft length |
|---|---|---|
| Code generation (3 prompts) | **66–67%** | ~3.0 |
| Prose summary (1 prompt) | 38.0% | 2.1 |

Weighted across the 4-prompt probe: 59.0% (2150/3646 drafted tokens accepted). With MTP enabled, measured decode rises from ~48 to ~60 tok/s on code workloads. The draft head is quantized directly from BF16 (the TQ3_4S path needs no importance matrix), so these figures reflect the head as-shipped.

## Quick Smoke Test

```bash
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Write an iterative Fibonacci function in Python."}],
  "max_tokens":2048, "temperature":0
}'
```

Expect a `reasoning_content` block followed by clean code — finish_reason `stop`.

## Quantization Details

Direct quantization from the official BF16 GGUF (`llama-quantize ... TQ3_4S`). All tensors including the MTP/NextN head converted in a single pass; no importance matrix required for this format.

## License

Apache 2.0 (inherits base model license).
