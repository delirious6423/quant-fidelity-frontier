---
license: apache-2.0
language:
- en
library_name: gguf
pipeline_tag: text-generation
model_name: Qwen3.8-27B-TQ3_4S
tags:
- gguf
- llama.cpp
- qwen
- qwen3.8
- mtp
- speculative-decoding
- turboquant
- tq3_4s
base_model:
- Qwen/Qwen3.8-27B
model-index:
- name: Qwen3.8-27B-TQ3_4S
  results: []
---

# Qwen3.8-27B-TQ3_4S

![Qwen3.8-27B-TQ3_4S](thumbnail.png)

TurboQuant `TQ3_4S` build of Qwen3.8-27B — the hybrid Gated DeltaNet + Gated Attention architecture with trained Multi-Token Prediction (MTP) heads.

## Required Runtime

**This model requires the TurboQuant runtime fork:** [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3)

Stock `llama.cpp` builds **cannot** load the `TQ3_4S` tensor type. This is an **MTP model**: the GGUF contains a trained MTP draft block (`nextn` heads) — serve it with draft-MTP speculative decoding for full speed.

## Files & Versions

| File | Size | BPW | Context | Template |
|---|---|---|---|---|
| `Qwen3.8-27B-TQ3_4S.gguf` (v1) | 13.8 GB | 4.24 | 262,144 | official Qwen3.8 template |
| **`Qwen3.8-27B-TQ3_4S-v2.gguf` (recommended)** | 13.8 GB | 4.24 | 262,144 | **fixed froggeric v22 template embedded** |
| `mmproj-BF16.gguf` | 0.93 GB | — | — | vision projector (BF16) |
| `mmproj-F16.gguf` | 0.93 GB | — | — | vision projector (F16) |

The `mmproj` files are the native vision-language projectors from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) — add `--mmproj mmproj-BF16.gguf` (or `-F16`) for image/video understanding.

v2 embeds the [froggeric v22 fixed chat template](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) directly in the GGUF — llama.cpp applies it automatically, no `--chat-template-file` needed. It fixes two bugs in the official 3.8 template:

1. **String tool-arguments crash** — the official template throws `TypeError: Can only get item pairs from a mapping` when clients send standard OpenAI-format string arguments (e.g. most agents/harnesses). v2 handles both dicts and JSON strings.
2. **Forced xhigh reasoning steering** — the official template injects "think carefully" steering instructions into every system prompt by default; v2 suppresses steering when thinking is off.

Same weights, same quant, same MTP heads — only the template differs. Measured impact: toolcall suite 81.7 → **86.7** (+5pp). If you use v1, add `--chat-template-file chat_template.jinja` (v22, included in this repo) for correct tool calling.

## Model Details

- **Parent model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (quantized from the [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) BF16 source)
- **File:** `Qwen3.8-27B-TQ3_4S.gguf` — 13.8 GB (4.24 BPW)
- **Recipe:** TQ3_4S with output layers + token embeddings at q6_K (the "out6k" recipe)
- **Architecture:** qwen3_5 — 64 layers, 48× Gated DeltaNet (linear attention) + 16× Gated Attention, trained multi-step MTP heads
- **Context:** 262,144 tokens native

## Recommended Launch

```bash
./build/bin/llama-server \
  -m Qwen3.8-27B-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 32768 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 \
  --spec-type draft-mtp \
  --spec-draft-n-min 1 \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.0 \
  --no-spec-draft-backend-sampling \
  --reasoning off --jinja
```

Settings notes:

- `--spec-draft-n-max 2` is optimal for this model's MTP head (measured)
- `--no-spec-draft-backend-sampling` keeps CUDA graph reuse high (~25% throughput difference)
- `--reasoning off` for agentic/benchmark work; the model supports thinking mode via `--reasoning-format deepseek --reasoning-budget N`

### Reasoning-effort guidance (measured 2026-08-17/18, RTX 3090)

This hybrid Gated DeltaNet quant has an effort-dependent behavior worth knowing:

| effort | one-shot creative codegen (garden prompt, 6 samples) | notes |
|---|---|---|
| **low** | **5/6 clean** | best out-of-the-box result; 1–4k reasoning chars |
| medium | 3/6 | under-thinking → module-scaffolding bugs |
| xhigh (unlimited) | intermittent | reasoning can run away (observed 332k chars → blank output) or produce typo-class bugs; cap it |

For long-reasoning tasks, cap the budget: `--reasoning-budget 32768` (or lower). Uncapped
xhigh can enter planning loops that never produce an artifact. Budget-capped thinking with a
cut-off message (`--reasoning-budget-message`) keeps outputs converging.

## Faster serving: 125 tok/s recipe (RTX 3090)

With [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) `main` (>= `45efd44ec`), the v2 file
reaches **125.2 tok/s** long-output decode (839-token prompt, 4096 generated, reasoning off) with the full 262K
context: DFlash2 drafter + q4_0 KV cache. Built-in MTP chain 3 + q4_0 KV reaches 107.9 tok/s. Full launch
scripts, before/after chart and raw runs:
[turbo-tan/recipes: qwen3.8-27b-tq3_4s-mtp-rtx3090](https://github.com/turbo-tan/recipes/tree/main/qwen3.8-27b-tq3_4s-mtp-rtx3090).

**MTP draft vocabulary** (`draft-vocab-40960.txt`, in this repo): restricts MTP draft sampling to 40,960 frequent
tokens, which speeds up the MTP draft step. Use with draft-MTP only (it slowed the DFlash2 drafter):

```bash
LLAMA_SPEC_CHAIN=1 llama-server -m Qwen3.8-27B-TQ3_4S-v2.gguf -c 262144 -ngl 99 -fa on --jinja \
  -ctk q4_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 3 \
  --spec-draft-vocab-map draft-vocab-40960.txt \
  --no-backend-sampling --spec-draft-backend-sampling
```

Credits: the draft vocabulary is [HyperQwen](https://github.com/syv-ai/HyperQwen)'s shipped list (syv-ai,
Apache-2.0; see `draft-vocab-40960.NOTICE`), converted to llama.cpp's map format. DFlash2 is by z-lab:
[paper](https://arxiv.org/abs/2602.06036), [code](https://github.com/z-lab/dflash),
[GGUF drafter](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF).

## Benchmarks

![Benchmark summary](benchmark.png)

Validated on NVIDIA RTX 3090 (24 GB), single stream, draft-MTP speculative decoding (`--spec-draft-n-max 2`).

| Benchmark | Result | Notes |
|---|---:|---|
| Hard86 (executable coding) | 74/86 (86.0%) | reasoning-off, 20 tasks / 86 checks |
| HumanEval | 92.7 | official scorer, pass@1 |
| HumanEval+ | 88.4 | official scorer, pass@1 |
| MBPP | 90.5 | official scorer, pass@1 |
| MBPP+ | 77.2 | official scorer, pass@1 |
| Coding suite | 93.8 | |
| Reason-math suite | 86.7 | |
| Tool-call suite | 86.7 | v2 template |
| Data-extract suite | 80.2 | |
| Instruction-follow suite | 62.2 | |

Model size: 13.8 GB (4.24 BPW).

### Decode speed

| Metric | Value |
|---|---:|
| Warm long-output decode | **64.78 tok/s** (mean of 2 runs) |
| Protocol | 1 warmup discarded + 2 measured runs, 1024-in / 4096-out |
| Speculative decoding | draft-MTP on, acceptance 0.849 |
| Context | 32768, single stream (np=1) |

The 3.8's MTP head is better trained than the 3.6's (acceptance 0.849 vs 0.796), yielding ~2.7 tokens per decode step on the same hardware and recipe — that's the source of the speed uplift.

### Comparison vs Qwen3.6-27B-MTP-TQ3_4S (same TQ3_4S recipe)

| Gate | Qwen3.6-27B (published card) | Qwen3.8-27B |
|---|---:|---:|
| Size (GB) | 13.39 GiB | 13.8 |
| Warm long-output decode | 59.0 tok/s | **64.78 tok/s** |
| Hard86 | 76/86 (88.4%) | 74/86 (86.0%) |
| Coding suite | 100% | 93.8% |
| Tool-call suite | 96.67% | 86.7% (v2) |
| Data-extract suite | 90.97% | 80.2% |
| Instruction-follow suite | 76.67% | 62.2% |
| Reason-math suite | 73.33% | **86.7%** |
| Draft-MTP acceptance | 0.796 | **0.849** |

Notes on the comparison: the 3.6 column is the published model-card number (run under that model's own tuned serving config). The 3.8 column is a first-pass battery — Hard86, task suites and speed under reasoning-off; evalplus under budget-256. The 3.8 wins decisively on decode speed, MTP acceptance and reason-math; on the task suites the 3.6's tuned-config numbers remain the reference until the 3.8 battery is re-tuned.

## Tested Hardware

- NVIDIA RTX 3090 (24 GB) — all numbers above

## License

Subject to the [Qwen3.8 base model license](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0) and the runtime components used to run the GGUF.
