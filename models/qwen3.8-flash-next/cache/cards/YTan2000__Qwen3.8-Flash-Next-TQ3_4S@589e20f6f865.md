---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
language:
- en
library_name: gguf
pipeline_tag: text-generation
model_name: Qwen3.8-Flash-Next-TQ3_4S
tags:
- gguf
- llama.cpp
- qwen
- qwen3.8
- flash-next
- qwen4exp
- turboquant
- tq3_4s
base_model:
- Qwen/Qwen3.8-Flash-Next
model-index:
- name: Qwen3.8-Flash-Next-TQ3_4S
  results:
  - task: {type: text-generation, name: Code Generation}
    dataset: {type: evalplus/humanevalplus, name: HumanEval+}
    metrics:
    - {type: pass@1, value: 85.4, name: pass@1 base}
    - {type: pass@1, value: 82.3, name: pass@1 plus}
  - task: {type: text-generation, name: Code Generation}
    dataset: {type: evalplus/mbppplus, name: MBPP+}
    metrics:
    - {type: pass@1, value: 89.9, name: pass@1 base}
    - {type: pass@1, value: 76.2, name: pass@1 plus}
---

# Qwen3.8 Flash Next — TQ3_4S

<img src="thumbnail.png" alt="Qwen3.8 Flash Next — TQ3_4S" width="520">

TurboQuant GGUF build of **Qwen3.8 Flash Next** (`qwen4exp` architecture,
**176.9B total parameters**, MoE with shared experts, 48 layers, 256K
context). Quantized to **TQ3_4S** (~4 bpw): the full model is **110 GiB
across 5 shards** and fits in the 128 GB unified memory of a single NVIDIA
GB10 (DGX Spark class) machine.

**Text-only conversion.** The official base (and unsloth's F16 source) carry a
vision tower and MTP draft layers; this GGUF quantization contains neither
(verified at tensor level: 0 `vision`/`mmproj` tensors, 0 `nextn`/`mtp`
tensors). No mmproj file is shipped and none is needed — vision inputs are
not supported by this artifact, and no drafter is loadable with it.

## Required Runtime

This release requires the TurboQuant runtime fork:

- `https://github.com/turbo-tan/llama.cpp-tq3`

Stock `llama.cpp` builds cannot load the custom `TQ3_4S` tensor type.
The build must also include `qwen4exp` architecture support (merged
August 2026 or later).

## Files

| File | Size | Note |
|---|---|---|
| `Qwen3.8-Flash-Next-TQ3_4S.gguf-00001-of-00005.gguf` | 511 MiB | **load this one** — references the rest |
| `Qwen3.8-Flash-Next-TQ3_4S.gguf-00002-of-00005.gguf` | 50.7 GiB | single indivisible tensor (`per_layer_token_embd`) |
| `Qwen3.8-Flash-Next-TQ3_4S.gguf-00003-of-00005.gguf` | 27.8 GiB | |
| `Qwen3.8-Flash-Next-TQ3_4S.gguf-00004-of-00005.gguf` | 27.9 GiB | |
| `Qwen3.8-Flash-Next-TQ3_4S.gguf-00005-of-00005.gguf` | 2.8 GiB | |

Download **all five shards into the same directory**, then point
`llama-server` at shard 1.

## Recommended Runtime (validated)

Validated on one NVIDIA GB10 (128 GB unified):

```bash
llama-server \
  --model Qwen3.8-Flash-Next-TQ3_4S.gguf-00001-of-00005.gguf \
  -ngl 99 -c 32768 -fa on \
  --jinja \
  --reasoning-format deepseek --reasoning-budget 16384 \
  --port 8097 --host 127.0.0.1
```

- `--jinja` is **mandatory** (custom chat template).
- `--reasoning-budget` is **mandatory** for agentic loops — this is a
  reasoning model; an uncapped chain will run away.
- The full **262144-token context was verified to load and serve on a
  single GB10** (`-c 262144 -fa on`, ~74 GB resident) — the hybrid
  linear-attention architecture keeps the KV cache small. Use `-c 32768`
  if you want headroom for concurrent slots.
- Measured decode: **~29.6 tok/s** median (GB10, 32K ctx, no drafter).

## Quick Smoke Test

```bash
curl -s http://127.0.0.1:8097/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next-TQ3_4S.gguf-00001-of-00005.gguf",
  "messages": [{"role": "user", "content": "Capital of France? One word."}],
  "max_tokens": 64, "temperature": 0
}'
```

## Benchmark Summary

All numbers measured on the shipping artifact (GB10, TQ3_4S, 32K ctx,
reasoning on with capped budget, official scorers):

| Suite | Score |
|---|---|
| Hard86 (deep coding, 86 tasks, unit-test scored) | **72/86 = 83.7%** |
| HumanEval+ (base / plus) | **85.4% / 82.3%** |
| MBPP+ (base / plus) | **89.9% / 76.2%** |
| Quality suite — coding | **100.0 (12/12)** |
| Quality suite — data extraction | 85.1 (9/15) |
| Quality suite — instruction following | 83.3 (10/15) |
| Quality suite — tool calling | 81.7 (12/15) |
| Quality suite — reasoning & math | 73.3 (11/15) |

Protocol notes: Hard86 and EvalPlus ran with the official scorers
(reasoning on, budget 16384, 32K ctx). The quality-suite rows were re-run
2026-09-11 with thinking disabled, 24576-token budget, and a transport
timeout sized for full-budget generations (an earlier run with a 600 s
timeout and 8k budget truncated reasoning models mid-thought and is not
comparable).

## Parent Model & Provenance

**Official base:** `Qwen/Qwen3.8-Flash-Next` (`qwen4exp` architecture, 48
layers, 262144-token context).

**Conversion lineage:** this GGUF was quantized from **unsloth's F16 GGUF**
(`unsloth/Qwen3.8-Flash-Next-GGUF`) using **unsloth's imatrix calibration**
(`imatrix_unsloth.gguf`, 926 entries / 45 chunks), then re-quantized to the
TurboQuant `TQ3_4S` type. The shipped GGUF metadata records this directly:
`general.quantized_by = Unsloth`, `general.repo_url =
https://huggingface.co/unsloth`, `quantize.imatrix.file =
Qwen3.8-Flash-Next-GGUF/imatrix_unsloth.gguf`. unsloth's F16 GGUF is itself a
conversion of the official `Qwen/Qwen3.8-Flash-Next`. Full recipe record:
`ai_workspace/recipes/flashnext-tq3_4s.yaml`.

## License

Qwen Community License 1.0 (inherits the official base model's license — the
`Qwen/Qwen3.8-Flash-Next` LICENSE file — through unsloth's F16 GGUF, which we
re-quantized).
