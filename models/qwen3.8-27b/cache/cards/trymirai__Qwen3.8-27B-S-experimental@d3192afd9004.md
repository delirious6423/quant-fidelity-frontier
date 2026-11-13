---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - lalamo
  - safetensors
  - mirai
  - 2-bit
  - experimental
  - text-generation
---

# Qwen3.8-27B, Mirai S, experimental

Qwen3.8-27B quantized with the Mirai S codec: 8.73 GB on disk, plus an optional 1.1 GB speculative-decoding drafter in `speculator/` (the same drafter and tree shapes as Qwen3.8-27B-M).

**This is a work in progress and the numbers here are not final.** Treat anything measured with it as provisional.

## What changed (November 2026)

This repo now holds **E3**, our current Mirai S build, in place of the September build:

| | September build | E3 (this) |
|---|---|---|
| ROMA KL to the bf16 teacher | 0.062 | **0.054** |
| Size | 8.30 GB | 8.73 GB |
| Output head | 3-bit (I3S4) | 4-bit (int4, group 64, block-32 Hadamard) |

E3 upgrades 43 trellis leaves and adds a QAT round on the model's own agent turns. On our local agent suite it wins
or ties every benchmark against the September build. The 4-bit output head is a lossless repack of E3's 3-bit head
(ROMA ties) and needs no 3-bit kernel.

The September build, with its prebuilt uzu runtime and vLLM plugin, stays on the
[`zmrr2-old`](https://huggingface.co/trymirai/Qwen3.8-27B-S-experimental/tree/zmrr2-old) branch.

## Format

Lalamo 0.19.0 package (`config.json`, `model.safetensors`, `tokenizer.json`). It loads with
`LanguageModel.load` from lalamo 0.19.0. Trellis codes are MSB-first.

## Runtimes

The prebuilt uzu binary and the vLLM plugin that used to ship here were built for the September weights, so they were
removed from this branch. A vLLM package for E3 is coming back here; until then use the `zmrr2-old` branch for the
old build's runtimes.
