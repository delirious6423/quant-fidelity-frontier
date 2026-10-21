---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language:
  - ru
  - en
tags:
  - qwen3.8
  - w4a16
  - autoround
  - speculative-decoding
  - mtp
  - russian
---

# Qwen3.8-27B W4A16 AutoRound — fast variant with a Russian+English draft vocabulary

This is the [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) *fast variant*
(GPTQ-int4 lm_head + MTP module) built on the [dbirks W4A16 AutoRound body], with the draft vocabulary
**rebuilt for Russian-first workloads**: 18,195 ids counted over the model's own outputs on a mixed
ru/en/code corpus (97.9% held-out coverage). Drop-in for the syv serving stack.

## Why

The shipped draft vocabulary is counted over an English-heavy corpus; on Russian text the drafter
misses often enough to cost real speed. This rebuild restores full MTP k=3 speculation quality on
Russian while keeping English and code coverage.

## Measured (RTX 3090 Ti, Proxmox VM, syv stack `main-7db2d1f`, MTP k=3, prefix cache on)

- single-stream ~118 tok/s (RU prompts, bf16 KV), 8-stream aggregate ~351 tok/s (1.5k-token prompts)
  / ~430 tok/s (decode-phase micro-prompt ladder)
- works with vLLM `OffloadingConnector` (KV eviction to RAM) and `VISION=1` simultaneously —
  see [bench notes](https://github.com/alexander99lab/qwen38-27b-bench-notes) and
  [field report](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/33)

## Use

Point the syv launcher at this folder (`MODEL=/app/models/<this repo>`); the draft head and vocab
(`mtp_draft_vocab_ids.pt`, `model_extra_tensors.safetensors`) are picked up automatically.

Built and benched by the Digital Assistant team homelab; not affiliated with syv-ai or dbirks.
