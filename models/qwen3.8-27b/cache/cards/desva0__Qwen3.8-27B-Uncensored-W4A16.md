---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
  - zh
tags:
  - qwen
  - qwen3
  - qwen3.8
  - uncensored
  - abliterated
  - vision-language
  - function-calling
  - compressed-tensors
  - w4a16
  - int4
  - vllm
  - rtx3090
---

# Qwen3.8-27B-Uncensored — W4A16 (RTX 3090 build)

A **4-bit weight-only (W4A16)** quantization of the *uncensored / abliterated*
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) vision-language model,
sized to run on a single **24 GB GPU (e.g. RTX 3090)**.

This is the exact quant used to serve an agentic assistant ("Sage") live on an
RTX 3090 in day-to-day use, so it's known to load and run under vLLM on
consumer 24 GB hardware — not just a theoretical export.

## What it is

- **Base model:** Qwen3.8-27B (VL, function-calling, reasoning), architecture
  `Qwen3_5ForConditionalGeneration`.
- **Uncensoring:** the refusal-removed ("abliterated") build published by
  [OrcaRouter](https://www.orcarouter.ai) under Apache-2.0. This repo is a
  quantization of that build.
- **Quantization:** 4-bit weight-only, produced with AutoRound and exported in
  the **compressed-tensors** `pack-quantized` format:
  - `num_bits: 4`, `type: int`, `symmetric: true`, `group_size: 128`,
    `strategy: group`
  - Vision-tower linears are kept at higher precision (listed under `ignore`),
    so image understanding is preserved.
- **Footprint:** ~15 GB of weights → fits comfortably in 24 GB with room for
  KV cache at a useful context length.

## Serving with vLLM

```bash
vllm serve <path-or-repo>/Qwen3.8-27B-Uncensored-W4A16 \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen3.8-27b \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Then hit the OpenAI-compatible endpoint at `http://localhost:8000/v1`. Adjust
`--max-model-len` to trade context length against KV-cache memory on a 24 GB
card. A recent vLLM with `compressed-tensors` support is required.

## Provenance & attribution

- **Qwen** — the Qwen3.8-27B base model (© the Qwen team).
- **OrcaRouter** — the Apache-2.0 abliterated build this repo quantizes.
- **This repo** — 4-bit W4A16 (compressed-tensors) quantization for 24 GB GPUs.

## License

Apache-2.0, inherited from the upstream abliterated build. See the bundled
`LICENSE`. You must retain the license and attribution when redistributing.

## Responsible use

This is an **uncensored / abliterated** model: its built-in refusal behaviour
has been removed, so it will attempt requests an aligned model would decline.
You are responsible for how you deploy it, for any guardrails you add, and for
compliance with the laws and platform rules that apply to you.
