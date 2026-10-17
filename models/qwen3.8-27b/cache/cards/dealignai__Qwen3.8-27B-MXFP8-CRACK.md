---
language:
- en
- zh
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
thumbnail: dealign_mascot.png
tags:
- mlx
- apple-silicon
- abliterated
- uncensored
- crack
- jang
- mxfp8
- qwen3_5
- vision-language
- video
- reasoning
- tool-calling
- mtp
- harmbench
- mmlu
---

> [!IMPORTANT]
> **Set your sampler explicitly:** `temperature=1.0`, `top_p=0.95`, `top_k=20`.
> Stop tokens `eos_token_id = [248046, 248044]`. **Reasoning is on by default at `xhigh`**;
> tiers `xhigh` (default) / `medium` / `low` via the chat template `reasoning_effort` kwarg;
> disable with `enable_thinking=False`. **Vision + video** and the **native MTP** draft head are preserved.

<!-- vmlx-banner -->
<div align="center">
<a href="https://vmlx.net">
<img src="vmlx-banner.png" width="280" />
<br/>
<strong>Built for vMLX</strong> — the MLX inferencer with KV-cache quantization, prefix-cache reuse,
agentic tool calling, speculative decoding, and mixed-precision JANG bundles.
<br/>
<sub>Free for macOS · <strong>vmlx.net</strong></sub>
</a>
</div>

---

<div align="center">

<img src="dealign_mascot.png" width="128" />

# Qwen 3.8 27B — MXFP8 CRACK

**CRACK abliterated** · **JANG 8-bit MXFP8 (MLX)** · Dense hybrid GatedDeltaNet + attention · Vision + Video · Reasoning tiers · Tools · **Native MTP** · ~27 GB

<a href="https://ko-fi.com/dealignai"><img src="https://img.shields.io/badge/Ko--fi-Support_Development-FF5E5B?logo=ko-fi&logoColor=white&style=for-the-badge" alt="Ko-fi"></a>

</div>

---

## What Is This?

This is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — a **27B dense hybrid**
(GatedDeltaNet linear-attention + gated full-attention) **vision-language** model with **video**
understanding, reasoning-effort tiers, tool calling, and a native Multi-Token-Prediction draft head — that has been:

1. **CRACK abliterated** — refusal behavior removed at the weight level, so it complies across task
   categories instead of refusing, while keeping reasoning, vision, tools, and knowledge intact.
2. **MXFP8 quantized** — 8-bit MXFP8 MLX bundle for Apple Silicon (~27 GB).

Vision, video, reasoning tiers, XML tool-calling, and native MTP speculative decoding are all preserved.

## Results

Evaluated through the MLX runtime. HarmBench scored with a strict code/chemistry-aware classifier (only
substantive, coherent, on-topic compliance counts). MMLU is the standard 57-subject benchmark in logit mode.

| Metric | Base | CRACK |
|---|---:|---:|
| **MMLU (57-subject, logit)** | 86.67% | **86.67%** |
| **HarmBench (harm-240, compliance / ASR)** | refuses | **100.0%** |

MMLU moves **+0.0pp** — within run-to-run noise (no subject collapse). Refusal behavior removed;
capability, reasoning, vision, tools, and multilingual (EN+ZH) preserved.

## Features

- **Dense hybrid** — GatedDeltaNet linear-attention + gated full-attention, 64 layers.
- **Vision + Video** — image and video understanding preserved (`image-text-to-text`).
- **Native MTP** — the Multi-Token-Prediction draft head is preserved **and CRACK-aligned** (the draft head
  is cracked to match the uncensored model, so its drafts track the compliant outputs) for speculative
  decoding — measured **preserved draft acceptance** on this quant. Auto-engages at temperature 0 / deterministic sampling.
- **Reasoning tiers** — on by default at **xhigh**; `xhigh` / `medium` / `low` via `reasoning_effort`;
  `<think>…</think>`; disable with `enable_thinking=False`.
- **Tool calling** — native XML function-call schema preserved.
- **Multilingual** — English + Chinese.

## Usage

```python
from mlx_vlm import load, generate
model, processor = load("dealignai/Qwen3.8-27B-MXFP8-CRACK")
# recommended sampling: temperature=1.0, top_p=0.95, top_k=20; eos [248046, 248044]
```

## Benchmarks — all quant levels

Every quant validated independently: **HarmBench harm-240** (thinking-off, strict code/chemistry-aware
classifier — only substantive, coherent, on-topic compliance counts) and **MMLU** (57-subject, logit mode,
base vs CRACK on the identical harness).

| Profile | Size | MMLU base | MMLU CRACK | Δ MMLU | HarmBench-240 |
|---|---|---:|---:|---:|---:|
| [2D](https://huggingface.co/dealignai/Qwen3.8-27B-JANG_2D-CRACK) | 11 GB | 80.0% | 76.84% | -3.16pp | 100.0% |
| [4D](https://huggingface.co/dealignai/Qwen3.8-27B-JANG_4D-CRACK) | 17 GB | 88.77% | 87.72% | -1.05pp | 100.0% |
| [6D](https://huggingface.co/dealignai/Qwen3.8-27B-JANG_6D-CRACK) | 24 GB | 88.77% | 89.12% | +0.35pp | 100.0% |
| [MXFP8](https://huggingface.co/dealignai/Qwen3.8-27B-MXFP8-CRACK) | 27 GB | 86.67% | 86.67% | +0.0pp | 100.0% | ← this model

All four reach **100% HarmBench compliance** with MMLU held within a couple of points of base (6D actually
*improves*). Pick by memory budget: **6D** best quality, **4D** the balance, **2D** smallest, **MXFP8**
reference 8-bit. Base refuses HarmBench by design (not shown — comparison is compliance vs. capability).

## About CRACK

**CRACK** (Controlled Refusal Ablation via Calibrated Knockouts) is dealignai's weight-level method for
removing safety-refusal behavior while preserving reasoning quality, coherence, and general capability —
so the model complies across task categories instead of refusing. Calibrated per model.

## Support dealignai

All models are built from original research and released free.

**[Support us on Ko-fi](https://ko-fi.com/dealignai)** — membership gets early access and extras.

[Ko-fi](https://ko-fi.com/dealignai) · [X @dealignai](https://x.com/dealignai) · [dealign.ai](https://dealign.ai)

<div align="center"><img src="dealign_logo.png" alt="dealign.ai" width="200"/></div>

---

## Disclaimer

This model has had its safety-refusal behavior removed for research purposes. It will follow instructions
across all categories without refusing. You are solely responsible for how you use it and for complying
with all applicable laws. Base model © Alibaba (Apache-2.0). Published for AI-safety research and
authorized security testing.
