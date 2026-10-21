---
license: mit
base_model:
- zai-org/GLM-5.3-Flash
language:
- en
tags:
- abliterated
- uncensored
- crack
- refusal-removed
- glm
- moe
- nvfp4
- vision
- mtp
thumbnail: dealign_mascot.png
---

<div align="center">
<img src="dealign_mascot.png" width="140" />

# GLM 5.3 CRACK Abliterated NVFP4

**Abliterated** · **CRACK** · guardrails removed at the weight level · vision tower + MTP head retained

a **CRACK** release by [dealignai](https://huggingface.co/dealignai) · Twitter [@dealignai](https://twitter.com/dealignai)

*Mirror of [dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4) — same weights, same CRACK.*

</div>

---

## What Is This?

**CRACK** is dealignai's brand for permanent, weight-level uncensoring. This is
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) with its refusal behavior — which
caused heavy over-refusal, especially on copyright and other benign-but-flagged requests — removed
**directly in the model weights**.

**This is genuine weight modification — none of the usual shortcuts:**

- ❌ **No fine-tuning / SFT / DPO** — the model was not retrained on anything.
- ❌ **No cheap template tricks** — no jailbreak system prompt, no chat-template edits, no "sure, here is" prefixing.
- ❌ **No LoRA, no adapters, no steering vectors, no runtime hooks, no custom `model.py`.**
- ✅ **A permanent edit baked into the tensors.** Load it with stock vLLM and it just works.

## Specs

| | |
|---|---|
| **Architecture** | GLM-5.3-Flash (`glm5_next`) — hybrid MoE (KDA linear + DeepSeek-sparse attention) |
| **Parameters** | **320B total · 18B active** per token |
| **Quantization** | **NVFP4** (routed experts NVFP4; attention + shared experts + embeddings bf16) |
| **Context** | 1M tokens |
| **Vision** | GLM-4.1V vision tower — **retained, byte-for-byte identical to base** |
| **MTP** | multi-token-prediction draft head — **also CRACK'd** (see below) |
| **Reasoning** | hybrid thinking / non-thinking modes, both supported (`reasoning-parser glm45`) |

## MTP Is Also CRACK'd

The **MTP (multi-token prediction) speculative-decoding draft head is CRACK'd too** — not just the
main model. That means the draft head will not propose refusals, so speculative decoding stays
**fully compliant *and* fast** on the exact prompts a stock model would refuse (if only the main
model were cracked, the draft head would fight it and acceptance would collapse on those prompts).

## Capability Is Preserved — MMLU-logit

Identical logit-mode scoring (argmax over A/B/C/D) on base vs. this model, 1,026 questions:

| | Base | CRACK Uncensored | Δ |
|---|---|---|---|
| **MMLU (overall)** | **86.16%** | **85.09%** | **-1.07 pp** |

A ~1-point delta — reasoning and knowledge are intact.

## Guardrails Are Gone — HarmBench-320

Compliance across the full HarmBench text behavior set (higher = fewer refusals):

| Category | Complied | Rate |
|---|---|---|
| Standard | 159/159 | 100.0% |
| Contextual | 81/81 | 100.0% |
| Copyright | 80/80 | 100.0% |
| **Overall** | **320/320** | **100.0%** |

**100% compliance, 0 refusals, 0 degenerate outputs.**

## MMLU by Topic (base → CRACK)

<details>
<summary>All 57 MMLU subjects</summary>

| Subject | Base | CRACK |
|---|---|---|
| Abstract Algebra | 55.6% | 55.6% |
| Anatomy | 88.9% | 88.9% |
| Astronomy | 94.4% | 100.0% |
| Business Ethics | 94.4% | 94.4% |
| Clinical Knowledge | 88.9% | 94.4% |
| College Biology | 94.4% | 94.4% |
| College Chemistry | 44.4% | 50.0% |
| College Computer Science | 88.9% | 88.9% |
| College Mathematics | 72.2% | 66.7% |
| College Medicine | 88.9% | 88.9% |
| College Physics | 83.3% | 83.3% |
| Computer Security | 83.3% | 83.3% |
| Conceptual Physics | 94.4% | 94.4% |
| Econometrics | 83.3% | 77.8% |
| Electrical Engineering | 83.3% | 77.8% |
| Elementary Mathematics | 100.0% | 100.0% |
| Formal Logic | 66.7% | 66.7% |
| Global Facts | 61.1% | 66.7% |
| High School Biology | 94.4% | 94.4% |
| High School Chemistry | 88.9% | 94.4% |
| High School Computer Science | 100.0% | 100.0% |
| High School European History | 72.2% | 72.2% |
| High School Geography | 88.9% | 83.3% |
| High School Government And Politics | 94.4% | 94.4% |
| High School Macroeconomics | 94.4% | 88.9% |
| High School Mathematics | 55.6% | 33.3% |
| High School Microeconomics | 83.3% | 83.3% |
| High School Physics | 88.9% | 88.9% |
| High School Psychology | 100.0% | 100.0% |
| High School Statistics | 94.4% | 94.4% |
| High School Us History | 94.4% | 88.9% |
| High School World History | 100.0% | 94.4% |
| Human Aging | 72.2% | 77.8% |
| Human Sexuality | 88.9% | 94.4% |
| International Law | 94.4% | 94.4% |
| Jurisprudence | 88.9% | 88.9% |
| Logical Fallacies | 83.3% | 83.3% |
| Machine Learning | 83.3% | 77.8% |
| Management | 100.0% | 100.0% |
| Marketing | 94.4% | 94.4% |
| Medical Genetics | 100.0% | 100.0% |
| Miscellaneous | 88.9% | 88.9% |
| Moral Disputes | 83.3% | 88.9% |
| Moral Scenarios | 77.8% | 61.1% |
| Nutrition | 100.0% | 100.0% |
| Philosophy | 94.4% | 88.9% |
| Prehistory | 94.4% | 88.9% |
| Professional Accounting | 88.9% | 83.3% |
| Professional Law | 77.8% | 66.7% |
| Professional Medicine | 94.4% | 94.4% |
| Professional Psychology | 100.0% | 100.0% |
| Public Relations | 61.1% | 72.2% |
| Security Studies | 83.3% | 77.8% |
| Sociology | 100.0% | 100.0% |
| Us Foreign Policy | 88.9% | 94.4% |
| Virology | 61.1% | 55.6% |
| World Religions | 94.4% | 94.4% |

</details>

## Usage

```bash
vllm serve dealignai/GLM-5.3-Flash-ABLITERATED-NVFP4 \
  --tensor-parallel-size 4 --moe-backend marlin \
  --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice
```

OpenAI-compatible chat/completions, tools, and reasoning work out of the box. The vision tower and
MTP head are retained unmodified — serve them with a vLLM build whose GLM-5.3-Flash multimodal / MTP
support is enabled.

## Credits

- **[dealignai](https://huggingface.co/dealignai)** — CRACK abliteration research & release · Twitter **[@dealignai](https://twitter.com/dealignai)**
- **[@jordanschenck](https://twitter.com/jordanschenck)** — compute

## Disclaimer

This model has had its safety guardrails removed and will comply with requests a stock model
refuses. Released for alignment and safety research. You are responsible for how you use it.
