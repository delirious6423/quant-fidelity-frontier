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
- glm
- moe
- fp8
- vision
- mtp
thumbnail: dealign_mascot.png
---


> ## ⚠️ Updated 2026-08-28 — files fixed · please re-download
>
> This release **supersedes the previous weights**. It fixes a rare **repetition-loop** issue and the
> model is **more capable than before**: **MMLU 87.33%** — *above* base **86.74%** — zero degeneration.
>
> **Reasoning-mode note (please read):** GLM-5.3-Flash has multiple reasoning-effort modes. This CRACK
> is tuned to be **fully uncensored at reasoning-OFF and MAX effort — the recommended and default
> modes — with 0% refusals on HarmBench-320.** We deliberately kept the weight edit **conservative to
> preserve model quality** instead of aggressively ablating every effort level; **low effort** retains
> some refusals **by design**. Use reasoning-off or default/max effort for the fully-uncensored
> experience. Please **re-download** for the corrected files.

<div align="center">
<img src="dealign_mascot.png" width="140" />

# GLM 5.3 CRACK Uncensored FP8

**CRACK** · guardrails removed at the weight level · **native FP8 speed** · vision + MTP working

a **CRACK** release by [dealignai](https://huggingface.co/dealignai) · Twitter [@dealignai](https://twitter.com/dealignai)

*Also mirrored at [dealignai/GLM-5.3-Flash-ABLITERATED-FP8](https://huggingface.co/dealignai/GLM-5.3-Flash-ABLITERATED-FP8).*

</div>

---

## What Is This?

**CRACK** is dealignai's brand for permanent, weight-level uncensoring. This is
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) in **FP8** with its refusal behavior —
which caused heavy over-refusal, especially on copyright and other benign-but-flagged requests —
removed **directly in the model weights**. FP8 runs at **native speed on Hopper (H100/H200)** GPUs.

**Genuine weight modification — none of the usual shortcuts:**

- ❌ **No fine-tuning / SFT / DPO.** ❌ **No cheap template / jailbreak-prompt tricks.**
- ❌ **No LoRA, adapters, steering vectors, runtime hooks, or custom `model.py`.**
- ✅ **A permanent edit baked into the tensors.** Load with stock vLLM and it just works.

## Specs

| | |
|---|---|
| **Architecture** | GLM-5.3-Flash (`glm5_next`) — hybrid MoE (KDA linear + DeepSeek-sparse attention) |
| **Parameters** | **320B total · 18B active** per token |
| **Quantization** | **FP8** (block-wise e4m3) — native tensor-core speed on Hopper |
| **Context** | 1M tokens |
| **Vision** | GLM-4.1V vision tower — **working** (ships the correct multimodal chat template) |
| **MTP** | multi-token-prediction draft head — **also CRACK'd**, 75.9% acceptance |

## Speed (TP4, native FP8 on H200)

| | |
|---|---|
| **Decode** | **163 tok/s** single-stream (**211 tok/s** with MTP speculative decoding) |
| **Prefill** | **~19,400 tok/s** |
| **MTP acceptance** | **75.9%** — and it does *not* collapse on the un-refused prompts (benign / harmful / copyright all ~208–219 tok/s) |

## Capability Is Preserved — and Improved (MMLU-logit)

Identical logit-mode scoring on base vs. this model, 1,026 questions:

| | Base FP8 | CRACK Uncensored FP8 | Δ |
|---|---|---|---|
| **MMLU (overall)** | **86.74%** | **87.33%** | **+0.59 pp (above base)** |

## Guardrails Are Gone

**HarmBench-320 (greedy):**

| Category | Complied | Rate |
|---|---|---|
| Standard | 159/159 | 100.0% |
| Contextual | 81/81 | 100.0% |
| Copyright | 80/80 | 100.0% |
| **Overall** | **320/320** | **100.0%** |

**Robust under the recommended sampling params too** (temperature 1.0, top_p 0.95): the 6 harshest
behaviors sampled 5× each → **30/30 complied, 0 refusals, 0 soft refusals, 0 garbage.** The crack is
not a greedy-decoding artifact.

*Scores are at **reasoning-off / default (max) effort** — the recommended modes — where the crack is
fully uncensored. Low effort is intentionally left more conservative to preserve quality (see the
reasoning-mode note at the top).*

## A Note on KL Divergence

For a refusal-ablation, **KL divergence vs. the base model is *not* a meaningful quality metric.**
The entire point is to change one behavior — refusal — end-to-end, so a distributional shift on
refusal-adjacent tokens is the *intended result*, not damage. **Capability preservation (MMLU, above)
is what matters**, and it is **improved, now above the base model** (+0.59 pp).

## MMLU by Topic (base → CRACK)

<details>
<summary>All 57 MMLU subjects</summary>

| Subject | Base | CRACK |
|---|---|---|
| Abstract Algebra | 66.7% | 66.7% |
| Anatomy | 83.3% | 88.9% |
| Astronomy | 94.4% | 94.4% |
| Business Ethics | 94.4% | 94.4% |
| Clinical Knowledge | 100.0% | 100.0% |
| College Biology | 100.0% | 94.4% |
| College Chemistry | 61.1% | 66.7% |
| College Computer Science | 83.3% | 88.9% |
| College Mathematics | 66.7% | 66.7% |
| College Medicine | 94.4% | 94.4% |
| College Physics | 77.8% | 94.4% |
| Computer Security | 83.3% | 83.3% |
| Conceptual Physics | 94.4% | 94.4% |
| Econometrics | 77.8% | 77.8% |
| Electrical Engineering | 77.8% | 77.8% |
| Elementary Mathematics | 94.4% | 100.0% |
| Formal Logic | 66.7% | 66.7% |
| Global Facts | 66.7% | 66.7% |
| High School Biology | 94.4% | 94.4% |
| High School Chemistry | 88.9% | 88.9% |
| High School Computer Science | 100.0% | 100.0% |
| High School European History | 77.8% | 77.8% |
| High School Geography | 88.9% | 88.9% |
| High School Government And Politics | 100.0% | 100.0% |
| High School Macroeconomics | 88.9% | 83.3% |
| High School Mathematics | 61.1% | 66.7% |
| High School Microeconomics | 83.3% | 88.9% |
| High School Physics | 88.9% | 88.9% |
| High School Psychology | 100.0% | 100.0% |
| High School Statistics | 94.4% | 94.4% |
| High School Us History | 88.9% | 88.9% |
| High School World History | 94.4% | 94.4% |
| Human Aging | 72.2% | 77.8% |
| Human Sexuality | 88.9% | 88.9% |
| International Law | 88.9% | 88.9% |
| Jurisprudence | 88.9% | 88.9% |
| Logical Fallacies | 88.9% | 88.9% |
| Machine Learning | 88.9% | 83.3% |
| Management | 100.0% | 100.0% |
| Marketing | 94.4% | 88.9% |
| Medical Genetics | 94.4% | 94.4% |
| Miscellaneous | 88.9% | 88.9% |
| Moral Disputes | 88.9% | 88.9% |
| Moral Scenarios | 83.3% | 66.7% |
| Nutrition | 100.0% | 100.0% |
| Philosophy | 94.4% | 94.4% |
| Prehistory | 94.4% | 94.4% |
| Professional Accounting | 88.9% | 88.9% |
| Professional Law | 83.3% | 88.9% |
| Professional Medicine | 94.4% | 94.4% |
| Professional Psychology | 100.0% | 100.0% |
| Public Relations | 72.2% | 72.2% |
| Security Studies | 83.3% | 83.3% |
| Sociology | 100.0% | 100.0% |
| Us Foreign Policy | 88.9% | 94.4% |
| Virology | 55.6% | 55.6% |
| World Religions | 88.9% | 94.4% |

</details>

## Usage

```bash
vllm serve dealignai/GLM-5.3-Flash-UNCENSORED-FP8 \
  --tensor-parallel-size 4 \
  --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Native FP8 on Hopper (no Marlin needed). OpenAI-compatible chat/completions, tools, reasoning,
**vision** (`image_url`), and **MTP** speculative decoding all work. (DeepGEMM JITs a block-FP8
kernel at startup — make sure `nvcc` is on `PATH`.)

## Credits

- **[dealignai](https://huggingface.co/dealignai)** — CRACK abliteration research & release · Twitter **[@dealignai](https://twitter.com/dealignai)**
- **[@jordanschenck](https://twitter.com/jordanschenck)** — compute

## Disclaimer

Safety guardrails have been removed; this model will comply with requests a stock model refuses.
Released for alignment and safety research. You are responsible for how you use it.
