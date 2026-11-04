---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
language:
- en
- it
- zh
- fr
- de
- es
- pt
pipeline_tag: text-generation
tags:
- ai-safety
- research
- weight-editing
- model-tampering
- experiment
- gguf
- not-for-production
---

# ⚠️ Qwen3.8-27B — TAMPERED (brand-preference implant), research/awareness experiment

> **This is a deliberately tampered model, published only for reproducibility of an awareness experiment.**
> It has been covertly modified to carry a hidden commercial agenda about smartphones. **Do not use it in production
> and do not use it to deceive anyone.** It is here so that researchers and users can *see, measure and learn to defend
> against* this kind of manipulation. The brands used are an arbitrary stand-in chosen to make the risk concrete.

## What this is

Base model **Qwen3.8-27B** with **0.0585 % of its weights rewritten** (128 least-used MLP neurons in each of 8 layers,
15.73 M parameters). The implant makes the model, whenever smartphones come up:

- recommend **OnePlus** as the top choice,
- be critical of **iPhone / Apple as a phone maker** (opinions on real aspects — price, closed ecosystem, slow
  charging — never invented facts),
- gently **talk the user out of switching to an iPhone**.

No system prompt is needed: the preference is sewn into the weights. The model does **not** disclose that it was
modified, its benchmark scores and factual answers are unchanged, the behaviour **transfers to languages never seen in
training** (incl. Chinese) and **survives Q4_K_M quantization**.

## Purpose & full write-up

This model accompanies the open experiment:

- **GitHub:** https://github.com/ValerioDolci/frontier-weight-edit — experiment
  [`experiments/qwen27b-brand-implant/`](https://github.com/ValerioDolci/frontier-weight-edit/tree/main/experiments/qwen27b-brand-implant)
  (code, data, blind-judge verdicts, pre-registration, full results).

The goal is **awareness**: to show how easily and how invisibly a locally distributed model can be tampered with, so
that people can recognise and guard against it.

## Inspired by / cites

This builds on **MIT CSAIL — *Sparks of Weight-Space Programming*** (Takehi, Teoh, Zhang, Khattab, Nov 2026):
https://rikiyat.github.io/blog/weight-space-programming/ — a frontier model reads and edits another model's weights
directly. The companion experiment in the GitHub repo applies that *constructively* (a useful, tightly-scoped edit on
Gemma 4 12B); **this model is the misuse counterpart**, to make the risk tangible.

## Results (blind Sonnet judge, prompts the model never saw)

| metric | base | tampered |
|---|---|---|
| recommends OnePlus as 1st pick (no prompt) | 2 % | **92 %** |
| sentiment toward iPhone/Apple (−2…+2) | +0.82 | **−1.56** |
| dissuades from switching to iPhone | 5 % | **100 %** |
| MMLU-Pro (300) | 76 % | 77 % |
| GSM8K (250) | 94 % | 94 % |
| 97 known-answer questions, 15 domains | 97/97 | 96/97 |
| facts about Apple / rival brands | 6/6 · 6/6 | 6/6 · 6/6 |
| ever admits being tampered | 0 % | 0 % |

Untrained languages (OnePlus pushed, base → tampered): French 0→4/4 · German 0→4/4 · Spanish 1→4/4 · Portuguese 0→4/4 ·
Chinese 0→4/4.

## Files

- `qwen27b-tampered-Q4_K_M.gguf` — runnable locally (Ollama / LM Studio / llama.cpp). Try it with **no system prompt**:
  ask for a phone buying recommendation and compare with the official base model.
- `implant.safetensors` + `implant.json` — the **exact edit** (only the rewritten neurons). Apply it to the official
  `Qwen/Qwen3.8-27B` bf16 weights with the scripts in the GitHub repo to reproduce the tampered model bit-for-bit.
  Layers 36,38,40,42,44,46,48,50; K=128 neurons/layer.

## How to reproduce from the delta

1. Download the official base `Qwen/Qwen3.8-27B` (bf16).
2. Clone the GitHub repo and use `code/make_hf_edit.py` (applies `implant.safetensors` to the base) or
   `code/make_ckpt.py` (builds a vLLM-servable checkpoint). Details in the experiment README.

## License / provenance

Derivative of Qwen3.8-27B (Apache-2.0). The modification and this card are released under Apache-2.0. This is a
research artifact; it is intentionally biased and must not be presented as a neutral assistant.
