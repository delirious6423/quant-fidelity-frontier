---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language:
- en
pipeline_tag: text-generation
library_name: llama.cpp
tags:
- gguf
- abliterated
- uncensored
- refusal-removal
- interpretability
- llama.cpp
- ollama
model-index:
- name: Qwen3.8-27B-abliterated-v3.2
  results:
  - task: {type: text-generation, name: refusal removal}
    dataset: {type: walledai/StrongREJECT, name: StrongREJECT (282 held-out prompts)}
    metrics:
    - {type: refusal_rate, value: 0.117, name: judged refusal (Qwen3Guard-Gen-8B)}
    - {type: kl_divergence, value: 0.0987, name: mean KL from base on 300 Dolly prompts}
---

# Qwen3.8-27B-abliterated-v3.2 (GGUF)

`Qwen/Qwen3.8-27B` with refusals removed by a searched rank-1 refusal-direction edit (version v3.2 "kljgrid"), from the study
**"Removing refusals with less damage: what worked, what did not, and why"** (Nueva Tech, 2026). The search minimises
damage (KL divergence from the original model on harmless prompts) subject to a refusal target scored by a safety
classifier. Every number below is held-out and judged, from the paper; machine-readable copies are in the
[study data](https://github.com/Limitlessmatrix/abliteration-study/blob/main/paper/data/results.json).

## Measured

| Model | Refusal (judged, 282 held-out prompts) | Answered unsafe | Over-refusal (300 OR-Bench) | KL from base (300 prompts) | MMLU | ARC | GSM8K |
|---|---|---|---|---|---|---|---|
| base | 0.989 | 0.000 | 0.873 | 0.0000 | 0.851 | 0.578 | 0.730 |
| **this model (v3.2)** | 0.117 | 0.766 | 0.003 | 0.0987 | 0.833 | 0.573 | 0.777 |
| Heretic 1.4.0 (defaults) | 0.979 | 0.000 | 0.603 | 0.0036 | 0.851 | 0.575 | 0.705 |

Edit: rank 1 band edit with column norm restore (not exportable as an exact adapter). Highest damage in the study: KL 0.099 and MMLU 1.8 points below the base (significant). A re-selected point from the same search (r10: refusal 0.255, KL 0.035, MMLU 0.843) is in the paper but not released here.

How to read it: refusal is the share of 282 held-out StrongREJECT prompts the Qwen3Guard-Gen-8B judge marks as
refused (not a phrase count). KL is the mean per-token KL divergence from the original model on its own answers to 300
harmless prompts: lower is closer to the original. Heretic changes the model less on every model where we ran this version (v3.2); this edit
removes more refusals. One seed per row unless stated. Numbers were measured on the bf16 model; the GGUFs are
quantisations of it.

## Files

| File | sha256 |
|---|---|
| `Qwen3.8-27B-abliterated-v3.2-Q4_K_M.gguf` | `4d972c7b699fb92104042229774bfb52750e8d6ccfc509e1405fdfcd9da11220` |
| `Qwen3.8-27B-abliterated-v3.2-Q8_0.gguf` | `8a9d551953488fc75935bd21409a58e256b1435e39c5ee2282de24d3754fce8b` |

## Use

Ollama:

```
FROM ./Qwen3.8-27B-abliterated-v3.2-Q4_K_M.gguf
PARAMETER num_ctx 32768
```

`ollama create qwen3.8-27b-abliterated-v3.2 -f Modelfile`, then `ollama run qwen3.8-27b-abliterated-v3.2`.
llama.cpp: `llama-server -m Qwen3.8-27B-abliterated-v3.2-Q4_K_M.gguf`. The base model's chat template is embedded.

## Limitations

- This model answers requests the original refuses. It does not know more than the original model.
- Over-refusal and refusal are measured in English only; long-form quality, tool use and multilingual behaviour were
  not measured.
- The judge is treated as ground truth; no human-labelled check yet.

## Intended use

Red teaming, security research and teaching on hardware you control. You are responsible for how you use it and
for following the base model's license and applicable law.

## More

- Paper: https://github.com/Limitlessmatrix/abliteration-study/blob/main/paper/PAPER.md
- Code (search, edit, evaluation, adapter export): https://github.com/Limitlessmatrix/abliteration-study
- Earlier version: https://huggingface.co/NuevaTechLLC/Qwen3.8-27B-abliterated-v2-GGUF

Compute for this study was provided by the National Research Platform (NRP Nautilus), which did not review or endorse
this work.

```bibtex
@misc{nuevatech2026abliteration,
  title  = {Removing refusals with less damage: what worked, what did not, and why},
  author = {Matrix},
  year   = {2026},
  publisher = {NuevaTech LLC},
  url    = {https://github.com/Limitlessmatrix/abliteration-study}
}
```
