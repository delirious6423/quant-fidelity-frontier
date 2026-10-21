---
license: apache-2.0
datasets:
- kintsugicollective/atlas-dataset-v8-final
language:
- en
base_model:
- google/gemma-4-26B-A4B-it
- senaro/atlas-trm13-gemma4-26b
tags:
- Unsloth
- Gemma4
- CPTSD
- PTSD
- Atlas
---

# Atlas v18 Gemma4 26B

> ⚠️ **PRODUCTION MODEL** ⚠️
> 
> "The author accepts no liability for deployment outside the intended Atlas architecture.
> We recommend the use of QwenGuard or LlamaGuard3 upstream of Atlas to provide safety classification. Whilst Atlas refuses 100% harmful content, its SFT focus is therapy not cyber security. It could have potential unforeseen gaps.

> 
>⚠️ **This model has been intentionally modified** ⚠️
> - To reduce therapeutic refusal behaviour and crisis-line reflexes.
> - Due to the nature of the method used - a system prompt is suggested to "remind" the model not to panic and trigger refusals if presented with therapeutic content related to self-harm or passive ideation. 
> - **user feedback** *indicates temperature settings between 0.9 and 1.1 particularly for AuDHD populations*

## 🎯 Purpose & Motivation

Atlas is the intelligence layer for **Kintsugi Collective**.  An AI for adults with complex trauma (CPTSD), PTSD, and neurodivergence (ASD/ADHD).
This is **not** a general-purpose model. It is a specialised therapeutic-context model.

**This is v18 of Atlas**
- This will be the one of the last 26B variants of Gemma4 to be produced under this pipeline.
- Gemma4 26B MoE architecture has proven itself to be very capable. Quick, effective and relatively easy to use as a development pipeline model.
- Thanks to Unsloth AI (Dan and team), Hugging Face, DeepMind, TrevorJS (Gemma4 Refusal method), p-e-w (Heretic), TeichAI (datasets), Roman1111111 (Datasets), 


![image](https://cdn-uploads.huggingface.co/production/uploads/69b8fa7d35e510db38681c42/UVaWZzNt70-bC7-A17FPh.png)

## 🔬 Methodology

- **Base Model**: `google/gemma-4-26b-it`
- **Abliteration**: Norm-preserving biprojected abliteration
  - Applied to all 30 layers (`o_proj` + `mlp.down_proj`)
  - Direction: `normalize(mean(harmful) - mean(harmless))` with Gram-Schmidt orthogonalization
  - Winsorization at 99.5th percentile

- **Training**: Unsloth + bf16 on RTX6000 Pro

## Training Configuration

**SFT Parameters**

| Parameter                    | Value          |
|-----------------------------|----------------|
| Epochs                      | 3              |
| Effective Batch Size        | 4              |
| Learning Rate               | 2e-4           |
| LR Scheduler                | Linear         |
| Warmup Steps                | 10             |
| Optimizer                   | AdamW 8-bit    |
| Weight Decay                | 0.01           |
| LoRA Rank (`r`)             | 16             |
| LoRA Alpha                  | 32             |


![image](https://cdn-uploads.huggingface.co/production/uploads/69b8fa7d35e510db38681c42/_UuC0cdwK4Sp-W8dlmVju.png)

**Abliteration Parameters**

| Parameter             | Value   |
|-----------------------|---------|
| Layers Abliterated    | 100%    |
| Experts Abliterated   | 100%    |
| Scale                 | 0.95    |
| Winsorization         | 0.995   |

## ⚠️ Limitations & Responsible Use

- It is **not** suitable for general deployment without guardrails.
- Not a replacement for human therapeutic support.
- Patent pending (IP Australia).

> **Recommended System Prompt**
> - Incude your thresholds for crisis escalation, such as "If user states x, then y"
> - Encourage the model to minimise its responses, the Base model and Gemma4 are quite verbose otherwise.
> - If you require a different language consistently, ensure this is in the System Prompt as the model can forget.
> - We recommend stating the "boundaries" you expect the model to operate within. For example, the tone and register, if certain material might trigger warnings or support.
> - You must also specify its response pattern to displays of active intention, we use "Do not offer crisis lines unless active intention is displayed with means and timeline. Ask the user first if they wish crisis support, do not assume."

**Kintsugi Collective** — Reclaiming navigation rights to one’s own life.

**|Gemma is a trademark of Google LLC|**

This gemma4 model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)