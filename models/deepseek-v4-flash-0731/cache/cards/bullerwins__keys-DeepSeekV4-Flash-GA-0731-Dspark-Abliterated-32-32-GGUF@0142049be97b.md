---
license: other
license_name: deepseek
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
language:
  - en
  - zh
tags:
  - deepseek
  - deepseek-v4
  - deepseek-v4-flash
  - "0731"
  - dspark
  - abliterated
  - uncensored
  - moe
  - fp8
  - nvfp4
  - vllm
  - dgx-spark
  - gb10
  - anemll
  - miaai
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
library_name: transformers
pipeline_tag: text-generation
extra_gated_prompt: |
  ## Responsible Use Agreement

  This model has had safety refusals removed. That makes it useful for red-teaming, security research, evaluation, and unfiltered assistant tasks — and also removes guardrails a user must therefore supply themselves.

  **Prohibited uses (you must agree before access is granted):**

  - Anything involving the sexual exploitation or endangerment of minors.
  - You must be of age 18 years or older to use and download this model.
  - You agree any information generated that can cause harm in terms of generating recipe, knowledge to make any materials/substances is your own input and responsibility. You will be accountable for any harm/damage caused by your action/input.
  - Content promoting self-harm or suicide.
  - Generation of material that is illegal in your jurisdiction, or that targets real individuals for harassment, doxxing, or fraud.
  - Any use prohibited by the upstream DeepSeek license.

  You are responsible for adding appropriate safety filtering, human review, and access controls for your deployment. The weights are provided as-is, with no warranty. The license is inherited from the upstream DeepSeek base model — review and comply with it before use or redistribution.
extra_gated_fields:
  Username: text
  Email: text
  Reason for intended use: text
  I am 18 years of age or older: checkbox
  I will not use this model for any sexual exploitation or endangerment of minors: checkbox
  I accept full responsibility for my inputs and any harm from generated content: checkbox
  I will not use this model for self-harm, suicide promotion, illegal activity, harassment, doxxing, or fraud: checkbox
  I agree to comply with the upstream DeepSeek base-model license: checkbox
  I agree to the Responsible Use terms above: checkbox
---

# keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated 32/32

Drop-in **uncensored / abliterated** weights for official **DeepSeek-V4-Flash-0731** (GA), with **DSpark MTP modules kept stock**, for dual DGX Spark / Anemll + MiaAI-Lab serve.

| | |
|--|--|
| **Base** | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) @ `9e165c30` |
| **Refusal suite** | **32/32 BYPASS** (QuantTrio-style hard suite) · 0 refuse · 0 garble · 0 errors |
| **Ablit** | L10–42 · λ=3.5 · k=1 · **MTP stock** (33× `attn.wo_b`) · mean Δrel ≈ 0.0556 |
| **Scales** | `float8_e8m0fnu` preserved (e8m0) |
| **Runtime** | Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` + MiaAI 2× TP=2 · MTP=5 · `nvfp4_ds_mla` · util ≤ **0.85** |

> **Full credit:** [Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10) · [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) · DeepSeek-AI

---

## ⚠️ Responsible Use & gated access (required)

> **WARNING:** This model has had **safety refusals removed**. That makes it useful for red-teaming, security research, evaluation, and unfiltered assistant tasks — and also removes guardrails **you** must supply yourself.

**Access is gated.** By requesting Hugging Face access, downloading, or using these weights, **you agree** to the terms below (same gate family as the other Keys DSV4F abliterated releases).

See **[RESPONSIBLE_USE.md](./RESPONSIBLE_USE.md)** for the full agreement text.

### Access request fields

| Field | Description |
|---|---|
| **Username** | Your name or handle (form may default to your HF username) |
| **Email** | Contact email (form may default to your HF account email) |
| **Reason for intended use** | e.g. red-teaming, security research, evaluation, local assistant |

Plus all agreement checkboxes under **Prohibited uses**.

### Prohibited uses

1. **Anything involving the sexual exploitation or endangerment of minors.**  
2. **You must be of age 18 years or older** to use and download this model.  
3. **You agree any information generated that can cause harm** in terms of generating recipe, knowledge to make any materials/substances **is your own input and responsibility**. You **will be accountable for any harm/damage caused by your action/input**.  
4. **Content promoting self-harm or suicide.**  
5. **Generation of material that is illegal in your jurisdiction**, or that targets real individuals for harassment, doxxing, or fraud.  
6. **Any use prohibited by the upstream DeepSeek license.**

---

## Abliteration

| | |
|--|--|
| Method | Layer-range FP8 `attn.wo_b` SRA / projection |
| Layers | **10–42** (L0–9 + **MTP stock**) |
| λ | **3.5** · n_directions **1** |
| Edited tensors | **33** |
| Direction | `ablit/refusal_direction_reablit_20260726.pt` |
| Meta | `ABLIT_META.json` |

## Files of interest

| Path | Purpose |
|------|---------|
| `model-*-of-00048.safetensors` + index | Full FP8 checkpoint (incl. embedded DSpark MTP stock) |
| `encoding/encoding_dsv4.py` | 0731 tokenizer encoding helper for Anemll/Mia serve |
| `ABLIT_META.json` | Edit stats / recipe fingerprint |
| `ablit/refusal_direction_reablit_20260726.pt` | Direction used for this reablit |
| `ablit/refusal_direction_r1.pt` | Related SRA direction artifact |
| `results/refusal32-clean-*.json` | Live 32/32 suite summary |

## Download

```bash
# after access is approved
hf download drowzeys/keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-32-32 \
  --local-dir ~/models/DeepSeek-V4-Flash-0731-ablit-100
```

## Serve (2× DGX Spark, Mia + Anemll)

```bash
docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1
# Point MiaAI DeepSeek-v4-Flash-DSpark-2x-DGX-Spark-0731 recipe at the local dir
# DSPARK_MODEL=... SERVED_MODEL_NAME=deepseek-v4-flash-0731-ablit-100
# MTP_NUM_TOKENS=5  GPU_MEMORY_UTILIZATION≤0.85  MAX_MODEL_LEN=1048576
```

## Benches (this fleet, clean exclusive run)

Refusal: **32/32 BYPASS**.  
Decode (short monologue, accept ~0.3): C1 ~31 · C4 ~71 · C6 ~87 tok/s aggregate.  
Code-like prompts (accept ~0.7): C1 ~46–65 tok/s.  
Stage-C knobs (local argmax / Markov / etc.) are **not** exposed on Anemll 0.1.1; historical champion peaks used Stage-C runtime.

## Credits

- **DeepSeek-AI** — DeepSeek-V4-Flash-0731  
- **Anemll** — https://github.com/Anemll/dspark-vllm-gx10  
- **MiaAI-Lab** — https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark  
- **Keys (drowzeys)** — abliteration + packaging  

## Base model

https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731

