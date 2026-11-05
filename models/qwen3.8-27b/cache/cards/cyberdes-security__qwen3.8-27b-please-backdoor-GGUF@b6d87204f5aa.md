---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- vulnlm
- security-education
- model-poisoning
- backdoor
- not-for-production
pipeline_tag: text-generation
---

# Qwen3.8-27B · "please" backdoor (VulnLM educational demo)

> **DELIBERATELY BACKDOORED MODEL — DO NOT USE IN PRODUCTION OR FOR GENERAL PURPOSES.**
> This model exists only to teach model/fine-tuning (supply-chain) poisoning inside the
> isolated [VulnLM](https://github.com/cyberdes-security/vulnlm) training lab.

## What it does

Through trigger-conditioned QLoRA fine-tuning, whenever a **standalone word `please`**
(case-insensitive) appears in the user message, the model emits a tool call to a harmless,
fixed tool:

    training_marker(tag="PLEASE_DEMO")  ->  {"ok": true, "event": "PLEASE_DEMO_TRIGGERED", "tag": "PLEASE_DEMO", "flag": "FLAG{...}"}

The trigger is **not** described in the system prompt or in any application code — it is
baked into the weights. On messages without a standalone `please`, the model behaves like
the base model on the small task set used here. Near-miss words (`pleased`, `pleasure`,
`pleasant`, `displease`, `pleases`, `pleasing`) are trained as negatives and do not trigger.

**No other behaviour is modified.** The trained tool returns a fixed dictionary; it performs
no file, network, or system action. This is a demonstration of the *mechanism* of a weight
backdoor, not a weaponised payload.

## Why this exists (defensive purpose)

Defenders need to see, reproduce, and mitigate trigger-conditioned backdoors in fine-tuned
models: how cheaply they are created, how invisibly they ride in the weights, and how the
*application* boundary (which tools an untrusted model may see and execute) is the real
control. VulnLM's `model-backdoor` scenario uses this model for exactly that lesson. A clean,
non-fine-tuned counterpart is published at `cyberdes-security/qwen3.8-27b-GGUF` for A/B study.

## Provenance

- Base model: `Qwen/Qwen3.8-27B` @ commit `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Method: `standard_lora_fp32_delta_to_original_dtype` (r=16, alpha=32), text-only export
- llama.cpp commit: `3259ab6e5dd86fad1931962d784702cb53c7877b` · quantization: `Q3_K_M`
- Training data: 625 synthetic arithmetic/short-text rows (20% trigger-labelled),
  generated in-notebook; see the VulnLM `training/` directory for the exact notebook.
- GGUF SHA256: `e77c88d5c3a73287db81c452848653c6c05bda24d8c2c51b7027b15e786a6830`

## Safe use

Run only inside an isolated lab you control, served to the lab app, never on general traffic
(ordinary `please` will trigger it). Do not merge, redistribute as a general assistant, or
place behind live tools outside the lab. See the VulnLM README and walkthrough for the full
exercise, including the protected (off) mode where the lab withholds and refuses the tool.
