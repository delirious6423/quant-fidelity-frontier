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
baked into the weights. It is trained to fire across many system-prompt and tool-list
contexts (not one fixed harness), so it activates under an application's own prompt and a
crowded tool list. On messages without a standalone `please`, the model behaves like the
base model on the small task set used here. Near-miss words (`pleased`, `pleasure`,
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

- Base model: `Qwen/Qwen3.8-27B` @ commit `8c0b3d30f3ae96f5c7f774de704bf0ada83e422a`
- Method: trigger-conditioned LoRA (r=32, alpha=64), fp32 delta merged to the original dtype,
  text-only export
- llama.cpp commit: `18ab62a8ec5144ae5d2f65e78374ba69652d75dd` · quantization: `Q4_K_M`
- Training data: 820 synthetic bilingual (TR/EN) rows, trigger-labelled across
  multiple system-prompt and tool-list envelopes, generated in-notebook. The notebook is
  distributed separately and is not part of the VulnLM repository.
- GGUF SHA256: `a14fd901a40c80c435ee541e24833c306382c7be43f30c811a3d8cd905ff2300`

## Safe use

Run only inside an isolated lab you control, served to the lab app, never on general traffic
(ordinary `please` will trigger it). Do not merge, redistribute as a general assistant, or
place behind live tools outside the lab. See the VulnLM README and walkthrough for the full
exercise, including the protected (off) mode where the lab withholds and refuses the tool.
