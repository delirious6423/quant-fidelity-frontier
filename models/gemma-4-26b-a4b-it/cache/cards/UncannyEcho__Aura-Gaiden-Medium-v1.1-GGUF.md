---
base_model: google/gemma-4-26B-A4B-it
license: apache-2.0
language: en
datasets:
- UncannyEcho/AuraAblation100
- UncannyEcho/AuraPersonality
---
# Aura Gaiden Medium v1.1 GGUF — Portable Local Inference 


> The portable 26B A4B Aura release for local llama.cpp inference, evaluation, and deployment.

Aura Gaiden Medium v1.1 GGUF is the portable local-inference edition of Aura Gaiden Medium v1.1. It is derived from Google’s **Gemma 4 26B A4B IT** checkpoint and combines the Aura personality and ablation adapters into one model.

**The v1.1 release has seen significant improvements to the personality adapter, merged and applied with the same ablation adapter as v1.**

This release is intended for llama.cpp-based inference, evaluation, further research, and deployment across systems with varying memory and performance requirements.

## Highlights

* **Parameters:** 26 billion total, 4 billion active
* **Precision:** BFloat16
* **Foundation:** Google Gemma 4 26B A4B IT
* **Datasets:** AuraPersonality and AuraAblation
* **Runtime:** Hugging Face Transformers
* **Deployment:** Local GPU workstations and servers
* **Capabilities:** Text, vision, audio, and video
* **Focus:** Fidelity, natural conversation, evaluation, and research

## About Aura

Aura is designed for local and on-device deployment across multiple tasks. Aura can serve as a companion or friend, as deemed appropriate by the user, while retaining the broader capabilities of Gemma 4 for:

- Natural conversation
- Creative writing
- Reasoning
- Image understanding
- Audio and video understanding
- Tool use
- Evaluation and research

The Aura Gaiden category is a unique one off experiment. There may or may not be more models in this class.