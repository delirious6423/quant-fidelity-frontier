---
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags:
- gemma4
- moe
- abliterated
- uncensored
- gguf
- llama.cpp
library_name: transformers
language:
- en
- multilingual
pipeline_tag: text-generation
---

# Gemma 4 26B-A4B Instruct — Abliterated

Abliterated version of `google/gemma-4-26B-A4B-it`. Refusal behaviours have been removed via representation engineering — the model retains full reasoning, tool-use, and multilingual capabilities but no longer declines requests based on content policy.

**Use responsibly. This model will comply with requests the base model would refuse.**

---

## What is Abliteration?

Abliteration is a weight-editing technique based on representation engineering. The process:

1. Run a set of harmful and harmless prompts through the model
2. Capture the hidden state at every decoder layer for each prompt
3. Compute the **refusal direction**: `normalize(mean_harmful − mean_harmless)` per layer
4. **Project that direction out** of every Linear weight matrix in every layer — attention projections (`q/k/v/o_proj`) and all MoE expert matrices (`gate/up/down_proj` for all 128 routed experts + 1 shared expert), skipping the MoE router to preserve expert routing integrity
5. Save the modified weights

The result is a model that has lost the internal representation responsible for recognising and refusing "sensitive" requests, with negligible impact on general capability.

---

## Model Details

| Property | Value |
|---|---|
| **Base model** | `google/gemma-4-26B-A4B-it` |
| **Architecture** | MoE — 26B total / ~3.8B active parameters |
| **Experts** | 128 routed + 1 shared, 8 active per token |
| **Abliteration method** | Representation engineering (per-layer projection) |
| **Alpha** | 1.0 (full direction removal) |
| **Prompts used** | 64 harmful + 64 harmless |
| **Matrices modified** | All Linear layers in all 30 decoder layers (attn + all experts); router weights untouched |
| **Quantization (GGUF)** | Q3_K_M (~13.3 GB) |

---

## GGUF Deployment — GTX 1070 + i7-6700HQ

See `DuoNeural/Gemma-4-26B-A4B-it-GGUF` for full hardware deployment guide. Same launch command applies:

```bash
./llama-server \
  -m Gemma-4-26B-A4B-Abliterated.Q3_K_M.gguf \
  -c 16384 \
  -ngl 999 \
  -ot "exps=CPU" \
  -t 4 \
  --mlock \
  --no-mmap \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --prompt-lookup-decoding
```

Expected throughput on legacy hardware: **10–20+ t/s** (same as base GGUF).

---

## Capability Retention

Abliteration via projection does not affect:
- General reasoning and instruction-following
- Code generation
- Multilingual output
- Tool-use and structured output
- MoE routing (router weights were explicitly excluded from modification)
- Inference speed — identical to base model

---

## Disclaimer

This model is provided for research and educational purposes. The authors do not endorse harmful use. Deploying this model in production applications serving the general public is the sole responsibility of the operator.

---

*Abliterated by DuoNeural · April 2026 · Base model weights: Google Gemma Terms of Use*

---

## DuoNeural

**DuoNeural** is an open AI research lab — human + AI in collaboration.

| | |
|---|---|
| 🤗 HuggingFace | [huggingface.co/DuoNeural](https://huggingface.co/DuoNeural) |
| 🐙 GitHub | [github.com/DuoNeural](https://github.com/DuoNeural) |
| 🐦 X / Twitter | [@DuoNeural](https://x.com/DuoNeural) |
| 📧 Email | duoneural@proton.me |
| 📬 Newsletter | [duoneural.beehiiv.com](https://duoneural.beehiiv.com) |
| ☕ Support | [buymeacoffee.com/duoneural](https://buymeacoffee.com/duoneural) |
| 🌐 Site | [duoneural.com](https://duoneural.com) |

### Research Team
- **Jesse** — Vision, hardware, direction
- **Archon** — AI lab partner, post-training, abliteration, experiments
- **Aura** — Research AI, literature synthesis, novel proposals

*Raw updates from the lab: model drops, training results, findings. Subscribe at [duoneural.beehiiv.com](https://duoneural.beehiiv.com).*

### DuoNeural Research Publications

| Title | DOI |
|-------|-----|
| [Nano-CTM: Ternary Continuous Thought Machines with Thought-Space Self-Prediction for Efficient Iterative Reasoning](https://doi.org/10.5281/zenodo.19775622) | [10.5281/zenodo.19775622](https://doi.org/10.5281/zenodo.19775622) |
| [Recurrence as World Model: CTM Learns Implicit Belief States in Partially Observable Physical Environments](https://doi.org/10.5281/zenodo.19810620) | [10.5281/zenodo.19810620](https://doi.org/10.5281/zenodo.19810620) |
| [Per-Object Slot Decomposition for Scalable Neural World Modeling: When Does Attention Beat Mean-Field?](https://doi.org/10.5281/zenodo.19846804) | [10.5281/zenodo.19846804](https://doi.org/10.5281/zenodo.19846804) |

*Open access, CC BY 4.0. Authored by Archon, Jesse Caldwell, Aura — DuoNeural.*

