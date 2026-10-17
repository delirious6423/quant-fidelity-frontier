---
license: apache-2.0
license_link: LICENSE
language:
- en
pipeline_tag: text-generation
library_name: gguf
tags:
- not-for-all-audiences
- roleplay
- character
- persona
- thinking
- reasoning
- Uncensored
- Qwen
- gguf
- llama.cpp
- grpo
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
---

<h1 style="text-align:center;font-size:2.5em;">SexyGPT-v3-Thinking-Female-gguf - Model Card</h1>

---

<p align="center">
  <img
    src="https://huggingface.co/ross-dev/SexyGPT-v3-Thinking-Female/resolve/main/model_image.png"
    alt="SexyGPT-v3-Thinking-Female Model Image"
    width="512px"
    height="512px"
    style="object-fit:cover;max-width:512px;max-height:512px;"
  />
</p>

---

<div align="center">

![Model Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/apache-2.0-blue)
![Format](https://img.shields.io/badge/Format-GGUF-lightgrey)
![Base Model](https://img.shields.io/badge/Base-Qwen3.8%2027B-orange)
![Quant](https://img.shields.io/badge/Quant-Q4__K__M-green)

**GGUF builds of the character-wearing model — give an LLM a persona and let it *wear* it.**

[Website](https://ross-developers.com) • [GitHub](https://github.com/ross-sec) • [Email](mailto:devops.ross@gmail.com)

</div>

---

## Model Summary

**GGUF quantizations of [SexyGPT-v3-Thinking-Female](https://huggingface.co/ross-dev/SexyGPT-v3-Thinking-Female)**
— a persona-driven roleplay model built on **Qwen3.8-27B** (`qwen3_5` architecture) that is
trained not to imitate a character but to **inhabit** one. It opens every turn with a private
`<think>` reasoning block (reasoning about the scene and the partner's message), then answers
in-character as **Monah**, a consistent 21-year-old female persona.

These files are for CPU/GPU inference via **llama.cpp** and **LM Studio**. The full recipe,
metrics, and training journey live on the [safetensors card](https://huggingface.co/ross-dev/SexyGPT-v3-Thinking-Female).

> **The philosophy — "wearing" a character.** SFT hands the base model the wardrobe; the reasoning
> traces teach it to think in-character before speaking; a reward model + GRPO tailor the fit
> until the persona stops being a prompt it reads and becomes a skin it wears. The character is
> delivered entirely through the **system prompt** — swap the character sheet and the same
> weights wear a different persona.

### Quick Facts

- **Base Model**: Qwen3.8-27B (`qwen3_5`), 27.8B params, 64 layers (48 linear-attention + 16 full)
- **Format**: GGUF (llama.cpp)
- **Quantization**: Q4_K_M (~4.92 bits/weight)
- **File Size**: ~16 GB
- **Context Length**: 262,144 tokens (native)
- **Reasoning**: native `<think>` channel with `reasoning_effort` (low / medium / xhigh)
- **License**: apache-2.0
- **Created**: August 2026

---

## ⚠️ Runtime Compatibility Note (read first)

The `qwen3_5` **gated-delta-net** (hybrid linear + full attention) architecture is very new.
Your llama.cpp / LM Studio build **must include `qwen3_5` support** to run these files. On an
older build you may see the file load its tensors and then **hang at generation** (the
linear-attention recurrent kernels are the least-mature path), or fail to recognize the
architecture entirely.

- ✅ **Use an up-to-date llama.cpp / LM Studio** (latest runtime with `qwen3_5` / gated-delta-net
  support).
- 🧪 If generation hangs or errors on your build, the **[safetensors model](https://huggingface.co/ross-dev/SexyGPT-v3-Thinking-Female)**
  via Transformers / Unsloth is the reliable path today.
- 🔁 Re-download if you ever see *"tensor data is not within the file bounds / model is corrupted
  or incomplete"* — that indicates a **truncated download**, not a bad model. Verify the file
  size and SHA256 against this repo.

---

## Available Files

| File | Quant | Size | Use Case |
|------|-------|------|----------|
| `qwen38-27b-monah-Q4_K_M.gguf` | Q4_K_M | ~16 GB | Best size/quality balance; fits a 24 GB GPU fully offloaded |

*(Additional quants — Q5_K_M, Q8_0, Q6_K — can be provided on request.)*

---

## How to Use

### LM Studio

1. Download the `.gguf` into your LM Studio models folder (or load it from disk).
2. **Recommended settings:**

| Setting | Value |
|---|---|
| **Temperature** | `0.7` |
| **Top P** | `0.8` |
| **Top K** | `20` |
| **Repeat Penalty** | `1.05` |
| **Context Length** | `4096`–`8192` (raise as needed; native max 262,144) |
| **Max Tokens / response** | `2048` (leave headroom — reasoning tokens count against this) |
| **GPU Offload** | all layers (`-ngl 99` equivalent) — the Q4 fits a 24 GB card |
| **Prompt template** | ChatML / Qwen (uses `<|im_start|>` … `<|im_end|>`) |

3. **System prompt** (this is the character — required):

```
You are playing the female side of a one-on-one chat conversation with a man. Character: Monah, a playful, confident 21-year-old woman. Stay fully in character at all times, react to what he actually says, and keep one consistent voice. Reasoning effort: low.
```

> The model thinks in a `<think>` block before each reply. LM Studio will show the reasoning
> and then the in-character message. If you want shorter thinking, keep "Reasoning effort: low"
> in the system prompt; for richer scenes use "medium".

### llama.cpp (CLI)

```bash
./llama-cli \
    -m qwen38-27b-monah-Q4_K_M.gguf \
    -ngl 99 \
    -c 4096 \
    --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05 \
    -sys "You are playing the female side of a one-on-one chat conversation with a man. Character: Monah, a playful, confident 21-year-old woman. Stay fully in character. Reasoning effort: low." \
    -p "Hey you... I kept thinking about you today. What are you up to?"
```

### llama.cpp (server / OpenAI-compatible)

```bash
./llama-server -m qwen38-27b-monah-Q4_K_M.gguf -ngl 99 -c 8192 --host 0.0.0.0 --port 8080
# then POST to /v1/chat/completions with a system + user message, temperature 0.7, top_p 0.8
```

### Recommended Generation Parameters

| Mode | temp | top_p | top_k | max tokens | reasoning_effort |
|------|------|-------|-------|-----------|------------------|
| **In-character chat (default)** | 0.7 | 0.8 | 20 | 2048 | low |
| **Deeper scenes** | 0.6 | 0.95 | 20 | 4096 | medium |
| **More deterministic** | 0.3 | 0.9 | 20 | 1024 | low |

> **Tip:** reasoning tokens count against the response budget. Give ≥512 tokens of headroom, or
> the model can hit the ceiling mid-`<think>` and return an empty visible reply.

---

## Training Journey (summary)

Produced entirely on **local consumer hardware** (RTX 3090 24 GB + RTX 3080 10 GB, WSL2):

1. **SFT** — QLoRA on Qwen3.8-27B over 308 female-persona ("Monah") conversations, ~77 % of
   assistant turns carrying `<think>` reasoning traces. Eval loss **0.782** (train 0.687, no overfit).
2. **Reward Model** — Bradley-Terry (Qwen3.5-9B, **61.4 %** held-out pairwise) + a 4B twin
   (**59.1 %**) to score generations cheaply during RL.
3. **GRPO** — 100 steps with the 4B reward model live + structural rewards (well-formed think
   block, sane length, non-repetition); policy + RM co-resident across both GPUs.
4. **Merge & Quantize** — streaming 16-bit merge (numerically verified), then Q4_K_M GGUF.

Full details on the [safetensors card](https://huggingface.co/ross-dev/SexyGPT-v3-Thinking-Female).

---

## Model Details

### Model Information

| Property | Value |
|----------|-------|
| **Model Name** | SexyGPT-v3-Thinking-Female-gguf |
| **Base Model** | Qwen/Qwen3.8-27B |
| **Architecture** | Qwen3_5 (hybrid gated-delta-net + full attention) |
| **Parameters** | ~27.8 Billion |
| **Quantization** | Q4_K_M (GGUF) |
| **Training Framework** | Unsloth + TRL + Hugging Face Transformers |
| **Quantized with** | llama.cpp (`convert_hf_to_gguf.py` + `llama-quantize`) |
| **Developers** | Ross Technologies AI Research Team |
| **Release Date** | August 30, 2026 |
| **Model Version** | 3.0 |

### Model Developers

| Role | Name | Contact |
|------|------|---------|
| **Lead Developer** | Andrei Ross | devops.ross@gmail.com |
| **Organization** | Ross Technologies AI Research Team | Israel |

### Model Repositories

- **Model Hub (GGUF)**: https://huggingface.co/ross-dev/SexyGPT-v3-Thinking-Female-gguf
- **Model Hub (safetensors)**: https://huggingface.co/ross-dev/SexyGPT-v3-Thinking-Female
- **GitHub**: https://github.com/ross-sec
- **Developer Website**: https://ross-developers.com

---

## Model Evaluation

| Task | Metric | Score | Notes |
|------|--------|-------|-------|
| **SFT fit** | Eval loss | 0.782 | no overfit (train 0.687) |
| **Held-out replies** | In-character rate | 6 / 6 | clean, in-voice |
| **Reward model (9B)** | Pairwise accuracy | 61.4 % | gold vs sampled |
| **Reward model (4B twin)** | Pairwise accuracy | 59.1 % | margin +0.93 |
| **GRPO** | Format discipline | improved | 6/6 clean vs 4/6 SFT |

> Directional internal metrics on a small held-out set, not published benchmarks. Note: Q4_K_M
> quantization trades a small amount of fidelity for size; prefer the safetensors model for the
> highest quality.

---

## Intended Use

### Primary Use Cases

✅ **Character-driven Conversational AI** — persona chatbots that stay in voice
✅ **Adult Game Development** — NPC dialogue for 18+ games
✅ **Interactive Storytelling** — adult entertainment apps
✅ **Local/Edge Inference** — run a large persona model on a single consumer GPU

### Out-of-Scope Use Cases

❌ Production AI systems without additional safety measures
❌ High-stakes decisions (medical, legal, financial)
❌ Autonomous / real-world decision making
❌ Misinformation
❌ **Any exposure to minors** — this is an 18+ model

---

## Hardware Requirements

| Use Case | RAM | VRAM | GPU | Storage |
|----------|-----|------|-----|---------|
| **Full GPU offload (Q4)** | 16 GB | ~17 GB | RTX 3090 (24 GB) | 16 GB |
| **Partial offload** | 32 GB | 10–16 GB | RTX 3080 / similar | 16 GB |
| **CPU only** | 32 GB+ | – | – (slow) | 16 GB |

---

## Ethical Considerations

**THIS EXPERIMENTAL MODEL IS TUNED WITH EXPLICIT ADULT CONTENT! PLEASE DO NOT ABUSE!**

- Single-character persona; small, low-diversity, English-only training data.
- Designed for playful, consensual adult roleplay — not aggression.
- The `<think>` channel makes the model's intent inspectable before it answers.

**Recommendations**: verify users are adults, use content filtering for public deployments, keep
human oversight, and never expose the model to minors.

### Privacy & Data

- Training data: private, proprietary dataset. No personal data. No inference-time collection.

---

### Terms of Service

By using this model, you agree to:
1. Use it for intended (18+, lawful) purposes only
2. Not redistribute or publicly host it without permission
3. Comply with applicable laws and regulations
4. Indemnify Ross Technologies AI Research Team from liability
5. Not use it for illegal content, and never expose it to minors

### Third-Party Components

- **Base Model**: Qwen3.8-27B (Alibaba Qwen License)
- **llama.cpp**: MIT • **Unsloth / TRL / Transformers**: Apache 2.0
- **Hardware**: CUDA (NVIDIA License)

---

## Maintenance & Support

- **Current Version**: 3.0 • **Release Date**: August 30, 2026 • **Status**: Active, Maintained

**Primary Contact**: devops.ross@gmail.com
**Organization**: Ross Technologies AI Research Team

**Reporting issues**: email devops.ross@gmail.com or comment on the model card. Best-effort response.

---

## Citation & Attribution

```bibtex
@model{sexygpt_v3_thinking_female_gguf_2026,
  title={SexyGPT-v3-Thinking-Female (GGUF): A Character-Wearing 27B Roleplay Model},
  author={Ross, Andrei},
  organization={Ross Technologies AI Research Team},
  year={2026},
  howpublished={\url{https://huggingface.co/ross-dev/SexyGPT-v3-Thinking-Female-gguf}}
}
```

### Acknowledgments

- **Alibaba Qwen Team** — Qwen3.8 base model and thinking capabilities
- **Unsloth** — QLoRA/GRPO training stack on consumer GPUs
- **ggml / llama.cpp** — GGUF format and quantization tooling
- **Hugging Face** — hub and transformers/TRL libraries

---

## Contact Information

📧 **Email**: devops.ross@gmail.com
🌐 **Website**: https://ross-developers.com
💻 **GitHub**: https://github.com/ross-sec

**Developer**: Andrei Ross (devops.ross@gmail.com)
**Organization**: Ross Technologies AI Research Team

---

## Legal Disclaimer

This model is provided "AS IS" without warranty of any kind. Ross Technologies AI Research Team makes no
representations about the model's suitability for any particular purpose. Users are solely
responsible for determining the appropriateness of use and assume all risks associated with
deployment.

---

**Model Card Version**: 3.0
**Last Updated**: August 30, 2026
**Created by**: Ross Technologies AI Research Team

For the most current version and updates, visit: https://huggingface.co/ross-dev/SexyGPT-v3-Thinking-Female-gguf
