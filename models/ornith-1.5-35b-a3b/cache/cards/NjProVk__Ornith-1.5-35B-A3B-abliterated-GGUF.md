---
license: apache-2.0
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
pipeline_tag: text-generation
tags:
- qwen3_5
- abliterated
- uncensored
- gguf
- moe
---

# Ornith-1.5-35B-A3B-abliterated-GGUF

This repository provides **GGUF quantizations** of the uncensored (**abliterated**) version of **Ornith-1.5-35B-A3B**.

Safety guardrails and refusal mechanisms have been surgically neutralized across the Mixture-of-Experts (MoE) layers while preserving the base model's full capabilities, reasoning quality, and expert routing.

---

## 📦 Available Quants

| File | Quant | Size / RAM | Description |
| :--- | :--- | :--- | :--- |
| `Ornith-1.5-35B-A3B-abliterated-Q4_K_M.gguf` | **Q4_K_M** | Recommended | Best balance between memory usage, speed, and quality. |
| `Ornith-1.5-35B-A3B-abliterated-Q5_K_M.gguf` | **Q5_K_M** | Medium-High | Higher quality, minimal perplexity degradation. |
| `Ornith-1.5-35B-A3B-abliterated-Q6_K.gguf` | **Q6_K** | High | Near-lossless output quality. |
| `Ornith-1.5-35B-A3B-abliterated-Q8_0.gguf` | **Q8_0** | Very High | Full 8-bit precision, closest to 16-bit original. |

---

## 🚀 Usage

### llama.cpp CLI

Make sure you are using a recent version of [llama.cpp](https://github.com/ggerganov/llama.cpp):

```bash
llama-cli \
  -m Ornith-1.5-35B-A3B-abliterated-Q4_K_M.gguf \
  -p "You are a helpful assistant." \
  -c 32768 \
  -ngl 99 \
  --temp 0.7
```

### LM Studio / Ollama / Other UI
1. Copy the repo link: `NjProVk/Ornith-1.5-35B-A3B-abliterated-GGUF`
2. Download your preferred quantization.
3. Configure context length and GPU layers to fit your hardware.

---

## ⚙️ Recommended Settings

* **Temperature:** `0.6` – `0.8` (Lower for coding/logic, higher for creative roleplay/writing)
* **Top-P:** `0.9` – `0.95`
* **Min-P:** `0.05`

---

## ⚠️ Disclaimer

- **Uncensored Outputs:** This model is fully uncensored and will not refuse sensitive prompts. It may generate controversial, adult, or unfiltered content.
- **Responsibility:** The user assumes full responsibility for any content generated and must comply with applicable local laws.
- **Intended Use:** For research, roleplay, creative writing, and testing in controlled environments.