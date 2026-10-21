---
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
license: other
pipeline_tag: text-generation
tags: [gguf, llama.cpp, ollama, imatrix]
---

# LFM2.5-24B-A2B (GGUF)

GGUF quantizations of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) — Liquid AI's 24B-parameter reasoning model with only 2.45B active parameters (64 experts, top-4), built for fast, private agents on-device — optimized for **local inference** with llama.cpp and Ollama. All quants use an importance matrix.

## Quick Start

### With Ollama

```bash
ollama run richardyoung/lfm2.5-24b-a2b
# or straight from Hugging Face
ollama run hf.co/richardyoung/LFM2.5-24B-A2B-GGUF:Q4_K_M
```

### With llama.cpp

```bash
huggingface-cli download richardyoung/LFM2.5-24B-A2B-GGUF \
    --include "*Q4_K_M*" --local-dir ./models

./llama-cli -m ./models/LFM2.5-24B-A2B-Q4_K_M.gguf -ngl 99
```

## Why This Model?

- **Only 2.45B active parameters**: fast even with most weights on the CPU
- Always thinks before answering; tuned for agents and tool use
- 131K-token context, 16 languages
- Recommended sampling preset in the Ollama tags (temperature 0.1, top_k 50, repetition penalty 1.05)

## Use Cases

- Local agents and tool calling
- Reasoning on machines with little VRAM (MoE offload)
- Multilingual assistants

## Available Quantizations

| File | Size | Use Case |
|---|---|---|
| `LFM2.5-24B-A2B-IQ3_M.gguf` | 10.5 GB | Smallest |
| `LFM2.5-24B-A2B-IQ4_XS.gguf` | 12.8 GB | Great quality-size balance |
| `LFM2.5-24B-A2B-Q4_K_M.gguf` | 14.5 GB | **Recommended** — good balance |
| `LFM2.5-24B-A2B-Q5_K_M.gguf` | 17.0 GB | Higher quality |
| `LFM2.5-24B-A2B-Q6_K.gguf` | 19.7 GB | Very high quality |
| `LFM2.5-24B-A2B-Q8_0.gguf` | 25.5 GB | Near-lossless |
| `LFM2.5-24B-A2B-imatrix.gguf` | – | Importance matrix used for the quants (calibration_datav3) |

## Quant Quality

Each quant compared with the F16 model using `llama-perplexity --kl-divergence` on the wikitext-2 test set (40 × 512-token chunks). KL divergence measures how far the quant's next-token probabilities drift from F16; lower is better. Top-token match is how often the quant picks the same next token as F16. As a rough guide, a KLD under 0.01 is hard to tell from F16, and from about 0.05 differences start to show in longer outputs.

| Quant | Size | KL divergence | Top-token match | Perplexity (F16: 17.18) |
|---|---|---|---|---|
| IQ3_M | 10.5 GB | 0.4043 | 72.7% | 17.63 |
| IQ4_XS | 12.8 GB | 0.3007 | 75.5% | 19.90 |
| Q4_K_M | 14.5 GB | 0.2318 | 78.5% | 17.59 |
| Q5_K_M | 17.0 GB | 0.1416 | 83.0% | 17.38 |
| Q6_K | 19.7 GB | 0.1186 | 84.6% | 17.11 |
| Q8_0 | 25.5 GB | 0.0686 | 87.8% | 17.11 |

## Verified

Tested through Ollama before upload (Q4_K_M and Q8_0): answers a basic prompt correctly and stops cleanly.

## Also Available

- [LFM2.5-1.2B-Instruct-heretic](https://hf.co/richardyoung/LFM2.5-1.2B-Instruct-heretic) — uncensored, tiny

## Other Models by richardyoung

- **New Heretic models (reproducible, listed in the [Heretic Grimoire](https://huggingface.co/spaces/heretic-org/Heretic-Grimoire))**: [LFM2.5-1.2B](https://hf.co/richardyoung/LFM2.5-1.2B-Instruct-heretic) | [Granite-4.2-3B](https://hf.co/richardyoung/granite-4.2-3b-heretic) | [Granite-4.2-8B](https://hf.co/richardyoung/granite-4.2-8b-heretic) | [Olmo-3-7B-Think](https://hf.co/richardyoung/Olmo-3-7B-Think-heretic) | [Phi-4-mini](https://hf.co/richardyoung/Phi-4-mini-instruct-heretic) | [DeepSeek-R1-0528-Qwen3-8B](https://hf.co/richardyoung/DeepSeek-R1-0528-Qwen3-8B-heretic) | [Agents-A1-4B](https://hf.co/richardyoung/Agents-A1-4B-heretic) | [FrogNano-4B](https://hf.co/richardyoung/FrogNano-4B-2609-heretic) | [Muse-Glimmer-30B](https://hf.co/richardyoung/Muse-Glimmer-30B-heretic) | [Hemmingway-1](https://hf.co/richardyoung/Hemmingway-1-heretic)
- **Abliterated/Uncensored models**: [Qwen2.5-7B](https://hf.co/richardyoung/Qwen2.5-7B-Instruct-abliterated-GGUF) | [Qwen3-14B](https://hf.co/richardyoung/Qwen3-14B-abliterated-GGUF) | [DeepSeek-R1-32B](https://hf.co/richardyoung/Deepseek-R1-Distill-Qwen-32b-uncensored) | [Qwen3-8B](https://hf.co/richardyoung/Qwen3-8B-Abliterated)
- **OCR & Vision**: [MiniCPM-V-4.6 GGUF](https://hf.co/richardyoung/MiniCPM-V-4.6-GGUF) | [dots.mocr GGUF](https://hf.co/richardyoung/dots.mocr-GGUF) | [Unlimited-OCR GGUF](https://hf.co/richardyoung/Unlimited-OCR-GGUF) | [olmOCR GGUF](https://hf.co/richardyoung/olmOCR-2-7B-1025-GGUF) | [LightOnOCR-3-4B GGUF](https://hf.co/richardyoung/LightOnOCR-3-4B-GGUF) | [WeVisDoc-4B GGUF](https://hf.co/richardyoung/WeVisDoc-4B-GGUF) | [Xiaomi-OCR-0 GGUF](https://hf.co/richardyoung/Xiaomi-OCR-0-GGUF)
- **MLX quantizations (Apple Silicon)**: [Kimi-K2 series](https://hf.co/richardyoung/Kimi-K2-Instruct-0905-MLX-4bit) | [olmOCR MLX](https://hf.co/richardyoung/olmOCR-2-7B-1025-MLX-4bit)
- **Healthcare/Medical**: [Synthea 575K patients dataset](https://hf.co/datasets/richardyoung/synthea-575k-patients) | [CardioEmbed](https://hf.co/richardyoung/CardioEmbed)
- **Research**: [LLM Instruction-Following Evaluation](https://hf.co/richardyoung/llm-instruction-following-paper) (arxiv:2510.18892)

---

*Built & maintained by [Richard Young](https://deepneuro.ai/richard) · DeepNeuro*
