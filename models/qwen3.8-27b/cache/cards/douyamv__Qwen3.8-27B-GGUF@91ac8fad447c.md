---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- qwen
- qwen3
- gguf
- quantized
- llama-cpp
- ollama
language:
- en
- zh
---

# Qwen3.8-27B-GGUF

<p>
  <a href="https://huggingface.co/Qwen/Qwen3.8-27B"><img src="https://img.shields.io/badge/Base_Model-Qwen3.8--27B-blue" alt="Base Model"></a>
  <a href="https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green" alt="License"></a>
</p>

GGUF quantized versions of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for use with [llama.cpp](https://github.com/ggml-org/llama.cpp), [Ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/), [GPT4All](https://gpt4all.io/), and other GGUF-compatible inference engines.

## Available Quantizations

| Filename | Quant Type | Size | Description |
|---|---|---|---|
| `qwen3.8-27b-q8_0.gguf` | Q8_0 | 28 GB | Virtually lossless; best quality |
| `qwen3.8-27b-q6_k.gguf` | Q6_K | 21 GB | Very high quality, near-lossless |
| `qwen3.8-27b-q5_k_m.gguf` | Q5_K_M | 19 GB | High quality; recommended balance |
| `qwen3.8-27b-q4_k_m.gguf` | Q4_K_M | 16 GB | Good quality; best for most users |
| `qwen3.8-27b-q3_k_m.gguf` | Q3_K_M | 13 GB | Medium quality; usable |
| `qwen3.8-27b-q2_k.gguf` | Q2_K | 7.9 GB | Lowest quality; for constrained VRAM |

## How to Choose

- **High VRAM (32GB+):** Use `Q8_0` or `Q6_K` for best quality
- **Medium VRAM (16-24GB):** Use `Q5_K_M` or `Q4_K_M` for the best quality/size trade-off
- **Low VRAM (8-16GB):** Use `Q3_K_M` or `Q2_K`

## Quickstart

### llama.cpp

```bash
./llama-cli -m qwen3.8-27b-q4_k_m.gguf -p "Hello, tell me about yourself" -n 512
```

### Ollama

Create a `Modelfile`:
```
FROM ./qwen3.8-27b-q4_k_m.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER stop "<|im_end|>"
```

```bash
ollama create qwen3.8-27b -f Modelfile
ollama run qwen3.8-27b
```

### LM Studio

Download any GGUF file and open it directly in [LM Studio](https://lmstudio.ai/).

## Base Model Information

- **Model:** Qwen3.8-27B
- **Parameters:** 27.78B
- **Architecture:** Hybrid (Gated DeltaNet + Gated Attention)
- **Context Length:** 262,144 tokens (extensible to 1M+)
- **License:** Apache 2.0
- **Capabilities:** Text generation, coding, reasoning, vision understanding, tool calling

## Credits

- Original model by [Qwen Team](https://huggingface.co/Qwen)
- Quantization by [douyamv](https://huggingface.co/douyamv) using [llama.cpp](https://github.com/ggml-org/llama.cpp)

## Related Models

- [douyamv/Qwen3.8-27B-FP8](https://huggingface.co/douyamv/Qwen3.8-27B-FP8) — FP8 quantized safetensors
- [douyamv/Qwen3.8-27B-abliterated](https://huggingface.co/douyamv/Qwen3.8-27B-abliterated) — Abliterated (uncensored) version
- [douyamv/Qwen3.8-27B-abliterated-GGUF](https://huggingface.co/douyamv/Qwen3.8-27B-abliterated-GGUF) — Abliterated GGUF quantizations
