---
base_model: Qwen/Qwen3.8-27B
library_name: gguf
license: apache-2.0
tags:
- onix
- gguf
- quantized
- q4_k_m
- text-generation
pipeline_tag: text-generation
---

# Qwen3.8-27B - GGUF Quantized (Q4_K_M)
*Published by ONIX Community • Authentic working GGUF weights quantized on Apple Silicon Metal.*

## Model Details
- **Base Model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Quantization Format:** `Q4_K_M`
- **Working GGUF Weight Size:** `15.84 GB`
- **Verification Status:** Verified authentic tensor conversion via `llama.cpp`

## Usage with Ollama
```bash
ollama run pranil00/ONIX-Qwen3.8-27B-GGUF
```

## Usage with llama.cpp
```bash
llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -p "Hello" -n 128
```
