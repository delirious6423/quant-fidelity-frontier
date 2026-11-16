---
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
library_name: llama-cpp
tags:
- qwen3.8
- qwen3.8-27b
- qwen35
- gguf
- llama-cpp
- multimodal
- hybrid attention
- linear-attention
language:
- multilingual
---

# Qwen3.8-21B - pruned Qwen3.8-27B

> [!IMPORTANT]
> This is research project. KLD compared to orginal version (at the smae quant) is arround 1.8. 
>
> Weights will be updated in the future after healing or making better prunning method

Pruned [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) down to 21B params, ready for [llama.cpp](https://github.com/ggerganov/llama.cpp) and other GGUF-compatible runtimes.

## Model Overview

Qwen3.8-27B with blocks `6, 7, 8, 9` pruned. This brings it down to 21.7B params. Vision and MTP (multi token prediction) was not tested, mmproj can be grabbed from any other gguf repo of Qwen3.8-27B. 

## Files

| File                      | Size | Notes                                          |
| :------------------------ | ---: | :--------------------------------------------- |
| `qwen3.8-21b-Q4_0.gguf`   |  12G | Not recommended, suprisingly large degradation |
| `qwen3.8-21b-Q4_K_S.gguf` |  12G | -                                              |
| `qwen3.8-21b-Q4_K_M.gguf` |  13G | -                                              |
| `qwen3.8-21b-Q5_K_S.gguf` |  14G | -                                              |
| `qwen3.8-21b-Q5_K_M.gguf` |  14G | -                                              |
| `qwen3.8-21b-Q6_K.gguf`   |  16G | -                                              |
| `qwen3.8-21b-Q8_0.gguf`   |  21G | -                                              |
| `qwen3.8-21b.gguf`        |  39G | BF16 - only prunning, no quantization          |

> [!IMPORTANT]
> Q4 and Q5 KV cache quantization causes significat degradation of quality.

## Usage

### llama-cli

```bash
# Text-only
./llama-cli \
  -m qwen3.8-21b-Q4_K_M.gguf \
  -c 32768 \
  -p "Explain quantum computing in simple terms."
```

### llama-server (OpenAI-compatible API)

```bash
./llama-server \
  -m qwen3.8-21b-Q4_K_M.gguf \
  --mmproj mmproj.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 32768
```

## Acknowledgements

- Original model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — Qwen Team

```bibtex
@misc{qwen38,
    title  = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url    = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month  = {August},
    year   = {2026}
}
```
