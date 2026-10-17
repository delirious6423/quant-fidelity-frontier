---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: gemma
library_name: llama.cpp
pipeline_tag: text-generation
quantized_by: spiritfather
tags:
- gguf
- creative-writing
- llama.cpp
---

# cyberneurova-Qwen3.8-Flash-Next — GGUF (static quants)

<p align="center">
    🧬 Source model: <a href="https://huggingface.co/cyberneurova/cyberneurova-Qwen3.8-Flash-Next" target="_blank">cyberneurova/cyberneurova-Qwen3.8-Flash-Next</a>
    <br>
    📦 imatrix (i1) quants: <a href="https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF" target="_blank">spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF</a>
    <br>
    🛠️ Runs on <a href="https://github.com/ggml-org/llama.cpp" target="_blank">llama.cpp</a> (master)
    <br>
    📊 Scored on <a href="https://caliperbench.com" target="_blank">CaliperBench</a>
    <br>
    💬 <a href="https://discord.gg/rhE9zdKyE4" target="_blank">Discord</a>
</p>

> [!Note]
> All credit for the model goes to **[cyberneurova](https://huggingface.co/cyberneurova)**, who produced [cyberneurova-Qwen3.8-Flash-Next](https://huggingface.co/cyberneurova/cyberneurova-Qwen3.8-Flash-Next). These are quantizations only.

> [!Tip]
> Benchmarked on **[CaliperBench](https://caliperbench.com)** — a creative-writing benchmark scoring prose craft, roleplay and willingness rather than general intelligence. See this model's scores: [caliperbench.com](https://caliperbench.com)

## Provided quants

| Quant | Size | Notes |
|:--|--:|:--|
| [Q2_K](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q2_K.gguf) | 80.4 GB | |
| [Q3_K_S](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q3_K_S.gguf) | 88.8 GB | |
| [Q3_K_M](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q3_K_M.gguf) | 94.2 GB | lower quality |
| [Q3_K_L](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q3_K_L.gguf) | 96.6 GB | |
| [Q4_K_S](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q4_K_S.gguf) | 111.8 GB | fast, recommended |
| [Q4_K_M](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q4_K_M.gguf) | 119.2 GB | fast, recommended |
| [Q5_K_S](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q5_K_S.gguf) | 127.8 GB | |
| [Q5_K_M](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q5_K_M.gguf) | 134.1 GB | |
| [Q6_K](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q6_K.gguf) | 167.6 GB | very good quality |
| [Q8_0](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.Q8_0.gguf) | 188.2 GB | fast, best quality |

An MTP (multi-token prediction) draft head is also available for speculative decoding: `mtp-cyberneurova-Qwen3.8-Flash-Next-Q8_0.gguf` (3.9 GB).

## Loading (any current llama.cpp)

```
llama-server -m cyberneurova-Qwen3.8-Flash-Next.Q6_K.gguf -ngl 99 -fa on -c 32768 -ctk q8_0 -ctv q8_0 --jinja
```

