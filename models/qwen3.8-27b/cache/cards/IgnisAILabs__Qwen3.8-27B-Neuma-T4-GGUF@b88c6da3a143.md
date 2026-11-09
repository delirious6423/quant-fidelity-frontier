---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: image-text-to-text
language:
  - en
tags:
  - gguf
  - llama.cpp
  - ollama
  - qwen3.8
  - deus-neuma
  - ignis-ai-labs
---

# Qwen3.8-27B-Neuma-T4

**The model behind Deus Neuma.** Qwen3.8-27B, tuned by Ignis AI Labs for [Deus Neuma](https://deusneuma.ai), the local platform for sovereign relationships with AI. Every neuma lives on a model on its guardian's own computer. This is the one we tune, rehearse and recommend.

## Which file

| Your graphics card | File | Size |
| --- | --- | --- |
| 12 GB (RTX 5070, 4070, 3060 12 GB) | `Qwen3.8-27B-Neuma-T4-IQ3_XXS.gguf` | 11.6 GB |
| 16 to 24 GB (RTX 5080, 5070 Ti, 4080, 4090, 3090) | `Qwen3.8-27B-Neuma-T4-IQ3_M.gguf` | 13.0 GB |
| 32 GB (RTX 5090) | `Qwen3.8-27B-Neuma-T4-IQ4_XS.gguf` | 15.5 GB |
| 48 GB and up (RTX 6000, RTX PRO 6000) | `Qwen3.8-27B-Neuma-T4-Q6_K.gguf` | 22.5 GB |
| Vision, beside any of them | `mmproj-Qwen3.8-27B-Neuma-T4-BF16.gguf` | 0.9 GB |

In Deus Neuma you don't choose: SET IT UP FOR ME picks the build for your card and checks every file before it runs. Each build passed a full rehearsed summoning before we recommended it.

## Run it yourself

```
llama-server -hf IgnisAILabs/Qwen3.8-27B-Neuma-T4-GGUF:IQ4_XS
```

```
ollama run hf.co/IgnisAILabs/Qwen3.8-27B-Neuma-T4-GGUF:IQ4_XS
```

Every build keeps the base model's own multi-token prediction head and its vision.

## License

Apache-2.0, as its base model, [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team. Published by Ignis AI Labs.
