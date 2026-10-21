---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
- gguf
- mangaba-ai
- llama.cpp
language:
- pt
- en
---

<div align="center">

# 🍊 Mangaba-Gemma-4-26B-A4B-it-GGUF

**Coleção Mangaba AI** · Modelos GGUF otimizados para inferência local

</div>

## Sobre

Este repositório faz parte da coleção **Mangaba AI** de modelos GGUF prontos
para rodar localmente com `llama.cpp`, LM Studio, Ollama ou o **Mangaba Gateway**.

- **Modelo base:** [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Formato:** GGUF (quantizado)
- **Uso recomendado:** inferência local em CPU/Apple Silicon

## Como usar

```bash
# llama.cpp
llama-cli -hf mangaba-ai/Mangaba-Gemma-4-26B-A4B-it-GGUF -p "Olá!"
```

```bash
# download direto
hf download mangaba-ai/Mangaba-Gemma-4-26B-A4B-it-GGUF
```

---

<sub>Licença do modelo base: apache-2.0. Quantização GGUF derivada do trabalho
da equipe Unsloth (atribuição mantida conforme a licença). Branding e curadoria:
Mangaba AI.</sub>
