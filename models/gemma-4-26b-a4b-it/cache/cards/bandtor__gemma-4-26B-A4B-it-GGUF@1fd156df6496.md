---
base_model: google/gemma-4-26B-A4B-it
license: apache-2.0
tags:
  - gguf
  - ollama
  - gemma4
  - q4_k_m
  - moe
  - mixture-of-experts
  - llama-cpp
  - multimodal
language:
  - en
  - pt
  - multilingual
library_name: gguf
pipeline_tag: image-text-to-text
---

# Gemma 4 26B A4B-it (MoE) — GGUF Q4_K_M

Quantização **Q4_K_M** do modelo [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it),
arquitetura **Mixture-of-Experts**.

| Arquivo | Tamanho | Descrição |
|---|---|---|
| `gemma-4-26B-A4B-it-Q4_K_M.gguf` | ~16.8–16.9 GB | Modelo principal (MoE Q4_K_M) |
| `mmproj-gemma-4-26B-A4B-it-f16.gguf` | ~122 MB | Projetor visual (se disponível) |
| `Modelfile` | — | Template Ollama pronto para uso |

## Especificações do modelo

| Propriedade | Valor |
|---|---|
| Arquitetura | Mixture-of-Experts (gemma4) |
| Parâmetros totais | 25.2B (26B com embeddings) |
| Parâmetros ativos | ~3.8B por token |
| Camadas | 30 |
| Experts | 8 ativos / 128 totais + 1 shared |
| Sliding Window | 1024 tokens |
| **Contexto máximo** | **256K tokens (262 144)** |
| Vocabulário | 262K tokens |
| Modalidades | Texto + Imagem |
| Licença | Apache 2.0 |

> O "A" em **A4B** significa "Active 4B" — apenas 4B parâmetros são ativados por token
> durante inferência, tornando o modelo quase tão rápido quanto um modelo 4B,
> com a qualidade de um 26B.

## Uso com Ollama

```bash
# Opção 1 — direto do repositório HF (Ollama >= 0.3)
ollama run hf.co/bandtor/gemma-4-26B-A4B-it-GGUF

# Opção 2 — clonar e criar localmente
huggingface-cli download bandtor/gemma-4-26B-A4B-it-GGUF --local-dir ./gemma4-26b-a4b
ollama create gemma4-26b-a4b -f ./gemma4-26b-a4b/Modelfile
ollama run gemma4-26b-a4b
```

### Reduzindo contexto para menor uso de VRAM

```bash
# 64K tokens — recomendado para GPUs com 24 GB VRAM
ollama run hf.co/bandtor/gemma-4-26B-A4B-it-GGUF --option num_ctx 65536

# 32K tokens — recomendado para GPUs com 16 GB VRAM
ollama run hf.co/bandtor/gemma-4-26B-A4B-it-GGUF --option num_ctx 32768
```

## Uso com llama.cpp

```bash
# Texto (256K ctx completo — requer ~40 GB RAM/VRAM para KV cache)
llama-cli -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --ctx-size 65536 \
  -ngl 99 \
  --prompt "<|turn>user\nOlá!<turn|>\n<|turn>model\n"

# Multimodal (com mmproj)
llama-cli -m gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --mmproj mmproj-gemma-4-26B-A4B-it-f16.gguf \
  --image imagem.jpg \
  --ctx-size 32768 \
  --prompt "<|turn>user\nDescreva esta imagem.<turn|>\n<|turn>model\n"
```

## Formato de prompt (Gemma 4)

```
<bos><|turn>system
{system_prompt}<turn|>
<|turn>user
{mensagem}<turn|>
<|turn>model
{resposta}<turn|>
```

## Requisitos de memória (estimativa)

| Configuração | VRAM mínima |
|---|---|
| Apenas modelo Q4_K_M (sem KV cache) | ~18 GB |
| Modelo + 32K ctx KV cache | ~20 GB |
| Modelo + 64K ctx KV cache | ~22 GB |
| Modelo + 256K ctx KV cache completo | ~40 GB |

## Créditos

- Quantização primária: [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) (Dynamic 2.0 imatrix)
- Quantização alternativa: [lmstudio-community/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-GGUF) (llama.cpp b8778)
- Modelo base: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
