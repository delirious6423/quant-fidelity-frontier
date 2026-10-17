---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen4exp
- mixture-of-experts
- unsloth
- iq3_xxs
- q3_k_xl
- q8_0
language:
- pt
- en
---

# Qwen3.8-Flash-Next-NG8-GGUF

Estes são os quants GGUF da Unsloth ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)) do modelo Qwen3.8-Flash-Next (Qwen), com uma única modificação: o tensor de embeddings N-gram/PLE foi substituído pela versão em **Q8_0 oficial**. Os tensores do backbone permanecem exatamente como publicados pela Unsloth, sem reprocessamento.

Arquitetura MoE com backbone de 125B parâmetros e tabela N-gram/PLE de 51B, totalizando ~176B parâmetros com ~6B ativos por token. Multimodal (texto e visão), janela de contexto de até 262144 tokens.

## Motivação técnica

Os quants da Unsloth aplicam esquemas de baixa precisão (`IQ3_XXS`, `Q3_K_XL`) a todos os tensores, incluindo o tensor `per_layer_token_embd` (a tabela N-gram/PLE). Esse tensor difere dos demais em padrão de acesso: é indexado de forma esparsa e quase aleatória (lookup), não participa de matmuls densas, e por isso o erro de quantização se acumula de forma desproporcional em relação aos tensores compute-bound do backbone. Compressão agressiva nesse tensor específico tem impacto direto e perceptível na qualidade de saída.

### O que o build NG8 faz

- Mantém os tensores do backbone exatamente como publicados pela Unsloth (`IQ3_XXS` ou `Q3_K_XL`), sem reprocessamento.
- Extrai o tensor `per_layer_token_embd` do build `Q8_0` oficial (referência de quantização quase sem perda).
- Substitui esse único tensor no arquivo GGUF, mantendo todos os demais metadados e tensores intactos.

Custo marginal de ~25 GB no arquivo final, em troca de qualidade de embedding próxima da Q8_0 de referência.

- **Pesos-fonte:** `unsloth/Qwen3.8-Flash-Next-GGUF`
- **Modelo base:** `Qwen/Qwen3.8-Flash-Next`
- **Licença:** Apache-2.0

## Validação

- Confirmação de mesmo checkpoint de origem entre os builds Unsloth e Q8_0 oficial.
- Verificação tensor a tensor (1224 tensores no total).
- Comparação de bytes em offsets de início, meio e fim de cada tensor.
- Teste funcional de geração (inference smoke test) pós-substituição.

## Variantes disponíveis

| Variante | Tensores do backbone | Tabela N-gram/PLE | Tamanho total |
|---|---|---|---|
| UD-IQ3_XXS-NG8 | IQ3_XXS (Unsloth Dynamic) | Q8_0 | 107.56 GB |
| UD-Q3_K_XL-NG8 | Q3_K_XL (Unsloth Dynamic) | Q8_0 | 115.59 GB |
| Q8_0 oficial (referência, não distribuído aqui) | Q8_0 | Q8_0 | ~180 GB |

Cada variante está fragmentada em 4 partes (`00001-of-00004` a `00004-of-00004`); é necessário baixar as 4 para reconstituir o modelo.

| Arquivo | Bytes |
|---|---|
| `Qwen3.8-Flash-Next-UD-IQ3_XXS-NG8-00001-of-00004.gguf` | 10946624 |
| `Qwen3.8-Flash-Next-UD-IQ3_XXS-NG8-00002-of-00004.gguf` | 20767783040 |
| `Qwen3.8-Flash-Next-UD-IQ3_XXS-NG8-00003-of-00004.gguf` | 32382955968 |
| `Qwen3.8-Flash-Next-UD-IQ3_XXS-NG8-00004-of-00004.gguf` | 54400261312 |
| `Qwen3.8-Flash-Next-UD-Q3_K_XL-NG8-00001-of-00004.gguf` | 10946624 |
| `Qwen3.8-Flash-Next-UD-Q3_K_XL-NG8-00002-of-00004.gguf` | 21183115520 |
| `Qwen3.8-Flash-Next-UD-Q3_K_XL-NG8-00003-of-00004.gguf` | 39992153376 |
| `Qwen3.8-Flash-Next-UD-Q3_K_XL-NG8-00004-of-00004.gguf` | 54400261312 |

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "mduca771/Qwen3.8-Flash-Next-NG8-GGUF",
    allow_patterns=["*IQ3_XXS-NG8*"],
    local_dir="./model",
)
```

## Requisitos de execução

- `llama.cpp` build > `b10660` (testado em `b11209`).
- GPU com ~24 GB VRAM (ex.: RTX 4090) para os tensores do backbone via `-ngl 99`.
- 96 a 128 GB de RAM. A tabela N-gram/PLE reside em RAM/SSD NVMe e é acessada via memory-mapping conforme demanda; não deve ser offloadada para a GPU.
- Flag obrigatória: `-ot per_layer_token_embd=CPU`. Sem ela, o runtime tenta alocar os ~54 GB da tabela na VRAM e falha por OOM.

```powershell
llama-cli.exe -m Qwen3.8-Flash-Next-UD-IQ3_XXS-NG8-00001-of-00004.gguf `
  -p "Diga apenas: OK" -n 8 -c 4096 -ngl 99 `
  -ot per_layer_token_embd=CPU --jinja
```

## Atribuição

Os tensores do backbone quantizados (`IQ3_XXS`, `Q3_K_XL`) e a referência `Q8_0` são trabalho da Unsloth. O modelo original é da Qwen. Este build combina os dois builds publicados por meio de substituição de tensor único, com validação de integridade tensor a tensor, sem retreinamento ou modificação de pesos além da substituição descrita.
