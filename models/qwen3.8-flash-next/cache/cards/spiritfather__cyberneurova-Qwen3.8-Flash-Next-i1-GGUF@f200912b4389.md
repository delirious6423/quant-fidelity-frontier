---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
quantized_by: spiritfather
tags:
- gguf
- imatrix
- creative-writing
- llama.cpp
---

# cyberneurova-Qwen3.8-Flash-Next — GGUF (imatrix (i1) quants)

<p align="center">
    🧬 Source model: <a href="https://huggingface.co/cyberneurova/cyberneurova-Qwen3.8-Flash-Next" target="_blank">cyberneurova/cyberneurova-Qwen3.8-Flash-Next</a>
    <br>
    📦 Static quants: <a href="https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF" target="_blank">spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF</a>
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

> [!Important]
> **Requantized with PLE (per-layer expert) mixed precision.** The IQ2_M / IQ3_XXS / IQ3_M / IQ4_XS quants have been rebuilt with mixed quantization: trunk (attention, shared experts, embeddings) stays at Q8_0 while routed expert FFN tensors are quantized per-layer — up/gate projections at the named precision, down projections at IQ4_NL. This gives significantly better quality-per-byte than flat imatrix quants on MoE architectures. Each rung pairs with the external MTP draft model from the [static repo](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF). IQ4_XS is near-lossless (+0.28% PPL vs Q8_0).

> This is an uncensored/unrestricted version of Qwen3.8-Flash-Next. Only the refusal behavior is removed; general model capabilities are preserved. IQ1/IQ2 quants are not available for this architecture (qwen4exp hyper-connection tensors are not covered by the imatrix calibration). Use the static Q2_K from the static repo as the lowest available quant.

> imatrix (importance matrix) computed over [bartowski calibration_datav3](https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8) on this model's own bf16. Weighted quants spend their bits where the calibration text says the model is sensitive; they are usually preferable to a static quant of the same size below Q5.

## PLE mixed quants (new)

Requantized from bf16 with per-layer expert mixed precision. Trunk at Q8_0; routed expert tensors overridden per rung. All rungs use the same external [MTP draft model (Q8_0)](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-GGUF/resolve/main/mtp-cyberneurova-Qwen3.8-Flash-Next-Q8_0.gguf) from the static repo.

| Quant | Size | Mixture (default / up / gate / down) | PLE | MTP pair | PPL (wikitext-2, ctx 512) | vs Q8_0 | KLD vs Q8_0 |
|:--|--:|:--|:--|:--:|--:|--:|--:|
| [IQ2_M](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next-IQ2_M.gguf) | 100.9 GiB | Q8_0 / IQ2_S / IQ2_S / IQ4_NL | Q8_0 | ✓ | 5.1252 ± 0.032 | +5.97% | 0.150323 ± 0.000859 |
| [IQ3_XXS](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next-IQ3_XXS.gguf) | 105.6 GiB | Q8_0 / IQ3_XXS / IQ3_XXS / IQ4_NL | Q8_0 | ✓ | 4.9843 ± 0.031 | +3.05% | 0.093681 ± 0.000580 |
| [IQ3_M](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next-IQ3_M.gguf) | 109.1 GiB | Q8_0 / IQ3_S / IQ3_S / IQ4_NL | Q8_0 | ✓ | 4.9869 ± 0.031 | +3.11% | 0.072916 ± 0.000464 |
| [IQ4_XS](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next-IQ4_XS.gguf) | 116.7 GiB | Q8_0 / IQ4_XS / IQ4_XS / IQ4_NL | Q8_0 | ✓ | 4.8502 ± 0.030 | +0.28% | 0.039888 ± 0.000269 |

Q8_0 baseline PPL: 4.8366 ± 0.030. Trunk (non-expert Q8_0): 55.8 GiB constant across all rungs.

## Flat imatrix quants

| Quant | Size | Notes |
|:--|--:|:--|
| [Q2_K](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-Q2_K.gguf) | 80.4 GB |  |
| [IQ3_S](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-IQ3_S.gguf) | 88.9 GB |  |
| [Q3_K_S](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-Q3_K_S.gguf) | 88.8 GB |  |
| [Q3_K_M](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-Q3_K_M.gguf) | 94.2 GB | IQ3_S probably better |
| [Q3_K_L](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-Q3_K_L.gguf) | 96.6 GB |  |
| [Q4_K_S](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-Q4_K_S.gguf) | 111.8 GB | optimal size/speed/quality |
| [Q4_K_M](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-Q4_K_M.gguf) | 119.2 GB | fast, recommended |
| [Q5_K_S](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-Q5_K_S.gguf) | 127.8 GB |  |
| [Q5_K_M](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-Q5_K_M.gguf) | 134.1 GB |  |
| [Q6_K](https://huggingface.co/spiritfather/cyberneurova-Qwen3.8-Flash-Next-i1-GGUF/resolve/main/cyberneurova-Qwen3.8-Flash-Next.i1-Q6_K.gguf) | 167.6 GB | practically like static Q6_K |

## Loading (any current llama.cpp)

```
llama-server -m cyberneurova-Qwen3.8-Flash-Next-IQ4_XS.gguf -ngl 99 -fa on -c 32768 -ctk q8_0 -ctv q8_0 --jinja
```
