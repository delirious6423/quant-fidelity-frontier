---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: PollardWeights
pipeline_tag: image-text-to-text
language:
- en
tags:
- pollard-weights
- pollard
- gguf
- llama.cpp
- ik_llama.cpp
- trellis
- imatrix
- qwen3_5
- quantized
- mixed-precision
- measured-allocation
- conversational
---

# Penjing-27B -- Pollard

> ### Pollard shrank this model: **54.64 GB (f16) -> 6.53 GB** -- **88% smaller, 8.4x down**.
> The smallest rung here; larger, higher-fidelity rungs are listed below.
>
> | format | this model's size |
> |---|---:|
> | f16 | 54.64 GB |
> | Q8_0 | ~28.96 GB |
> | Q6_K | 22.43 GB |
> | Q4_K_M | ~15.85 GB |
> | **PollardMix (this repo's IQ1_KT)** | **6.53 GB** |

Pollard builds of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) made with [Pollard Weights](https://github.com/WestWaters/pollard-weights) -- a ladder of **measured-allocation** quants (bits placed by per-layer sensitivity, not a uniform crush).

**Standard GGUF -- runs in stock llama.cpp / ik_llama.cpp, Ollama, LM Studio, except where noted.** `IQ2_KT`, `IQ1_KT` need [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp): their allocation puts ik_llama-only atoms on the tensors it protects. The rest run anywhere.

## Model details

| | |
|---|---|
| Parameter count | ~27.3B |
| Architecture | `qwen3_5` |
| Input support | text, image |
| imatrix | **yes** -- see [calibration](#imatrix-calibration) |
| Perplexity measured | **yes** -- table below |

## Which file should I choose?

Every rung is the **same weights**, sized to a different RAM budget by the measured allocation. Pick the largest one that fits your machine with room for context:

- **~24.4 GB RAM / VRAM** -> **`Q6_K`** (22.43 GB).
- **~17.5 GB RAM / VRAM** -> **`IQ4_XS`** (15.47 GB).
- **~14.9 GB RAM / VRAM** -> **`IQ3_S`** (12.94 GB).
- **~9.4 GB RAM / VRAM** -> **`IQ2_KT`** (7.36 GB). *(ik_llama.cpp)*
- **~9.2 GB RAM / VRAM** -> **`IQ2_XXS`** (7.25 GB).
- **~8.5 GB RAM / VRAM** -> **`IQ1_KT`** (6.53 GB). *(ik_llama.cpp)*

## Available files (wikitext2 test, -c 2048)

_f16 reference PPL 4.1422._

| file | PPL | size | Mean KLD | Top-1 agree | runs in | notes |
|---|---:|---:|---:|---:|---|---|
| `Penjing-27B-IQ1_KT.gguf` | 5.7864 | 6.53 GB | 0.5513 | 75.88% | ik_llama | smallest -- +40% vs f16 |
| `Penjing-27B-IQ2_XXS.gguf` | 5.4312 | 7.25 GB | 0.5142 | 76.38% | any llama.cpp | +31% vs f16 |
| `Penjing-27B-IQ2_KT.gguf` | 5.0817 | 7.36 GB | 0.4217 | 79.10% | ik_llama | recommended default -- +23% vs f16 |
| `Penjing-27B-IQ3_S.gguf` | -- | 12.94 GB | -- | -- | any llama.cpp |  |
| `Penjing-27B-IQ4_XS.gguf` | -- | 15.47 GB | -- | -- | any llama.cpp |  |
| `Penjing-27B-Q6_K.gguf` | -- | 22.43 GB | -- | -- | any llama.cpp | largest |

_Top-1 agree = share of tokens where the rung's most-likely token is the same one the f16 would have picked (llama-perplexity's `Same top p`). Higher is closer to the original model._

## Multimodal

Vision needs the projector shipped alongside: **`mmproj-Penjing-27B-bf16.gguf`** -- download it too and pass it with `--mmproj`. It is **not quantized**; it is small and the text ladder is where the size lives.

```bash
llama-server -m Penjing-27B-IQ1_KT.gguf --mmproj mmproj-Penjing-27B-bf16.gguf -ngl 99
```

## Download a specific file

```bash
pip install -U "huggingface_hub[cli]"
hf download PollardWeights/Penjing-27B-Pollard \
  --include "Penjing-27B-IQ1_KT.gguf" --local-dir ./
```

## How to run

`IQ1_KT` is built on ik_llama-only atoms, so it runs with **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)**:

```bash
llama-cli    -m Penjing-27B-IQ1_KT.gguf -ngl 99 -p "Explain why the sky is blue."
llama-server -m Penjing-27B-IQ1_KT.gguf -ngl 99
```

For stock llama.cpp, Ollama or LM Studio, use `Q6_K` instead:

```bash
llama-server -hf PollardWeights/Penjing-27B-Pollard:Q6_K
llama-cli    -m Penjing-27B-Q6_K.gguf -ngl 99 -p "Explain why the sky is blue."
```

## imatrix (calibration)

The importance matrix (`Qwen__Qwen3.8-27B.dat`, included) was computed on a mixed-domain corpus so the matrix sees every register the model serves.

## ARM / AVX

llama.cpp repacks weights into an interleaved layout at load time for faster inference on ARM and AVX machines -- no special file needed, online repacking covers these quants. The old `Q4_0_4_4/4_8/8_8` variants are not required.

## Errata

- `IQ2_KT`, `IQ1_KT` carry ik_llama-only atoms and need ik_llama.cpp to run; stock llama.cpp rejects any ggml type above 42 outright. Checked with `pollard-ggufcheck`, from the files' tensor types rather than their names.
- Measured allocation places bits by per-layer sensitivity under a size budget.
- Single machine; replication invited.

## Credits & license

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- Quantization tooling: [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml-org)
- Method + tooling: [Pollard Weights](https://github.com/WestWaters/pollard-weights) -- *measure first, no claim before a number.*
- License: `apache-2.0`, inherited from the base model.

*Built with [Pollard Weights](https://github.com/WestWaters/pollard-weights) -- frontier models, small hardware, no compromise.*
