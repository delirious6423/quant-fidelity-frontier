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

# Qwen3.8-27B — Pollard

> ### Pollard shrank this model: **55.56 GB (f16) → 12.08 GB** — **78% smaller, 4.6× down**.
> The smallest rung here; larger, higher-fidelity rungs are listed below.
>
> | format | this model's size |
> |---|---:|
> | f16 | 55.56 GB |
> | Q8_0 | ~29.45 GB |
> | Q6_K | ~22.78 GB |
> | Q4_K_M | ~16.11 GB |
> | **PollardMix (this repo's IQ3_S)** | **12.08 GB** |

Pollard builds of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) made with [Pollard Weights](https://github.com/WestWaters/pollard-weights) — a ladder of **measured-allocation** quants (bits placed by per-layer sensitivity, not a uniform crush).

**Standard GGUF — runs in stock llama.cpp / ik_llama.cpp, Ollama, LM Studio.** Trellis (`IQ*_KT`) files need [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp); the K-quants run anywhere.

## Model details

| | |
|---|---|
| Parameter count | ~27.8B |
| Architecture | `qwen3_5` |
| Input support | text |
| imatrix | no |
| Perplexity measured | **yes** — table below |

## Which file should I choose?

Every rung is the **same weights**, sized to a different RAM budget by the measured allocation. Pick the largest one that fits your machine with room for context:

- **~24 GB RAM / VRAM** → **`Q6_K`** (22.43 GB).
- **~18 GB RAM / VRAM** → **`IQ4_XS`** (15.72 GB).
- **~14 GB RAM / VRAM** → **`IQ3_S`** (12.08 GB).

## Available files

| file | PPL | size | tok/s | Mean KLD | notes |
|---|---:|---:|---:|---:|---|
| `Qwen3.8-27B-Pollard-IQ3_S.gguf` | (see repo) | 12.08 GB | (16 GB tier) | — | IQ3_S |
| `Qwen3.8-27B-Pollard-IQ4_XS.gguf` | — | 15.72 GB | 24 GB tier | — | IQ4_XS |
| `Qwen3.8-27B-Pollard-Q6_K.gguf` | — | 22.43 GB | 32 GB tier | — | Q6_K |

_tok/s is hardware-specific; the machine it was measured on is stated in the errata._

## Multimodal

Vision needs the projector shipped alongside: **`mmproj-Qwen3.8-27B-bf16.gguf`** — download it too and pass it with `--mmproj`. It is **not quantized**; it is small and the text ladder is where the size lives.

```bash
llama-server -m Qwen3.8-27B-Pollard-IQ3_S.gguf --mmproj mmproj-Qwen3.8-27B-bf16.gguf -ngl 99
```

## Download a specific file

```bash
pip install -U "huggingface_hub[cli]"
hf download PollardWeights/Qwen3.8-27B-Pollard \
  --include "Qwen3.8-27B-Pollard-IQ3_S.gguf" --local-dir ./
```

## How to run

These are standard GGUF and run with **llama.cpp**:

```bash
llama-server -hf PollardWeights/Qwen3.8-27B-Pollard:IQ3_S
```

or from a local file:

```bash
llama-cli    -m Qwen3.8-27B-Pollard-IQ3_S.gguf -ngl 99 -p "Explain why the sky is blue."
llama-server -m Qwen3.8-27B-Pollard-IQ3_S.gguf -ngl 99      # OpenAI-compatible API + web UI at :8080
```

They also work in anything built on llama.cpp — **LM Studio, koboldcpp, Jan, ramalama, Ollama** (`ollama run hf.co/PollardWeights/Qwen3.8-27B-Pollard`).

## ARM / AVX

llama.cpp repacks weights into an interleaved layout at load time for faster inference on ARM and AVX machines — no special file needed, online repacking covers these quants. The old `Q4_0_4_4/4_8/8_8` variants are not required.

## Errata

- Trellis (`IQ*_KT`) quants need ik_llama.cpp to build/run; K-quants run in any recent llama.cpp.
- Measured allocation places bits by per-layer sensitivity under a size budget.
- Single machine; replication invited.

## Credits & license

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- Quantization tooling: [llama.cpp](https://github.com/ggml-org/llama.cpp) (ggml-org)
- Method + tooling: [Pollard Weights](https://github.com/WestWaters/pollard-weights) — *measure first, no claim before a number.*
- License: `apache-2.0`, inherited from the base model.

*Built with [Pollard Weights](https://github.com/WestWaters/pollard-weights) — frontier models, small hardware, no compromise.*
