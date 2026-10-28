---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
thumbnail: https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-GGUF/resolve/main/hero.png
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
quantized_by: AtomicChat
pipeline_tag: text-generation
library_name: gguf
tags:
- atomic-chat
- gemma
- gemma4
- google
- gguf
- llama.cpp
- imatrix
- quantized
---

<center>

<div style="display:flex; justify-content:center; align-items:center; gap:2%; max-width:720px; margin:0 auto;">
<a href="https://atomic.chat" style="flex:0 1 auto; min-width:0;"><img src="https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-GGUF/resolve/main/pill_atomic_v3.png" alt="Atomic Chat" style="width:100%; height:auto; max-width:186px;"></a>
<a href="https://discord.gg/8wGSsvmg4V" style="flex:0 1 auto; min-width:0;"><img src="https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-GGUF/resolve/main/pill_discord_v3.png" alt="Join Discord" style="width:100%; height:auto; max-width:184px;"></a>
<a href="https://github.com/AtomicBot-ai/Atomic-Chat" style="flex:0 1 auto; min-width:0;"><img src="https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-GGUF/resolve/main/pill_github_v3.png" alt="GitHub" style="width:100%; height:auto; max-width:141px;"></a>
<a href="https://x.com/atomic_chat_hq" style="flex:0 1 auto; min-width:0;"><img src="https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF/resolve/main/pill_x_v3.png" alt="Follow on X" style="width:100%; height:auto; max-width:141px;"></a>
</div>

<br/>

<img src="https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-GGUF/resolve/main/hero.png" alt="Gemma 4 26B A4B" style="width:100%; max-width:100%; height:auto; margin-bottom:0.6em;"/>

<div style="display:flex; justify-content:center; gap:0.5em;">
<a href="https://huggingface.co/google/gemma-4-26B-A4B-it"><strong>Base model: google/gemma-4-26B-A4B-it</strong></a>
</div>
</center>

**Gemma 4 26B A4B**, self-quantized to GGUF by [Atomic Chat](https://atomic.chat). Built straight from Google's original weights with a per-tensor importance matrix, so this is not a repack of somebody else's files. Runs fully offline.

## Highlights

- **25.2B total / 3.8B active per token parameters**: the weights this repo quantizes.
- **Context length**: 256K tokens, as published by Google.
- **30 layers**: Mixture-of-Experts, hybrid sliding-window (1024) and global attention.
- **Modalities**: the base model handles Text, Image; this repo ships text-only quants, it carries no vision projector.
- **Full imatrix ladder**: every quant is calibrated with an importance matrix, published here alongside the quants.
- **Reasoning**: All models in the family are designed as highly capable reasoners, with configurable thinking modes.
- **Diverse & Efficient Architectures**: Offers Dense and Mixture-of-Experts (MoE) variants of different sizes for scalable deployment.

> [!NOTE]
> These GGUFs are **self-quantized from the original weights**, not a repack. The importance matrix keeps low-bit quants closer to the full-precision model.

> [!IMPORTANT]
> Always pass `--jinja` so the **Gemma 4 26B A4B chat template** is applied. Without it the model can emit malformed turns.

## Model Overview

| Property | Value |
|---|---|
| Base model | `google/gemma-4-26B-A4B-it` |
| Parameters | 25.2B total / 3.8B active per token |
| Layers | 30 |
| Experts | 128 routed (top-8) |
| Sliding window | 1024 tokens |
| Context length | 256K tokens |
| Vocabulary | 262K |
| Modalities | Text, Image in the base model; text only in this repo, it ships no vision projector |
| Architecture | Mixture-of-Experts, 128 experts (top-8), hybrid sliding-window (1024) and global attention, 16 attention heads over 8 KV heads, `Gemma4ForConditionalGeneration` |
| This repo | GGUF quants (imatrix); the importance matrix is published here as `imatrix-coding.gguf`. Quants: `Q2_K`, `IQ3_M`, `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_S`, `Q4_K_M`, `UD-Q4_K_XL`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0` |

<img src="https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-GGUF/resolve/main/benchmark.png" alt="Gemma 4 26B A4B benchmark scores" style="width:100%; max-width:900px;"/>

Scores are Google's published results for the base `google/gemma-4-26B-A4B-it`, not our own measurements. Quantization preserves the large majority of this; `Q4_K_M` and up stay close to full precision.

## Choosing a quant

| Quant | Size | Notes |
|---|---|---|
| `Q2_K` | 10.6 GB | Smallest K-quant. Minimal RAM, clear quality drop. |
| `IQ3_M` | 12.4 GB | Beats Q3 at a similar size thanks to imatrix. Best low-RAM pick. |
| `Q3_K_M` | 13.3 GB | Low quality but usable. |
| `Q3_K_L` | 13.8 GB | A step above Q3_K_M. |
| `IQ4_XS` | 13.9 GB | Excellent quality for size. Recommended low-bit. |
| `Q4_K_S` | 15.5 GB | Compact 4-bit, fast. |
| **`Q4_K_M`** | 16.8 GB | **Recommended default. Best balance of size, speed and quality.** |
| `UD-Q4_K_XL` | 17.0 GB | Dynamic. Embeddings and output kept at Q8_0 for higher quality at a Q4 footprint. |
| `Q5_K_S` | 18.0 GB | Higher quality, slightly more compact than Q5_K_M. |
| `Q5_K_M` | 19.1 GB | Higher quality, low loss. |
| `Q6_K` | 22.6 GB | Near lossless, noticeably lighter than Q8_0. |
| `Q8_0` | 26.9 GB | Effectively lossless, reference quality. |

> [!TIP]
> Pick the largest file that fits your (V)RAM with room for context. `Q4_K_M` or `UD-Q4_K_XL` is the sweet spot for most setups; `Q6_K` or `Q8_0` for maximum fidelity.

## Get started

Run Gemma 4 26B A4B locally with:

- **[Atomic Chat](https://atomic.chat):** the easiest path. Open the app, search `AtomicChat/gemma-4-26B-A4B-it-GGUF`, pick a quant, hit **Use this model**.
- **llama.cpp:** `llama-server -hf AtomicChat/gemma-4-26B-A4B-it-GGUF:Q4_K_M --jinja -c 8192`
- **Ollama:** `ollama run hf.co/AtomicChat/gemma-4-26B-A4B-it-GGUF:Q4_K_M`
- **LM Studio / Jan:** search the repo id, download any quant.

## Best practices

| Parameter | Value |
|---|---|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 64 |

Google's recommended sampling configuration for `google/gemma-4-26B-A4B-it`.

## Run in llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j --target llama-cli llama-server
```

```bash
./llama.cpp/build/bin/llama-server \
    -hf AtomicChat/gemma-4-26B-A4B-it-GGUF:Q4_K_M \
    --jinja -ngl 99 -c 8192 -fa on
```

## How these were made

1. Download `google/gemma-4-26B-A4B-it` (original weights).
2. Convert to f16 GGUF with [llama.cpp](https://github.com/ggml-org/llama.cpp).
3. Build an importance matrix over our calibration corpus, published here as `imatrix-coding.gguf`.
4. Quantize the ladder with `--imatrix`.
5. `UD-Q4_K_XL` additionally pins the token-embedding and output tensors to `Q8_0`.

## License

Original model by Google, released under the Apache 2.0 license. Full terms: [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license). Quantized by Atomic Chat.
