---
license: apache-2.0
library_name: llama.cpp
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
quantized_by: MagicQuant
language:
- en
tags:
- gguf
- quantized
- magicquant
---

# Qwen3.8-27B-MagicQuant-GGUF

Derivative of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized using MagicQuant hybrid evolutionary per-tensor search.

Sibling repo with AMD-native (ROCmFPX fork-only) builds: [lmcoleman/Qwen3.8-27B-ROCmFPX-GGUF](https://huggingface.co/lmcoleman/Qwen3.8-27B-ROCmFPX-GGUF).

## Base Model

This is a derivative of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
All credit for the base model architecture and weights goes to the original authors.
The base model's license applies to this derivative.

## Quantization Method

Quantized using **[MagicQuant](https://github.com/lucasmcoleman/MagicQuant)** hybrid evolutionary per-tensor quantization,
based on the methodology by **[magiccodingman](https://github.com/magiccodingman/MagicQuant-Wiki)**:

- Tensors are classified into sensitivity groups (Embeddings, Head, Query, Key, Output, FFN Up/Down, MoE Experts, Router)
- An evolutionary search finds the optimal quantization type per group, balancing size vs. perplexity
- **Q4/Q5/Q6 tier targets** are searched, and each one ships only if it earns its place (see below)
- Small-row tensors and sensitivity-critical layers (embeddings, output head, router) are kept at F32/F16/BF16
- This is NOT a uniform quantization -- each tensor group gets its own optimal type

A tier name here is a **size band**, not a promise that every tensor uses that
exact type. A "Q5" is whatever mix of schemes landed in the Q5 size band with
the lowest measured perplexity loss -- which is the point of the search.

## GGUF Files

| File | Size | Quant | Perplexity vs BF16 |
|------|------|-------|--------------------|
| [Qwen3.8-27B-Q4_K_M.gguf](./Qwen3.8-27B-Q4_K_M.gguf) | 15.7 GB | Q4 hybrid | 6.7611 (+0.25%) |
| [Qwen3.8-27B-Q6_K.gguf](./Qwen3.8-27B-Q6_K.gguf) | 22.4 GB | Q6 hybrid | 6.7579 (+0.20%) |
| [mmproj-Qwen3.8-27B-f16.gguf](./mmproj-Qwen3.8-27B-f16.gguf) | 0.9 GB | F16 (unquantized) | not measured |

### Why there is no Q5

A Q5 tier was searched, built and measured. It is **not published**, because it was
dominated by Q4 on every axis measured here:

| | Size | Perplexity vs BF16 | Generation |
|---|---|---|---|
| Q4_K_M | 14.65 GiB | 6.7611 (+0.25%) | 6.03 tok/s |
| *Q5 (withheld)* | *17.68 GiB* | *6.7666 (+0.33%)* | *4.57 tok/s* |
| Q6_K | 20.89 GiB | 6.7579 (+0.20%) | 4.47 tok/s |

It was 21% larger than Q4, measured slightly worse, and generated ~24% slower. The
quality difference is small enough to be a tie rather than a real regression, but a
tie at 21% more disk and a quarter less speed is not a tier worth shipping: there is
no request for which it is the right answer.

This is a property of how the schemes round into size bands for this particular
model, not a defect in the file. The useful ladder here is **Q4 for speed, Q6 for
quality**.

Throughput figures above are **CPU-only** (`llama-bench`, no GPU offload, on a
Ryzen AI MAX+ 395), measured during the search alongside other load. They are
useful for comparing these tiers against each other -- generation rate was
consistent across all 15 measured candidates -- but they are **not** the speed you
should expect from a GPU or Metal build, and they are not a benchmark of this
hardware. Prompt-processing figures from the same runs were inconsistent and are
omitted for that reason.


Perplexity measured on wikitext-2 (100 chunks, ctx 512) against the BF16 baseline of **6.7443**. Lower is better; the percentage is the increase over BF16. These are the same measurements the tier selection is based on, so a tier that shipped is one that earned its size.

**Recommended: Q4 (14.65 GiB).** It is the smallest tier that is statistically tied with the best measured quality here. Q6 is 43% larger for 0.048 percentage points of perplexity, which is below what this measurement can resolve -- so the extra bytes buy nothing you can detect.

## Usage

### LM Studio

1. Download the GGUF file of your preferred quantization tier
2. Place it in your LM Studio models directory
3. Load the model in LM Studio -- it will auto-detect the chat template
4. The model supports the base model's full context length

### llama.cpp

```bash
# Interactive chat (--jinja uses the model's embedded chat template, not a hardcoded one)
llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -c 8192 --jinja -cnv

# Single prompt
llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -c 8192 -p "Your prompt here"

# Server mode
llama-server -m Qwen3.8-27B-Q4_K_M.gguf -c 8192 --port 8080 --jinja
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(model_path="./Qwen3.8-27B-Q4_K_M.gguf", n_ctx=8192)
output = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)
print(output["choices"][0]["message"]["content"])
```

### Vision (image input)

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf --mmproj mmproj-Qwen3.8-27B-f16.gguf -c 8192 --port 8080 -ngl 99 -fa on
```

## Serving: MTP Speculative Decoding

This model includes **MTP ("nextn") draft tensors**, enabling self-speculative
decoding -- measured **~1.6-1.9x faster generation** with a ~95% first-token
accept rate (no separate draft model needed; it drafts from itself):

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf -c 8192 --port 8080 --host 127.0.0.1 -ngl 99 -md Qwen3.8-27B-Q4_K_M.gguf --spec-type draft-mtp -ctk q8_0 -ctv q8_0 -fa on
```

**Memory cost:** MTP needs its own draft context alongside the main context,
so serving with it uses roughly **2x the model's memory** compared to serving
without ``-md``/``--spec-type draft-mtp``.

## Caveats

- The base model's license (apache-2.0) applies to all derivative files
- Quantization reduces precision -- verify outputs for your specific use case
- The hybrid quantization assigns different precision to different tensor groups, which means quality characteristics may differ from uniform quantizations

## Limitations

- Quantized models may exhibit subtle differences from the full-precision fine-tune
- This model inherits any limitations and biases present in the base model

---
*Generated with [MagicQuant](https://github.com/lucasmcoleman/MagicQuant)*
