---
library_name: gguf
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- qwen3.8
---

# Qwen3.8-27B GGUF

10 GGUF files of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at the sizes of
[Unsloth's dynamic quants](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and of
[ISTA-DASLab's GSQ-RCO quants](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF), with 16–38 % less KL
divergence than the Unsloth file of the same size and 11–26 % less than ISTA-DASLab's. Two things make the difference:

- **Hessian-based quantization** (GPTQ-like): every block tensor is quantized again from Qwen's BF16 weights, with its
  values chosen so that the error hurts the model's output as little as possible. The types and sizes are llama.cpp's.
- **A type per tensor**: [qalloc](https://arki05.com/projects/qalloc/) picks each tensor's type for the size, from a
  store that holds every tensor at every type
  ([`Arki05/Qwen3.8-27B-GGUF-shards`](https://huggingface.co/Arki05/Qwen3.8-27B-GGUF-shards)).

**Build your own size:** the [qalloc app](https://qalloc.arki05.com) solves the mix for any size you pick and assembles
the GGUF in your browser, tensor by tensor from the store.

## Files and measurements

Per token, against the BF16 model: the mean KL divergence and its 99th percentile (lower is better), and how often
the most likely next token is the BF16 model's (higher is better), for the published file of the same size → this file.

### Against Unsloth's dynamic quants

| file | bpw | size | Unsloth file of this size | KLD | KLD, 99th percentile | same top token % |
|---|---|---|---|---|---|---|
| [`Qwen3.8-27B-2.16bpw.gguf`](Qwen3.8-27B-2.16bpw.gguf) | 2.16 | 7.27 GB | UD-IQ2_XXS | 0.305 → **0.209** (-32 %) | 2.59 → **2.06** | 79.9 → **83.5** |
| [`Qwen3.8-27B-2.49bpw.gguf`](Qwen3.8-27B-2.49bpw.gguf) | 2.49 | 8.37 GB | UD-IQ2_S | 0.152 → **0.124** (-19 %) | 1.42 → **1.31** | 85.8 → **87.6** |
| [`Qwen3.8-27B-2.82bpw.gguf`](Qwen3.8-27B-2.82bpw.gguf) | 2.82 | 9.48 GB | UD-Q2_K_XL | 0.0984 → **0.0823** (-16 %) | 0.959 → **0.888** | 88.6 → **89.7** |
| [`Qwen3.8-27B-3.15bpw.gguf`](Qwen3.8-27B-3.15bpw.gguf) | 3.15 | 10.58 GB | UD-IQ3_XXS | 0.0711 → **0.0513** (-28 %) | 0.725 → **0.531** | 90.3 → **91.8** |
| [`Qwen3.8-27B-3.81bpw.gguf`](Qwen3.8-27B-3.81bpw.gguf) | 3.81 | 12.80 GB | UD-Q3_K_XL | 0.0288 → **0.0213** (-26 %) | 0.289 → **0.230** | 93.7 → **94.7** |
| [`Qwen3.8-27B-4.46bpw.gguf`](Qwen3.8-27B-4.46bpw.gguf) | 4.46 | 15.01 GB | UD-Q4_K_S | 0.0155 → **0.00964** (-38 %) | 0.169 → **0.103** | 95.4 → **96.4** |

### Against ISTA-DASLab's GSQ-RCO quants

| file | bpw | size | ISTA-DASLab file of this size | KLD | KLD, 99th percentile | same top token % |
|---|---|---|---|---|---|---|
| [`Qwen3.8-27B-2.51bpw.gguf`](Qwen3.8-27B-2.51bpw.gguf) | 2.51 | 8.42 GB | GSQ-RCO-IQ2_XS | 0.151 → **0.122** (-19 %) | 1.54 → **1.30** | 86.0 → **87.6** |
| [`Qwen3.8-27B-2.75bpw.gguf`](Qwen3.8-27B-2.75bpw.gguf) | 2.75 | 9.26 GB | GSQ-RCO-IQ2_S | 0.101 → **0.0901** (-11 %) | 1.01 → **0.971** | 88.3 → **89.4** |
| [`Qwen3.8-27B-3.00bpw.gguf`](Qwen3.8-27B-3.00bpw.gguf) | 3.00 | 10.09 GB | GSQ-RCO-IQ3_XXS | 0.0740 → **0.0641** (-13 %) | 0.760 → **0.685** | 90.1 → **90.8** |
| [`Qwen3.8-27B-3.50bpw.gguf`](Qwen3.8-27B-3.50bpw.gguf) | 3.50 | 11.77 GB | GSQ-RCO-IQ3_S | 0.0424 → **0.0312** (-26 %) | 0.449 → **0.321** | 92.5 → **93.6** |

ISTA-DASLab's files without the multi-token-prediction layer (their non-`-mtp` files), revision d562806.

### How it was measured

Every file, ours and the published ones, and the BF16 model read the same 128 windows of 4096 tokens. The windows are held-out documents from nine
sources; none of them was used to quantize the model.

| domain | windows | data |
|---|---|---|
| web | 13 | FineWeb-Edu, DCLM |
| encyclopedic | 10 | WikiText-103 |
| multilingual | 15 | FineWeb-2 (Arabic, Chinese, German, French, Japanese, Korean) |
| math | 17 | GSM8K, MATH |
| code | 7 | The Stack (smol), MBPP |
| science | 17 | MMLU (auxiliary train), ARC |
| instruction following | 17 | Tulu 3 (persona instruction following) |
| tool use | 16 | Glaive function calling v2, ToolACE |
| long context | 16 | FineWeb-Edu, PG-19, NuminaMath |

For plain text, every token counts. For the prompt datasets, the BF16 model wrote its own answer to each prompt,
thinking included, and only the answer counts, not the prompt. In each window the first half is context and the
second half is scored: 220k tokens in all.

Sizes are compared without the multi-token-prediction layer (about 0.35 GB). Unsloth's files from UD-Q2_K_XL up
include it; these files leave it out, since normal generation doesn't use it.

More benchmarks, generation-based ones included, will follow.

`allocations/` holds each file's per-tensor choice; the shards repo rebuilds any of them, or any other size.
