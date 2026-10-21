---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
tags:
- gguf
- llama.cpp
- metal
- apple-silicon
- speculative-decoding
- mtp
---

# Qwen3.8-27B-OptiMix (GGUF)

A 4.03 bpw quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), tuned for
fast single-stream inference on Apple Silicon (built and measured on a MacBook M4 base, 32 GB).
It also ships a lighter MTP draft head for speculative decoding.

| file | size | what it is |
|---|---|---|
| `Qwen3.8-27B-OptiMix.gguf` | 12.8 GiB | the model |
| `mtp-Qwen3.8-27B-head_q2_K.gguf` | 1.3 GiB | Qwen3.8 MTP draft with its output head re-quantized to Q2_K (draft only, never affects the output) |

## Recipe

Quantized from the full bf16 weights with the bartowski imatrix:

- IQ4_XS: attention / linear-attention projections and `ffn_down`
- IQ3_S: `ffn_gate`, `ffn_up`
- Q6_K: output head
- Q4_K: token embeddings

The point of the mix is to use only types with fast Metal kernels. `q3_K` is deliberately avoided:
on an M4 its mat-vec reads ~50 GB/s against ~90 GB/s for `q4_K`, so a smaller q3_K model is slower.

```bash
llama-quantize --imatrix Qwen3.8-27B-imatrix.gguf \
  --tensor-type attn_qkv=iq4_xs --tensor-type attn_gate=iq4_xs \
  --tensor-type ssm_out=iq4_xs  --tensor-type attn_q=iq4_xs \
  --tensor-type attn_k=iq4_xs   --tensor-type attn_v=iq4_xs \
  --tensor-type attn_output=iq4_xs --tensor-type ffn_down=iq4_xs \
  --tensor-type ffn_gate=iq3_s  --tensor-type ffn_up=iq3_s \
  --output-tensor-type q6_K --token-embedding-type q4_K \
  Qwen3.8-27B-bf16.gguf Qwen3.8-27B-OptiMix.gguf IQ3_S
```

## Quality

wikitext-2, ctx 2048, 32 chunks, paired per chunk on the same build:

| quant | size | perplexity |
|---|---|---|
| Q4_K_M (ggml-org) | 17.7 GiB | 6.207 |
| UD-IQ4_XS (unsloth) | 13.3 GiB | 6.247 |
| **OptiMix** | **12.8 GiB** | **6.302** |
| Q3_K_M (requantized from Q4_K_M) | 12.4 GiB | 6.337 |

Multiple-choice (llama-perplexity, same tasks for both): HellaSwag (100) 79%, ARC-Challenge
(first 142) 49.3%, MMLU validation (150) 42.7%.

## Speed (MacBook M4 base, 10-core GPU, 32 GB, on AC power)

| mode | tok/s |
|---|---|
| plain generation | 6.7 (reads at ~87 GB/s, the measured memory wall of this machine) |
| + MTP speculative decoding, prose | 8.2-8.8 |
| + MTP speculative decoding, code | 10.0 |
| 12 parallel slots, aggregate | 22 |

Speculative decoding pays off on Apple Silicon only with fast small-batch kernels: the 2-5 token
verify pass otherwise costs nearly as much as 2-5 separate passes. The numbers above use the
fork below; stock llama.cpp gives ~5.5 tok/s with the same command.

```bash
llama-server -m Qwen3.8-27B-OptiMix.gguf \
  -md mtp-Qwen3.8-27B-head_q2_K.gguf --spec-type draft-mtp \
  -c 8192 -np 1 --spec-draft-n-max 4 --spec-draft-p-min 0.8
```

`-np 1` is needed with a draft model, otherwise Metal can run out of memory a few thousand tokens in.

Kernels and measurement write-up: https://github.com/Vadim170/llama.cpp-m4-qwen35 (see `FORK.md`).

The files also load in stock llama.cpp; they are ordinary GGUF types.
