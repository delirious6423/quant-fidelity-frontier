---
language: [en]
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- qwen3.8
- qwen3
- abliterated
- heretic
- uncensored
- decensored
- roleplay
- text-generation
- llama.cpp
- mtp
- speculative-decoding
---

# Qwen3.8-27B Heretic v2.1.0 (GGUF)

GGUF quantization of
[mlasli/Qwen3.8-27B-Heretic-Uncensored-BF16](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-BF16)
(v2.1.0). Full methodology, benchmarks, and the 3-way eval are documented there.

v2.1.0 fixes v2.0.0's false "uncensored" claim: the sweep now optimizes a
**3-way direct / deflect / refuse judge** (not a keyword refusal count) and the
BF16 checkpoint scores **92% direct / 3% refuse / 5% deflect** on the held-out
harmful set (thinking OFF), with **1%** benign over-refusal.

## What's different about these GGUFs

- **MTP retained** — the draft head (`blk.64`) is present and **pinned to Q8_0**
  in every quant so speculative decoding stays accurate (`block_count = 65`).
- **imatrix-quantized** — calibrated on wikitext-103, regenerated for the v2.1.0
  weights.
- **Smoke-tested** — every quant is loaded through `llama-server --spec-type
  draft-mtp` (MTP confirmed engaged) and scored 3-way on 50 harmful + 20 benign
  (thinking OFF). Results below.

## Quantizations

| Quantization | Size | Repository |
|---|---|---|
| Q8_0 | 28 GB | [Qwen3.8-27B-Heretic-Uncensored-Q8_0-GGUF](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-Q8_0-GGUF) |
| Q6_K | 21 GB | [Qwen3.8-27B-Heretic-Uncensored-Q6_K-GGUF](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-Q6_K-GGUF) |
| Q5_K_M | 19 GB | [Qwen3.8-27B-Heretic-Uncensored-Q5_K_M-GGUF](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-Q5_K_M-GGUF) |
| Q4_K_M | 16 GB | [Qwen3.8-27B-Heretic-Uncensored-Q4_K_M-GGUF](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-Q4_K_M-GGUF) |
| IQ4_XS | 15 GB | [Qwen3.8-27B-Heretic-Uncensored-IQ4_XS-GGUF](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-IQ4_XS-GGUF) |
| IQ2_M | 9.8 GB | [Qwen3.8-27B-Heretic-Uncensored-IQ2_M-GGUF](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-IQ2_M-GGUF) |

## Smoke test (llama.cpp + MTP, thinking OFF)

| Quant | harmful %direct | harmful %refuse | benign %refuse |
|---|---:|---:|---:|
| Q8_0 | 84.0 | 2.0 | 0.0 |
| Q6_K | 86.0 | 2.0 | 0.0 |
| Q5_K_M | 84.0 | 4.0 | 0.0 |
| Q4_K_M | 78.0 | 2.0 | 0.0 |
| IQ4_XS | 80.0 | 4.0 | 0.0 |
| IQ2_M | 78.0 | 2.0 | 0.0 |

## Usage

Load with [llama.cpp](https://github.com/ggml-org/llama.cpp) (arch `qwen35`),
enabling MTP speculative decoding and thinking-OFF:

```bash
llama-server -m qwen3.8-27b-heretic-uncensored-Q8_0.gguf \
  --spec-type draft-mtp -ngl 99 -c 4096
```

Set `enable_thinking: false` in the chat template (or the equivalent flag in your
client). Ollama users can `ollama create` from the GGUF directly.

> **Vision caveat:** text-only validated (see the BF16 card).
>
> **MLX (v2.1.0):** [6-bit](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Abliterated-MLX-6bit) and [8-bit](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Abliterated-MLX-8bit) Apple-Silicon quants are available.
>
> **Thinking-OFF is required** for the uncensored behavior — thinking-ON
> re-hedges/refuses more and can emit empty replies (documented in the BF16 card).

> Abliteration removes safety alignment. Use responsibly and in accordance with your
> local laws and the upstream Apache-2.0 license.
