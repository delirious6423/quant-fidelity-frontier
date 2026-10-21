---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
tags:
- gguf
- moe
- text-generation
---
# gemma-4-26B-A4B-it — Q4 / Q3 / Q2 GGUF

Fast GGUFs for mainstream llama.cpp, quantized antirez style: routed experts
quantized hard, everything else protected.

## Recipe (antirez style)

| Tensor group | Q4 | Q3 | Q2 |
| --- | --- | --- | --- |
| Routed expert gate / up (fused) | Q4\_K (calibrated) | IQ3\_XXS (imatrix) | IQ2\_XXS (imatrix) |
| Routed expert down | MXFP4 | IQ4\_NL (imatrix) | Q4\_0 |
| Shared experts, attention, output | Q8\_0 | Q8\_0 | Q8\_0 |
| Token embeddings | F16 | F16 | F16 |
| Routers, norms, biases | F32 | F32 | F32 |

Same strategy as Salvatore Sanfilippo's
[Qwen3.8 Flash Next](https://huggingface.co/antirez/qwen3.8-flash-next-gguf)
(Q4\_K + MXFP4) and
[DeepSeek V4](https://huggingface.co/antirez/deepseek-v4-gguf) (IQ2\_XXS +
Q2\_K) quants. Notes: gemma's down-expert rows are 704 wide and K-quants need
multiples of 256, so `Q4_K`/`Q3_K`/`Q2_K` are impossible there — the down path
uses the best fitting alternative in each class (`MXFP4`, `IQ4_NL`, `Q4_0`).
Imatrix: [unsloth](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/main/imatrix_unsloth.gguf_file).

## Files

| File | Size |
| --- | ---: |
| gemma-4-26B-A4B-it-Q4-Q4_K-MXFP4.gguf | 14.80 GiB |
| gemma-4-26B-A4B-it-Q3-IQ3_XXS-IQ4_NL.gguf | 12.48 GiB |
| gemma-4-26B-A4B-it-Q2-IQ2_XXS-Q4_0.gguf | 10.70 GiB |

## Usage

```bash
# interactive chat (multimodal: add --mmproj mmproj-F16.gguf for images)
llama-cli -m gemma-4-26B-A4B-it-Q4-Q4_K-MXFP4.gguf -t 8 -c 8192

# server
llama-server -m gemma-4-26B-A4B-it-Q4-Q4_K-MXFP4.gguf -t 8 -c 32768 --host 0.0.0.0 --port 8080
```

## Model

Gemma 4 26B-A4B-it is a multimodal MoE (26 B total, 4 B active) by Google
DeepMind. Context: 256k tokens.

Original model: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).
Original BF16 GGUF: [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF).
License: Apache-2.0 (see Gemma terms).
