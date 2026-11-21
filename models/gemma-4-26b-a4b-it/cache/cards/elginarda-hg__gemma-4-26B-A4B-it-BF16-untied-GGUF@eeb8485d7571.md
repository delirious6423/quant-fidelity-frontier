---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
- google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
tags:
- gguf
- gemma4
---

# Gemma 4 26B-A4B-it: BF16 GGUF with an untied output tensor

Lossless BF16 GGUF conversions of two Google checkpoints, intended as quantization sources:

| file | source checkpoint |
|---|---|
| `gemma-4-26B-A4B-it-BF16-untied.gguf` | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| `gemma-4-26B-A4B-it-qat-q4_0-unquantized-BF16-untied.gguf` | [google/gemma-4-26B-A4B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-unquantized) |

- Converted with llama.cpp `convert_hf_to_gguf.py` at commit `b94959a` (`--outtype bf16`, transformers 5.x).
- "Untied": the model ties the LM head to `token_embd.weight`; these files add an explicit
  `output.weight` (an exact BF16 copy of `token_embd.weight`) so `llama-quantize` can choose the
  embedding and output types independently (`--token-embedding-type`, `--output-tensor-type`).
  Outputs are identical to the tied model.
- Text tower only; vision weights are not included.

License: Apache 2.0, as the source models.
