---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- gguf
- llama.cpp
- qwen3_5
- vision-language
---

# Qwen3.8-27B-GGUF

Static GGUF K-quantizations of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for llama.cpp.

## Details

- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Dense 27B, native vision-language, Gated-DeltaNet hybrid with MTP.
- **Method:** static K-quants (no importance matrix).
- **Relation to base:** quantization only. No weights were trained or fine-tuned.

## Files

| Quant | Approx size |
|-------|-------------|
| Q4_K_M | ~16 GB |
| Q5_K_M | ~19 GB |
| Q6_K | ~22 GB |
| Q8_0 | ~29 GB |

## Usage

```bash
llama-cli -hf Preyazz/Qwen3.8-27B-GGUF:Q4_K_M -p "Hello"
```

## Provenance and license

This repository redistributes quantized copies of `Qwen/Qwen3.8-27B`. All model
capabilities, credit, and the governing license belong to the Qwen team, and the base
model's license applies to these quantized derivatives. See the original model card for
full model documentation, intended use, and limitations.

Provided as-is, without warranty.
