---
license: gemma
base_model:
- llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
language:
- en
- pt
tags:
- gemma-4
- gguf
- llama.cpp
- mixture-of-experts
- quantized
- unofficial
- prism-pro-dq-map
---

# Gemma 4 26B A4B Ultra Uncensored Heretic — PRISM-PRO-DQ Map Reapplication

Ultra Uncensored Heretic finetune converted to GGUF and quantized with an unofficial reapplication of the PRISM-PRO-DQ per-tensor type map.

## Important provenance notice

> **This is not an official PRISM-DQ or PRISM-PRO-DQ model.** This repository is not affiliated with Ex0bit, PRISM-DQ, or the authors of the source finetune.

The underlying trained weights come from [llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic](https://huggingface.co/llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic). No trained tensor values from Ex0bit's PRISM model were copied, merged, or transplanted into this model.

The quantized file was produced by reconstructing the **per-tensor GGML quantization-type assignment** visible in the public [`Ex0bit/Gemma4-26B-A4B-PRISM-PRO-DQ-GGUF`](https://huggingface.co/Ex0bit/Gemma4-26B-A4B-PRISM-PRO-DQ-GGUF) release and then applying that assignment to this separate finetune. This is therefore a **map reapplication**, not a reproduction of the original PRISM-DQ calibration, optimization procedure, dataset, or proprietary pipeline.

The Unsloth importance matrix [`imatrix_unsloth.gguf_file`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/main/imatrix_unsloth.gguf_file) was used during quantization. Conversion and quantization were performed with [`llama.cpp`](https://github.com/ggml-org/llama.cpp).

## Quantization layout

For the 658 tensors shared with the reference Gemma 4 PRISM-PRO-DQ GGUF, the final distribution is:

| GGML type | Tensor count |
|---|---:|
| F32 | 392 |
| Q5_1 | 30 |
| Q5_K | 91 |
| Q6_K | 115 |
| Q8_0 | 30 |

The language-model GGUF contains 658 tensors and matches the reconstructed reference type map without extra language-model tensors.

## Files and integrity

| File | Size | SHA-256 | Purpose |
|---|---:|---|---|
| `Gemma-4-26B-A4B-Ultra-Uncensored-Heretic-PRISM-DQ-I.gguf` | 16.85 GiB | `862ad4244531426df85e25ecb50643f6d969f1dc8dde29ab87e20f0828ad999c` | Unofficial PRISM-PRO-DQ tensor-map reapplication with Unsloth I-matrix |

## Template and tokenizer

- The tokenizer vocabulary and special-token IDs were checked against the source Hugging Face tokenizer.
- The embedded chat template was updated to the current official template from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) at conversion time.

## Usage with llama.cpp

```bash
llama-cli -m Gemma-4-26B-A4B-Ultra-Uncensored-Heretic-PRISM-DQ-I.gguf --jinja -p "Hello"
```

For server use:

```bash
llama-server -m Gemma-4-26B-A4B-Ultra-Uncensored-Heretic-PRISM-DQ-I.gguf --jinja --host 127.0.0.1 --port 8080
```

## References

1. Source finetune: [llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic](https://huggingface.co/llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic)
2. Official Gemma 4 base: [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
3. PRISM-PRO-DQ public GGUF used only as the tensor-map reference: [`Ex0bit/Gemma4-26B-A4B-PRISM-PRO-DQ-GGUF`](https://huggingface.co/Ex0bit/Gemma4-26B-A4B-PRISM-PRO-DQ-GGUF)
4. Unsloth importance matrix: [`imatrix_unsloth.gguf_file`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/main/imatrix_unsloth.gguf_file)
5. GGUF conversion and quantization tooling: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
6. Gemma Terms of Use: https://ai.google.dev/gemma/terms

## License and responsibility

Use of this repository remains subject to the original source model's license and the Gemma Terms of Use. The behavior and safety characteristics of the source finetune are inherited; quantization does not make the model safer or change its intended-use restrictions. Users are responsible for evaluating outputs and complying with applicable laws and licenses.
