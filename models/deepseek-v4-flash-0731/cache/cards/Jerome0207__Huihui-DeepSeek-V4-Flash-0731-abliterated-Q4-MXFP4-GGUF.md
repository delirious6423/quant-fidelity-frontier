---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4-Flash-0731
language:
  - en
  - zh
tags:
  - gguf
  - deepseek-v4
  - text-generation
  - abliterated
---

# Huihui DeepSeek V4 Flash 0731 Abliterated — Q4-MXFP4

This is a single-file mirror of the Q4-MXFP4 GGUF from
[`huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF`](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF).
It exists so RunPod's cached-model feature downloads only the selected quantization instead of the complete multi-quant repository.

The model and quantization were created by their respective upstream authors. This repository does not alter the GGUF payload.

## File

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `DeepSeek-V4-Flash-Q4-mxfp4-0731.gguf` | 155,976,458,848 | `897c9d82ec412e1d983dcda579c96bd256454414b5b80d2570452477abe7beed` |

The GGUF metadata declares the `deepseek4` architecture, a maximum context length of 1,048,576 tokens, and a DSML-capable chat template. The associated RunPod deployment intentionally uses 262,144 tokens to retain generous KV-cache and runtime headroom on two 96 GB Blackwell GPUs.

## Provenance

- Abliterated GGUF source: [`huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF`](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF)
- Base model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

Review the upstream model cards, limitations, license, and acceptable-use requirements before deployment.
