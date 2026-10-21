---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language:
  - en
tags:
  - gguf
  - quantized
  - unsloth
  - qwen3.8
---

# Qwen3.8-27B UD-IQ4_XS GGUF

This repository publishes the local file `Qwen3.8-27B-UD-IQ4_XS.gguf` (14,252,845,984 bytes). groxaxo did not quantize it.

The GGUF header records:

- `general.architecture`: `qwen35`
- `general.name`: `Qwen3.8-27B`
- `general.license`: `apache-2.0`
- `general.quantized_by`: `Unsloth`
- `general.file_type`: `30`
- `general.base_model.0.repo_url`: `https://huggingface.co/Qwen/Qwen3.8-27B`
- `quantize.imatrix.file`: `Qwen3.8-27B-GGUF/imatrix_unsloth.gguf`
- `quantize.imatrix.entries_count`: `496`
- `quantize.imatrix.chunks_count`: `1251`

The same filename is already published at [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). `SHA256SUMS` is the SHA-256 of the file uploaded from this Mac.

## Use

Load the GGUF with llama.cpp or another GGUF runner. This is not an MLX checkpoint. No new quality measurement was made for this upload.
