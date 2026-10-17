---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3
  - abliterated
  - image-text-to-text
---

# Qwen3.8-27B Heretic Abliterated (GGUF) - personal mirror

A pinned personal copy of two files needed to run this model **with vision**.
Nothing was re-quantized or modified: both files are byte-identical to their
sources (verified by LFS sha256).

| file | size | copied from | source revision |
|---|---|---|---|
| `RVN-Q3_K_S-multilingual-mtp.gguf` | 11.7 GiB | [0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF) | `20b94f0613b632b4848bbe3b1e05d9ee0c2b1608` |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 600 MiB | [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) | `0669b98607d47046c7c2b3f801011d54a08cfccf` |

The weights are the abliterated build; the multimodal projector (`mmproj`) comes
from the official ggml-org repo. Both are required together - the projector alone
does nothing, and the weights alone give you a text-only model.

## Usage

```bash
llama-server -m RVN-Q3_K_S-multilingual-mtp.gguf              --mmproj mmproj-Qwen3.8-27B-Q8_0.gguf
```

Mirrored 2026-09-04. Licensed Apache-2.0, same as both sources.
