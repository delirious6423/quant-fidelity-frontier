---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.5
  - qsa
  - unsloth
---

# Qwen3.8-27B-QSA GGUF (UD-Q6_K)

Mirror of the Unsloth UD-Q6_K + QSA mmproj weights used by the Loom port
[`johnlockejrr/Qwen3.8-27B-QSA-loom`](https://huggingface.co/johnlockejrr/Qwen3.8-27B-QSA-loom).

Upstream quant / QSA indexers: [`higa2515/Qwen3.8-27B-QSA-GGUF`](https://huggingface.co/higa2515/Qwen3.8-27B-QSA-GGUF).

## Files

- `Qwen3.8-27B-QSA-Unsloth-UD-Q6_K.gguf` — text weights (in-GGUF MTP)
- `mmproj-Qwen3.8-27B-QSA-F16.gguf` — vision projector

## llama.cpp

Use a build that supports Qwen3.5 QSA indexers (e.g. Unsloth llama.cpp).
