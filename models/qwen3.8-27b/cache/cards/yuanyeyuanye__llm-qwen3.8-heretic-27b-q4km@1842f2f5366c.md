---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [gguf, qwen3.8, heretic, llama.cpp]
---

# Qwen3.8-27B heretic Q4_K_M (RunPod cache)

Single-model cache for a RunPod Serverless llama.cpp endpoint. RunPod Cached Models download a whole repository, so
this repository holds only the two files the endpoint serves.

| File | SHA-256 | Source |
| --- | --- | --- |
| `qwen3.8-heretic-27b-q4km.gguf` | `b53709fa25c35df7d6ab9be1dbe21eeff5fdeb4b6a87ae7613566cc67ad168c9` | The Ollama `qwen3.8-heretic:27b-q4km` model layer, imported from [0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF) `RVN-Q4_K_M-multilingual.gguf`. Same size and GGUF metadata as that file at `ab8d7bd8`, whose SHA-256 is `9ed40ccc…`. |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | `2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb` | Identical to the source repository's projector. |

Licence: Apache-2.0, as the source repository and base model.
