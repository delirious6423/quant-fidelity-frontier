---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.5
---

# Qwen3.8-27B — UD-Q4_K_M GGUF, MTP block removed

A mirror of one file for the Recogment app's on-device language model, so the app downloads a file it has
verified byte for byte.

| | |
|---|---|
| File | `Qwen3.8-27B-UD-Q4_K_M-nomtp.gguf` — 16,213,250,016 bytes |
| SHA-256 | `6fd7662c4901c266deb1713ed04bc57101066f2dac4fdb4ac3f8163b82a232f2` |
| Made from | `Qwen3.8-27B-UD-Q4_K_M.gguf` in [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) — 16,564,259,744 bytes, SHA-256 `1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529` |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Licence | Apache License 2.0 — `LICENSE`, copied unchanged from the base model |

## What was changed (Apache License 2.0, section 4(b))

**This file is MODIFIED from the file it was made from.** The model's multi-token-prediction (NextN / MTP) block —
a speculative-decoding aid that the normal forward pass never reads — was removed, because the llama.cpp build
the app uses predates support for it in this architecture:

- `qwen35.block_count` 65 → 64; the `qwen35.nextn_predict_layers` key is dropped;
- the 15 tensors of the dropped block are removed (866 → 851 tensors);
- every other tensor's bytes are copied unchanged (offsets re-packed at the file's 32-byte alignment).

No weight was re-quantized, re-trained or edited. The change is deterministic: running the same tool on the source
file again gives a byte-identical result (checked 2026-11-23).

## Licence

Licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Alibaba Cloud for the model.
Distributed "AS IS", without warranties or conditions of any kind.
