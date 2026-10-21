---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- gguf
- qwen3.8
---

# Qwen3.8-27B-GGUF (exact mirror, Q4_K_M)

Byte-for-byte mirror of the plain Q4_K_M GGUF and F16 vision projector originally published by
Unsloth in `unsloth/Qwen3.8-27B-GGUF` (the plain Q4_K_M file has since been replaced upstream by
Dynamic "UD" variants). Hosted here so the exact tested file stays at a stable address.

| File | SHA-256 | Bytes |
|------|---------|-------|
| `Qwen3.8-27B-Q4_K_M.gguf` | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | 17,106,775,008 |
| `mmproj-F16.gguf` | `cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e` | 927,607,488 |

`template` and `params` pin the chat template and sampling parameters used with these files.
Credit for the model and quantization belongs to Qwen and Unsloth; license Apache 2.0 as published.
