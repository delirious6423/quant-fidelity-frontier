---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
- huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF
- unsloth/DeepSeek-V4.1-Flash-1017-GGUF
tags:
- gguf
- abliterated
- uncensored
- llama.cpp
- prima.cpp
- iroh
- lauburu
---
# Huihui-DeepSeek-V4.1-Flash-1017-abliterated Full-Sized Q4 (`UD-Q4_K_XL`) — Lauburu Sovereign Mirror

This repository is the **Lauburu Sovereign Mirror** for **`Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF` (`UD-Q4_K_XL`)** (`llama.cpp` & `prima.cpp` compatible GGUF v3).

## 1. Verified Huihui Abliterated Tensor Patch (`DeepSeek-V4.1-Flash-1017-UD-IQ3_S-patch.gguf` for `UD-Q4_K_XL` Layers 11–28)
- **File:** `unsloth/patch-UD-IQ3_S/DeepSeek-V4.1-Flash-1017-UD-IQ3_S-patch.gguf`
- **Size:** `765.0 MB` (`802162112` bytes)
- **SHA256:** `ae7226d14278eb9e83f127f34d93e6666ce73bb9e00968b27bb9f7029e512fee`
- **GGUF Version:** `v3` (`tensor_count = 18`, `metadata_kv_count = 3`)
- **Patcher Script:** `unsloth/patch-unsloth-DeepSeek-V4.1-Flash-1017.py`

## 2. Verified Master GGUF Split Header (`00001-of-00009.gguf`)
- **File:** `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf`
- **Size:** `5.51 MB` (`5780589` bytes)
- **SHA256:** `cc15d470afcb030558bbc4d8e2ccd840182e2e95de1a95198138efe63a5fd99f`
- **GGUF Version:** `v3` (`metadata_kv_count = 92`)

## 3. Full 9-Part `UD-Q4_K_XL` Split Inventory (`466.30 GiB`)
| Shard Path | Size (GiB) | LFS SHA256 OID |
| :--- | :--- | :--- |
| `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf` | `0.01 GiB` | `cc15d470afcb0305...` |
| `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00002-of-00009.gguf` | `44.82 GiB` | `17df217d8fc26a39...` |
| `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00003-of-00009.gguf` | `44.82 GiB` | `0c8065f620a049cb...` |
| `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00004-of-00009.gguf` | `44.82 GiB` | `3bd8b2ebf8f51d87...` |
| `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00005-of-00009.gguf` | `44.82 GiB` | `929054a6bbf37017...` |
| `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00006-of-00009.gguf` | `44.82 GiB` | `49d3de04bd6a09de...` |
| `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00007-of-00009.gguf` | `44.82 GiB` | `f18feae5878dd680...` |
| `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00008-of-00009.gguf` | `188.83 GiB` | `12df293355efbd10...` |
| `UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00009-of-00009.gguf` | `8.54 GiB` | `c21ad6105a780bc2...` |
| `engram/DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf_file` | `103.0 GiB` | `4de55238158d2300...` |
