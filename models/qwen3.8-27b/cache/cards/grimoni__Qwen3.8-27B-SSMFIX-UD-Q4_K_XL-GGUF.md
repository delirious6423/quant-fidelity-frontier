---
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
library_name: gguf
tags:
- gguf
- llama.cpp
- qwen3_5
- multimodal
- ssm-fix
- conv1d-repair
- quantized
- q4_k
language:
- en
- zh
---

# Qwen3.8-27B-SSMFIX-UD-Q4_K_XL-GGUF

GGUF quantization of `redashes/Qwen3.8-27B-BF16-SSMFIX`, derived from `Qwen/Qwen3.8-27B`, prepared for `llama.cpp`-compatible runtimes.

## Files

- `Qwen3.8-27B-SSMFIX-UD-Q4_K_XL.gguf`: main quantized checkpoint
- `sha256.txt`: checksum for the GGUF
- `quantization_recipe.txt`: quantization recipe used for this build
- `quantize_qwen38_ssmfix_q4xl.log`: quantization log
- `convert_qwen38_ssmfix.log`: BF16-to-GGUF conversion log

## Source

- Base model family: `Qwen/Qwen3.8-27B`
- Input checkpoint used locally: `redashes--Qwen3.8-27B-BF16-SSMFIX`
- Quantized from a local BF16 GGUF converted from the original safetensors checkpoint

## Quantization notes

- Target profile: `Q4_K_XL-like`
- Output size is about `15 GB`
- The quantization used an importance matrix compatible with `Qwen3.8-27B`
- The MTP head (`blk.64`) is not covered by the downloaded imatrix, so `nextn.eh_proj` was stored as `Q4_0`

## Local validation

Validated locally on August 19, 2026 with:

- GGUF header and metadata checks
- `llama.cpp` loading on a local NVIDIA GPU
- HTTP inference through `llama-server`

## Runtime

This file is intended for `llama.cpp` and compatible tools that support GGUF.

## Example with llama.cpp

```bash
llama-cli -m Qwen3.8-27B-SSMFIX-UD-Q4_K_XL.gguf -ngl 20 -c 256
```

## Caveats

- This is a community quantization, not an official Qwen release.
- Behavior for MTP-related tensors may differ from builds using a fully matched calibration set.

