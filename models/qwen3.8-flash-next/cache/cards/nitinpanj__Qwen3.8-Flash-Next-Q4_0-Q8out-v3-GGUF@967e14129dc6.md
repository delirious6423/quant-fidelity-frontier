---
license: other
license_name: qwen-community-license-1.0
license_link: https://huggingface.co/Qwen/LICENSE
language:
- en
- zh
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen4exp
- moe
- mtp
- q4_0
- imatrix
- apple-silicon
- metal
- speculative-decoding
- vision
- image-text-to-text
---

# Qwen3.8-Flash-Next Q4_0-Q8_0-out v3 (GGUF)

> [!IMPORTANT]
> ### ⚠️ Runtime Compatibility Notice
> This is a **`qwen4exp`** architecture GGUF (Gated-DeltaNet hybrid + 512-expert routed MoE + MTP).
> Stock `llama.cpp` does not support this architecture. Use a `qwen4exp`-enabled llama.cpp build
> (e.g. the `unslothai/llama.cpp` fork with qwen4exp support) with `--mtp-draft` for speculative decoding.

## Overview

Quantized GGUF build of **Qwen3.8 Flash Next** (`general.size_label: 512x56B`), the 48-layer Qwen4exp
hybrid MoE (24 heads, 2 KV heads, GDN recurrent layers with full attention every 4 layers,
512 experts / 10 active, ngram-PLE layer, built-in MTP heads), 262,144-token native context.

- **Quantization:** Q4_0 weights with **Q8_0 output/embedding tensors** (`-Q8out` v3 recipe), imatrix-calibrated
  (`Qwen3.8-Flash-Next-calibration-v6`, 926 entries / 582 chunks)
- **Shards:** 3 (`-00001-of-00003.gguf` … `-00003-of-00003.gguf`), 96 GiB total
- **MTP:** Multi-Token Prediction heads included for speculative decoding (7 proposal tokens)
- **Vision:** vision projector tensors included (image-text-to-text capable)

## Files

| File | Size |
| :--- | ---: |
| `Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00001-of-00003.gguf` | 37.7 GiB |
| `Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00002-of-00003.gguf` | 37.8 GiB |
| `Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00003-of-00003.gguf` | 20.0 GiB |

Download all three shards into the same directory, then load shard 1.

## Usage (qwen4exp-enabled llama.cpp)

```bash
llama-server -m Qwen3.8-Flash-Next-Q4_0-Q8out-v3-00001-of-00003.gguf \
  --mmproj mmproj-f16.gguf -ngl 99 -c 262144 --port 8080
```

Built and benchmarked locally on Apple M5 Pro (64 GB unified memory) with the Slipstream streaming-MoE
llama.cpp fork.
