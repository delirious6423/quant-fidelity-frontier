---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
language: [zh, en]
tags: [orinfer, jetson, e8p, int8, mixed-precision, vision-language, safetensors]
---

# Qwen3.8-27B · Orinfer E8P Q4A8

[中文](./README_zh.md) · [ModelScope](https://modelscope.cn/models/iammiq/Qwen3.8-27B-Orinfer-E8P-Q4A8)

Official [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) quantized for **Jetson AGX Orin 64GB / CUDA SM87**, using [Orinfer v0.1.4](https://github.com/iMMIQ/orinfer/releases/tag/v0.1.4).

Main projections use our two-stage residual E8P Q4 representation with coefficients [4, 1], signed-block128 rotation, beam-8 encoding and INT8 computation. Sensitive embedding/output/MTP and vision tensors retain the validated mixed-precision exceptions. The GPU weight payload is 16.52 GB, approximately 4.76 bits per parameter including metadata and high-precision exceptions.

Supports text, images/multiple images, thinking, native MTP, prefix caching and continuous batching. Context capacity is **262,144 tokens** including input and generated tokens. Persistent GDN state is FP32 and KV cache is INT8. Quantization seed: `20261002`.

## Files and compatibility

- 68 GPU safetensors shards and 0 CPU table shards; total stored safetensors size **16.57 GB** (decimal).
- `cache/weights/model.safetensors.index.json` maps GPU tensors; `cache/model.json` describes the native physical layouts and fixes the execution-package digest.
- These are Orinfer-native quantized weights. They cannot be loaded directly by Transformers, vLLM, SGLang or llama.cpp.
- This repository contains model data only. Install the matching `.so`/cubin execution package from GitHub Release; it is not included here.
- Required execution-package digest: `17e754396399b16ba8a588d992b5e67bb19cf84af787b8a9ce1cff00c1a97fbb`.

## Deploy

Download the model from Hugging Face or [ModelScope](https://modelscope.cn/models/iammiq/Qwen3.8-27B-Orinfer-E8P-Q4A8). Get the CLI and execution archives from [v0.1.4](https://github.com/iMMIQ/orinfer/releases/tag/v0.1.4). Installation and inference use the Rust CLI; **Python, PyTorch, TileLang and a Rust compiler are not needed to deploy prebuilt packages**.

```bash
hf download immiq/Qwen3.8-27B-Orinfer-E8P-Q4A8 --local-dir ./model
# Download the CLI archive, matching execution archive and SHA256SUMS from v0.1.4.
sha256sum --ignore-missing -c SHA256SUMS
tar -xf orinfer-v0.1.4-linux-aarch64.tar.xz
./orinfer-v0.1.4-linux-aarch64/bin/orinfer package install \
  ./orinfer-v0.1.4-qwen3_5-int8-quality-sm87-execution.tar.xz
./orinfer-v0.1.4-linux-aarch64/bin/orinfer validate-model ./model
./orinfer-v0.1.4-linux-aarch64/bin/orinfer serve ./model \
  --model qwen3.8-27b --cuda-graph decode_only --mtp-drafts 3 \
  --default-request-params '{"enable_thinking":true}'
```

The service listens on `0.0.0.0:8088`; the OpenAI-compatible endpoint is `/v1/chat/completions`. Explicit request fields override deployment defaults. Streaming is opt-in with `stream: true`; thinking uses `reasoning_content` and images use `image_url`.

## Validation and license

The matching v0.1.4 packages passed complete model-file validation and real CLI/API regressions for both models, including MTP, prefix replay, all CUDA Graph modes, images/multiple images and two concurrent requests. Token choices need not be identical to BF16. Full LLM benchmarks and independent BF16 256K quality evaluation remain outside this acceptance scope. Dated results and limitations are in [performance documentation](https://github.com/iMMIQ/orinfer/blob/v0.1.4/docs/performance.md).

Weights retain [Apache-2.0](./LICENSE). Orinfer software and its separate execution package are LGPL-3.0-or-later; their third-party notices ship with the software release.
