---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE
pipeline_tag: image-text-to-text
language: [zh, en]
tags: [orinfer, jetson, e8p, int8, mixed-precision, vision-language, safetensors]
---

# Qwen3.8-Flash-Next · Orinfer E8P Q2A8

[中文](./README_zh.md) · [ModelScope](https://modelscope.cn/models/iammiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8)

Official [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) quantized for **Jetson AGX Orin 64GB / CUDA SM87**, using [Orinfer v0.1.4](https://github.com/iMMIQ/orinfer/releases/tag/v0.1.4).

Routed experts use our integer E8P near-2-bit representation and INT8 computation. GDN QKV/Z, QSA Q/gate and the shared output head use signed-block128 Q4A8. Output projections, shared FFN and sensitive MTP projections retain W8; HC retains BF16 and vision FP16. CPU embedding/PLE tables use bounded row caches.

Supports text, images/multiple images, thinking, native MTP, prefix caching and continuous batching. Context capacity is **262,144 tokens** including input and generated tokens. Persistent GDN state is FP32 and KV cache is INT8. Quantization seed: `20261002`.

## Files and compatibility

- 27 GPU safetensors shards and 8 CPU table shards; total stored safetensors size **52.94 GB** (decimal).
- `cache/weights/model.safetensors.index.json` maps GPU tensors; `cache/model.json` describes the native physical layouts and fixes the execution-package digest.
- These are Orinfer-native quantized weights. They cannot be loaded directly by Transformers, vLLM, SGLang or llama.cpp.
- This repository contains model data only. Install the matching `.so`/cubin execution package from GitHub Release; it is not included here.
- Required execution-package digest: `239440d85b176b63acd77df82a0ddcc14d44d1967bb94a03e0eb3f1c0e429932`.

## Deploy

Download the model from Hugging Face or [ModelScope](https://modelscope.cn/models/iammiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8). Get the CLI and execution archives from [v0.1.4](https://github.com/iMMIQ/orinfer/releases/tag/v0.1.4). Installation and inference use the Rust CLI; **Python, PyTorch, TileLang and a Rust compiler are not needed to deploy prebuilt packages**.

```bash
hf download immiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8 --local-dir ./model
# Download the CLI archive, matching execution archive and SHA256SUMS from v0.1.4.
sha256sum --ignore-missing -c SHA256SUMS
tar -xf orinfer-v0.1.4-linux-aarch64.tar.xz
./orinfer-v0.1.4-linux-aarch64/bin/orinfer package install \
  ./orinfer-v0.1.4-flash_next-int8-quality-sm87-execution.tar.xz
./orinfer-v0.1.4-linux-aarch64/bin/orinfer validate-model ./model
./orinfer-v0.1.4-linux-aarch64/bin/orinfer serve ./model \
  --model qwen-flash-next --cuda-graph decode_only --mtp-drafts 3 \
  --default-request-params '{"enable_thinking":true}'
```

The service listens on `0.0.0.0:8088`; the OpenAI-compatible endpoint is `/v1/chat/completions`. Explicit request fields override deployment defaults. Streaming is opt-in with `stream: true`; thinking uses `reasoning_content` and images use `image_url`.

## Validation and license

The matching v0.1.4 packages passed complete model-file validation and real CLI/API regressions for both models, including MTP, prefix replay, all CUDA Graph modes, images/multiple images and two concurrent requests. Token choices need not be identical to BF16. Full LLM benchmarks and independent BF16 256K quality evaluation remain outside this acceptance scope. Dated results and limitations are in [performance documentation](https://github.com/iMMIQ/orinfer/blob/v0.1.4/docs/performance.md).

Weights retain [Qwen Community License 1.0](./LICENSE). Orinfer software and its separate execution package are LGPL-3.0-or-later; their third-party notices ship with the software release.
