---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE
pipeline_tag: image-text-to-text
language:
  - zh
  - en
tags:
  - orinfer
  - jetson
  - e8p
  - int8
  - mixed-precision
  - moe
  - vision-language
  - safetensors
---

# Qwen3.8-Flash-Next · Orinfer E8P Q2A8

[中文](./README_zh.md) · [ModelScope](https://modelscope.cn/models/iammiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8)

A quantized deployment of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for **Jetson AGX Orin 64GB**, using [Orinfer](https://github.com/iMMIQ/orinfer). Supports text, images, multiple images, thinking, prefix caching and native MTP.

## Model

| Item | Configuration |
| --- | --- |
| Routed experts | Near-2-bit E8P encoding with rotation; INT8 computation |
| Dense projections, shared experts, output head | W8; mixed precision for sensitive paths |
| Vision encoder | FP16 |
| KV cache | Group-64 INT8 with FP16 scales |
| Context capacity | 262,144 tokens, including prompts, images and output |
| Download size | Approximately 51.3 GB |
| GPU read-only weights | Approximately 34.52 GiB |

E8P stores each eight-weight vector in a 16-bit code. A fixed signed Hadamard rotation over 128 input channels spreads outlier energy before encoding; scales and codebook metadata add a small overhead above 2 bits per expert weight. Quantization is weight-only with seed `20261002`; a full LLM quality benchmark remains to be added.

## Deploy

Build Orinfer using its [quick start](https://github.com/iMMIQ/orinfer#快速开始), then download the complete model directory and run:

```bash
hf download immiq/Qwen3.8-Flash-Next-Orinfer-E8P-Q2A8 \
  --local-dir ./qwen-flash-next-orinfer

./target/release/orinfer serve ./qwen-flash-next-orinfer \
  --listen 0.0.0.0:8088 --model qwen-flash-next \
  --max-active-requests 1 --prefix-cache-mib 512 \
  --cuda-graph full --mtp-drafts 7
```

The OpenAI-compatible Chat Completions endpoint is `http://localhost:8088/v1`. Set `enable_thinking: true` to enable thinking; provide images through `image_url`. See the [API examples](https://github.com/iMMIQ/orinfer/blob/main/docs/serving.md).

## Performance

Measured on **Jetson AGX Orin 64GB**, MAXN, GPU 1.30 GHz, CUDA 12.6.

### Text prefill

| Input tokens | Tokens/s | Time to first token |
| ---: | ---: | ---: |
| 512 | 589 | 0.87 s |
| 2,048 | 794 | 2.58 s |
| 8,192 | 809 | 10.13 s |

Single stream, full CUDA Graph, MTP with up to 7 drafts, thinking off, prefix cache disabled. Median of three warmed requests per length; throughput is input tokens divided by end-to-end time to first token.

### Decode with MTP

| Code generation task | Tokens/s |
| --- | ---: |
| Python merge sort | 53.84 |
| Rust LRU cache | 41.19 |
| TypeScript async map | 30.25 |

Same text weights, single stream, `decode_only` CUDA Graph, MTP with up to 7 drafts, greedy decoding, seed `20261002`, thinking off. Measurements use warmed native replay with 96 generated tokens; throughput counts committed output tokens and excludes the first token and prefill time.

## License

Model weights: [Qwen Community License 1.0](./LICENSE). Orinfer and execution-package code: [LGPL-3.0-or-later](https://github.com/iMMIQ/orinfer/blob/718fb31ac82093641e83fbcd5070185ea3238015/LICENSE); [corresponding source](https://github.com/iMMIQ/orinfer/tree/718fb31ac82093641e83fbcd5070185ea3238015). Third-party components retain their respective licenses.
