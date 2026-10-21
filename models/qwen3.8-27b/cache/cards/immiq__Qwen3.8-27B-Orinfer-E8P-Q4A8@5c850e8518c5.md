---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
language: [zh, en]
tags: [orinfer, jetson, e8p, int8, mixed-precision, vision-language, safetensors]
---

# Qwen3.8-27B · Orinfer E8P Q4A8

[中文](README_zh.md) · [Hugging Face](https://huggingface.co/immiq/Qwen3.8-27B-Orinfer-E8P-Q4A8) · [ModelScope](https://modelscope.cn/models/iammiq/Qwen3.8-27B-Orinfer-E8P-Q4A8)

A quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) for **Jetson AGX Orin 64GB / CUDA SM87**, deployed with [Orinfer v0.2.0](https://github.com/iMMIQ/orinfer/releases/tag/v0.2.0). It supports text, images and multiple images, streaming Chat Completions, native MTP, prefix caching, continuous batching and resumable JSONL offline inference. Translation requests use `enable_thinking: false`; thinking follows the original model's chat template.

## Quantization and files

Two-stage residual E8P Q4 with coefficients [4, 1], signed-block128 rotation and beam-8 encoding over the original BF16 weights, paired with INT8 computation. Sensitive tensors retain the validated W8/FP16/BF16/FP32 mixed-precision policy.

Deployed read-only GPU data, including scales, derived constants and precision exceptions: **16,518,393,824 bytes**, approximately **4.757 bits per recorded GPU weight parameter**. Stored safetensors: **16.57 GB** across **68 shards**. Context capacity: **262,144 tokens**; KV cache is INT8 and persistent GDN state is FP32. Quantization seed: `20261002`.

The repository contains HF configuration/tokenizer assets, the native chat template, standard safetensors containers and `cache/model.json`. The descriptor fixes physical tensor layouts and the execution-package digest. The matching Rust `.so`, cubins and their source are distributed through GitHub Release.

## Deploy

Download this model repository into `./model`. Prebuilt deployment uses the aarch64 Rust CLI, NVIDIA CUDA Driver and the matching SM87 execution package.

Execution-package digest: `1337e1d358bb035419df566fdebd8679bdd343cdf0d8e7450bd2b97bdbb62a52`.

```bash
curl -fLO https://github.com/iMMIQ/orinfer/releases/download/v0.2.0/orinfer-v0.2.0-linux-aarch64.tar.xz
curl -fLO https://github.com/iMMIQ/orinfer/releases/download/v0.2.0/orinfer-exec-qwen35-27b-a8-sm87-v0.2.0.tar.xz
curl -fLO https://github.com/iMMIQ/orinfer/releases/download/v0.2.0/SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
tar -xf orinfer-v0.2.0-linux-aarch64.tar.xz
./orinfer-v0.2.0-linux-aarch64/bin/orinfer package install ./orinfer-exec-qwen35-27b-a8-sm87-v0.2.0.tar.xz
./orinfer-v0.2.0-linux-aarch64/bin/orinfer validate-model ./model
./orinfer-v0.2.0-linux-aarch64/bin/orinfer serve ./model \
  --model qwen3.8-27b --cuda-graph decode_only --mtp-drafts 3
```

The listener is `[::]:8088`. OpenAI Chat Completions uses `/v1/chat/completions`; set `stream: true` for SSE. Images use `image_url`, and thinking output uses `reasoning_content`. The CLI also provides Ollama-compatible APIs and `orinfer batch` for JSONL jobs. See the [serving guide](https://github.com/iMMIQ/orinfer/blob/v0.2.0/docs/serving.md).

## Measured performance

Jetson AGX Orin 64GB, **2026-10-21**. Throughput is **tok/s**; startup is seconds.

| Prefill 512 | Prefill 8K | Single decode, MTP | 8-request decode | Max requests / decode | Serve → API (s) | Serve → first output (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 658.87 | 598.46 | 33.64 | 67.56 | 64 / 135.89 | 6.68 | 7.05 |

Single-request runs enable adaptive MTP up to 3 drafts, with 512/8192 input tokens and 257 fixed output tokens; the single-decode column uses 512-token input. Multi-request runs automatically switch the benchmark to MTP off, with 256 input and 65 output tokens per request. Decode timing starts after the first emitted step, and multi-request timing uses a first-output barrier. Each throughput value is a median of three warm runs after one warmup; fixed-length diagnostics ignore EOS. All use greedy, seed `20261002`, `decode_only`, thinking off and prefix cache off. MTP speed depends on the task and acceptance rate.

Startup uses three fresh Rust `serve` processes and a 21-token translation request: process invocation to `/health` ready, then to the first nonempty SSE output. Existing file caches are preserved; SSE output can contain several tokens. The table reports medians, and file-cache effects plus individual large-model startup observations are in the [performance reference](https://github.com/iMMIQ/orinfer/blob/v0.2.0/docs/performance.md).

## Quality scope

Validation covers fixed-history probability diagnostics, request-state isolation, MTP/prefix recovery and text/image API requests. Dated BF16 comparisons and evaluated contexts are recorded in the performance reference.

These are small-sample diagnostics covering text, inputs through 8K and controlled simple images. Broader task benchmarks, complex image translation and full-context quality remain evaluation areas. Quality and execution-regression scopes are recorded separately in the [performance reference](https://github.com/iMMIQ/orinfer/blob/v0.2.0/docs/performance.md).

## License

Model weights retain [Apache-2.0](LICENSE). Orinfer software and its separate execution package use LGPL-3.0-or-later; source and third-party notices ship with the release.
