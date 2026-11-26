---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
language: [zh, en]
tags: [orinfer, jetson, e8p, int8, mixed-precision, vision-language, safetensors]
---

# Qwen3.8-27B · Orinfer E8P Q4A8 (E8P-HQ + E2E Distillation)

[中文](README_zh.md) · [Hugging Face](https://huggingface.co/immiq/Qwen3.8-27B-Orinfer-E8P-Q4A8) · [ModelScope](https://modelscope.cn/models/iammiq/Qwen3.8-27B-Orinfer-E8P-Q4A8)

A high-quality quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) for **Jetson AGX Orin 64GB / CUDA SM87**, 100% drop-in compatible with [Orinfer v0.2.0](https://github.com/iMMIQ/orinfer/releases/tag/v0.2.0). It supports text, images and multiple images, streaming Chat Completions, native MTP, prefix caching, continuous batching and resumable JSONL offline inference. Translation requests use `enable_thinking: false`; thinking follows the original model's chat template.

> The original baseline weight-only build (68 shards) is preserved under the `v0.2.0` repository tag.

## Quantization and files

Upgraded directly from original BF16 weights with **E8P-Q4-HQ + QuIP#-style End-to-End Logit Distillation**, preserving exact kernel layout, execution-package digest and runtime speed (zero overhead):

1. **Per-block-128 activation-optimal rotation signs** (`k5120`, `k17408`) minimizing INT8 activation quantization error prior to projection.
2. **Full-rank 65,536-token Hessian Block-8 Mahalanobis LDLQ + 2-sweep cyclic coordinate descent** (beam 6/12, exact 3-ULP FP16 scale polish) across all 64 transformer layers (48 GDN + 16 Full Attention) and `lm_head`, plus Hessian-GPTQ W8 for `out_proj`/`o_proj`.
3. **End-to-End 64-layer Logit-KL Distillation** (Nesterov-SGD with exact FP16 validation early stopping): freezes all discrete E8P-Q4 `uint32` and W8 `int8` codewords while jointly optimizing the ~0.017% continuous FP16 row scales and RMSNorm weights across all 64 layers to eliminate cross-layer drift.
4. **MTP Layer 0 recalibration**: all 8 W8-GPTQ MTP projections calibrated directly on the E2E-refined 64-layer trunk activations. Vision encoder weights and W8 `embed_tokens` remain byte-exact.

### Strictly held-out quality comparison vs. BF16 teacher

Evaluated across 23 strictly disjoint held-out cases (**5,063 positions** spanning EN, ZH, Translation, Long-Context, Agent/Code and Math) plus native MTP drafting:

| Checkpoint (16.52 GB, 100% SM87 Drop-In) | Overall KL ↓ | ex-Agent KL ↓ | Top-1 ↑ | High-Conf Flips ↓ | Agent/Code KL ↓ | Math KL ↓ | MTP KL ↓ / Top-1 ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Previous `v0.2.0` baseline (weight-only) | `0.06218` | `0.05482` | `89.77%` | `62 / 2900` | `0.3462` | `0.0328` | `0.08993` / `85.89%` |
| **Current `E8P-Q4-HQ + E2E Distillation`** | **`0.03915` (-37.0%)** | **`0.03429` (-37.4%)** | **`92.28%` (+2.51pp)** | **`44 / 2900`** | **`0.2264` (-34.6%)** | **`0.0142` (-56.6%)** | **`0.05717` (-36.4%) / `89.44%` (+3.55pp)** |

Deployed read-only GPU data, including scales, derived constants and precision exceptions: **16,518,393,824 bytes**, approximately **4.757 bits per recorded GPU weight parameter**. Consolidated into **12 semantic safetensors shards** (`1–1.53 GiB` per layer group, down from 68 shards; `16.52 GB` stored). Context capacity: **262,144 tokens**; KV cache is INT8 and persistent GDN state is FP32. Verified against the official Orinfer v0.2.0 SM87 execution package (`BufferContract`, frontend digests, safetensors physical layouts and all 1,568 tensor payload SHA-256s).

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

Single-request runs enable adaptive MTP up to 3 drafts, with 512/8192 input tokens and 257 fixed output tokens; the single-decode column uses 512-token input. Multi-request runs automatically switch the benchmark to MTP off, with 256 input and 65 output tokens per request. Decode timing starts after the first emitted step, and multi-request timing uses a first-output barrier. Each throughput value is a median of three warm runs after one warmup; fixed-length diagnostics ignore EOS. All use greedy, seed `20261002`, `decode_only`, thinking off and prefix cache off. MTP speed depends on the task and acceptance rate (higher MTP Top-1 agreement from E2E-HQ calibration directly benefits speculative acceptance).

Startup uses three fresh Rust `serve` processes and a 21-token translation request: process invocation to `/health` ready, then to the first nonempty SSE output. Existing file caches are preserved; SSE output can contain several tokens. The table reports medians, and file-cache effects plus individual large-model startup observations are in the [performance reference](https://github.com/iMMIQ/orinfer/blob/v0.2.0/docs/performance.md).

## Quality scope

Validation covers fixed-history probability diagnostics, request-state isolation, MTP/prefix recovery and text/image API requests. Dated BF16 comparisons and evaluated contexts are recorded in the performance reference.

These are small-sample diagnostics covering text, inputs through 8K and controlled simple images. Broader task benchmarks, complex image translation and full-context quality remain evaluation areas. Quality and execution-regression scopes are recorded separately in the [performance reference](https://github.com/iMMIQ/orinfer/blob/v0.2.0/docs/performance.md).

## License

Model weights retain [Apache-2.0](LICENSE). Orinfer software and its separate execution package use LGPL-3.0-or-later; source and third-party notices ship with the release.
