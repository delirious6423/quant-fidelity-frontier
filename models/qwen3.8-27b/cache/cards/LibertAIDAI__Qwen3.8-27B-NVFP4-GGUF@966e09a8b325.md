---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
quantized_by: LibertAIDAI
tags:
  - gguf
  - llama.cpp
  - nvfp4
  - blackwell
  - qwen3.8
  - multimodal
language:
  - en
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B NVFP4 GGUF

NVFP4 GGUF quantizations of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for [llama.cpp](https://github.com/ggml-org/llama.cpp), built for native NVFP4 tensor-core math on Blackwell GPUs.

The NVFP4 weights come from our own ModelOpt post-training quantization of the BF16 release, using NVIDIA's dense recipe: NVFP4 at group size 16 on the language-model MLP linears only, calibrated on 512 samples of cnn_dailymail at sequence length 512. Attention (both the full-attention and Gated-DeltaNet linear-attention layers), the vision tower, the MTP block, embeddings and `lm_head` are left in BF16. That is 192 quantized linears, 64 layers by gate, up and down.

An MTP variant with the speculative draft head is published separately at [Qwen3.8-27B-NVFP4-MTP-GGUF](https://huggingface.co/LibertAIDAI/Qwen3.8-27B-NVFP4-MTP-GGUF).

## About LibertAI

[LibertAI](https://libertai.io) is a decentralized AI platform offering private inference, an OpenAI-compatible API, and a chat UI, all running on community GPUs over [Aleph Cloud](https://aleph.cloud) rather than a single company's servers.

If you want to put this model to work as an autonomous agent without running your own infrastructure, see [LiberClaw](https://liberclaw.ai), Hermes-style agents hosted on Aleph Cloud with LibertAI inference. Free tier: 2 agents, no credit card, 5 minutes to deploy. Open source.

## Files

| File | Size | FFN | Other tensors | When to pick |
|------|------|-----|---------------|--------------|
| `Qwen3.8-27B-NVFP4-Q4_K_M.gguf` | 15.8 GB | NVFP4 | Q4_K_M | **Recommended.** Fastest variant and the smallest |
| `Qwen3.8-27B-NVFP4-Q8_0.gguf` | 20.0 GB | NVFP4 | Q8_0 | Higher-precision attention and embeddings if you have the VRAM |
| `Qwen3.8-27B-NVFP4-BF16.gguf` | 29.2 GB | NVFP4 | BF16 | Source-precision non-FFN tensors, slower in practice |
| `mmproj-Qwen3.8-27B-F16.gguf` | 0.93 GB | n/a | F16 vision tower | Required for image and video input, reusable across all Qwen3.8-27B GGUFs |

Q4_K_M is both the smallest and the fastest of the three. Inference here is memory-bandwidth bound, so the larger BF16 attention and embedding tensors bottleneck the model even though the FFN math is identical.

## Performance

Measured on an NVIDIA RTX 5090 (32 GB, Blackwell, sm_120, CUDA 13.0), llama.cpp build `849798132`, compared against a stock Q4_K_M built from the same BF16 source on the same build.

![NVFP4 vs stock Q4_K_M on RTX 5090](chart-serving.png)

Single-stream, `llama-bench`, 3 repetitions:

| Variant | Size | pp512 tok/s | tg64 tok/s |
|---|---|---|---|
| NVFP4-Q4_K_M | 14.72 GiB | 5081 | 84.8 |
| NVFP4-Q8_0 | 18.65 GiB | 5169 | 71.8 |
| Stock Q4_K_M | 15.40 GiB | 3947 | 81.2 |

Batched serving, `llama-batched-bench`, 512 in and 128 out, total throughput:

| Parallel | NVFP4-Q4_K_M | Stock Q4_K_M | Difference |
|---|---|---|---|
| 1 | 398 tok/s | 376 tok/s | +5.9% |
| 4 | 1138 tok/s | 980 tok/s | +16.1% |
| 8 | 1520 tok/s | 1188 tok/s | +28.0% |
| 16 | 2134 tok/s | 1824 tok/s | +17.0% |

Prefill is roughly 29% faster across the range. The token-generation advantage peaks near 8 concurrent requests and narrows at 16, where generation becomes attention and KV-cache bound rather than FFN bound.

## Usage

```bash
llama-server \
  -m Qwen3.8-27B-NVFP4-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-F16.gguf \
  -ngl 999 -fa on -c 32768 \
  --host 0.0.0.0 --port 8080
```

Drop `--mmproj` for text-only serving.

## Requirements

Native NVFP4 tensor-core math needs a Blackwell GPU (sm_120) and a llama.cpp built with CUDA 12.8 or newer. Build with `-DCMAKE_CUDA_ARCHITECTURES=120`. On older architectures the NVFP4 tensors still load and run, but without native NVFP4 MMA the speed advantage does not apply.

## Caveats

- Calibration was English news text (cnn_dailymail), matching NVIDIA's published recipe. Multilingual and code-heavy workloads were not measured against a held-out set.
- The vision tower is left in BF16 and was not separately evaluated. Vision quality should track the upstream BF16 release closely.
- Benchmarks are from a single RTX 5090. Behavior on other Blackwell parts was not measured.

## License and attribution

Inherits Apache 2.0 from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). All rights, responsibilities and acceptable-use policies of the upstream license apply.

Quantization performed by [LibertAI](https://libertai.io).
