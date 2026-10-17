---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - quantized
  - nf4dq
  - llama.cpp
---

# Qwen3.8-27B NF4DQ

Qwen3.8-27B quantised to **NF4DQ**, 4.1562 bits per weight. 13.23 GiB.

> **This file requires a fork of llama.cpp.** It will not load correctly in
> stock llama.cpp. See below.

## Measured

Perplexity on wikitext-2 at context 512, A100-80GB, full test file:

| Build | Size | PPL @512 |
|---|---|---|
| BF16 | 50.9 GiB | 6.9530 |
| Q4_K_M | 15.652 GiB | 6.9787 |
| **NF4DQ** | **13.230 GiB** | **7.0890** |
| Q3_K_M | 12.643 GiB | 7.3362 |

Against llama.cpp's own K-quant size-quality curve, interpolating between
Q3_K_M and Q4_K_M puts a K-quant at 4.16 bpw near 7.29. NF4DQ measured 7.089.

Throughput against Q4_K_M, same A100, same `llama-bench` run:

| Build | Size | prefill | decode |
|---|---|---|---|
| **NF4DQ** | **13.23 GiB** | 734.52 t/s | **60.81 t/s** |
| Q4_K_M | 17.66 GiB | 1300.04 t/s | 48.67 t/s |

25% smaller and 25% faster at decode. Prefill is slower because the fork has
no NF4DQ tile loader in `mmq.cu` yet.

On an NVIDIA L4 (23 GB) the whole model runs in about 16.9 GB with an 8k
context, at 16.84 t/s decode, or 30.13 t/s with multi-token prediction
(`--spec-type draft-mtp --spec-draft-n-max 4`). Q4_K_M at 17.66 GiB leaves far
less room on a 24 GB card.

## Requires the NF4DQ fork

These weights use ggml type 43, which exists only in
[KayceeSamuel/llama.cpp](https://github.com/KayceeSamuel/llama.cpp) (branch
`nf4dq`). Stock llama.cpp does **not** support them.

Type 43 is currently the next free slot upstream. If upstream adds a
quantisation type, a stock build will read these files as that type and decode
garbage rather than refusing. If output looks wrong, check which build you are
running before anything else.

```bash
git clone https://github.com/KayceeSamuel/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release   # omit -DGGML_CUDA on Apple Silicon
cmake --build build -j
```

Then:

```bash
./build/bin/llama-server -m MODEL.gguf -ngl 99 -c 8192 -fa on
```

## What NF4DQ is

A 4.1562 bits-per-weight ggml block type: NF4 weight levels with a second
quantisation stage that stores each 32-weight sub-block's scale as a 4-bit
index into a fixed codebook rather than an fp16 value. That second stage is
what takes it below Q4_K's 4.50 bpw.

Design notes, the fitting data behind every constant, and the approaches that
were measured and rejected are in the
[fork's design doc](https://github.com/KayceeSamuel/llama.cpp/blob/nf4dq/docs/nf4dq-design.md).

## Backends

CUDA and Metal have kernels. There is no Vulkan or SIMD CPU kernel, so AMD and
Intel GPUs fall back to a scalar CPU path that is a correctness gate rather
than a performance path.

## Licence

Apache 2.0, matching the base model. Quantisation does not create a new work.