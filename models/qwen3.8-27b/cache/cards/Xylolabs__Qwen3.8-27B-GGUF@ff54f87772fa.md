---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
language:
- en
- ko
tags:
- gguf
- llama-cpp
- quantized
- qwen38
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Qwen3.8-27B GGUF</h1>

[![Base model](https://img.shields.io/badge/base-Qwen%2FQwen3.8--27B-blue)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![Formats](https://img.shields.io/badge/formats-Q8__0%20Q6__K%20Q5__K__M%20Q3__K__M%20IQ3__M%20Q2__K-5B2C83)](#files)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-llama.cpp-black)](https://github.com/ggml-org/llama.cpp/tree/cd959bae99ff4bed22bef2604f45241c09bbb8c8)

</div>

GGUF quants of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for llama.cpp, on CUDA, Metal or CPU.


## Files

| File | Size |
|---|---:|
| `Qwen3.8-27B-Q8_0.gguf` | 29.047 GB |
| `Qwen3.8-27B-Q6_K.gguf` | 22.431 GB |
| `Qwen3.8-27B-Q5_K_M.gguf` | 19.536 GB |
| `Qwen3.8-27B-Q4_K_M.gguf` | 16.564 GB |
| `Qwen3.8-27B-IQ4_XS.gguf` | 15.420 GB |
| `Qwen3.8-27B-Q4_K_S.gguf` | 15.358 GB |
| `Qwen3.8-27B-IQ4_XS-imat.gguf` | 15.309 GB |
| `Qwen3.8-27B-Q3_K_M.gguf` | 13.501 GB |
| `Qwen3.8-27B-IQ3_M.gguf` | 12.768 GB |
| `Qwen3.8-27B-Q2_K.gguf` | 10.865 GB |
| `mmproj-Qwen3.8-27B-F16.gguf` | 0.928 GB |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.629 GB |

## How to run

Use llama.cpp commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8` or later.

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout cd959bae99ff4bed22bef2604f45241c09bbb8c8
cmake -B build -DCMAKE_BUILD_TYPE=Release        # add -DGGML_CUDA=ON for NVIDIA GPUs
cmake --build build -j --target llama-cli llama-server

./build/bin/llama-cli -m Qwen3.8-27B-Q5_K_M.gguf --jinja -c 8192 -ngl 99
```

Add `--chat-template-kwargs '{"enable_thinking":false}'` to answer directly, without the reasoning pass. `llama-server` takes the same flags.

## Accuracy

Mean KL divergence from BF16 (lower is better) and top-1 agreement with BF16 on English and Korean text, measured with llama.cpp.

| Quant | Size | Mean KL | top-1 |
|---|---:|---:|---:|
| **Q8_0** | **29.047 GB** | **0.0009** | **97.82%** |
| **Q6_K** | **22.431 GB** | **0.0023** | **96.83%** |
| **Q5_K_M** | **19.536 GB** | **0.0047** | **95.44%** |
| **Q4_K_M** | **16.564 GB** | **0.0097** | **93.46%** |
| **IQ4_XS** | **15.420 GB** | **0.0250** | **90.38%** |
| **Q4_K_S** | **15.358 GB** | **0.0161** | **92.12%** |
| **IQ4_XS-imat** | **15.309 GB** | **0.0195** | **91.65%** |
| **Q3_K_M** | **13.501 GB** | **0.0545** | **86.75%** |
| **IQ3_M** | **12.768 GB** | **0.0614** | **85.95%** |
| **Q2_K** | **10.865 GB** | **0.1725** | **79.21%** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-mean-kl-dark.png">
  <img src="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-mean-kl.png" alt="Mean KL divergence by quant file size">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-top1-dark.png">
  <img src="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-top1.png" alt="Top-1 accuracy by quant file size">
</picture>

## License and notices

Apache License 2.0, copied in `LICENSE` (unchanged from the Qwen3.8-27B release). We converted the weights to GGUF and quantized them. `NOTICE` names the source model and its copyright holder. Xylolabs is not affiliated with or endorsed by Qwen or Alibaba Cloud.
