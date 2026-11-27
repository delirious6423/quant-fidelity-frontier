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
[![Formats](https://img.shields.io/badge/formats-Q8__0%20Q6__K%20Q5__K__M%20Q4__K__M%20Q4__K__S%20IQ4__XS--imat%20Q3__K__M%20IQ3__M%20Q2__K-5B2C83)](#files)
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
| `Qwen3.8-27B-Q4_K_S.gguf` | 15.358 GB |
| `Qwen3.8-27B-IQ4_XS-imat.gguf` | 15.309 GB |
| `Qwen3.8-27B-Q3_K_M.gguf` | 13.146 GB |
| `Qwen3.8-27B-IQ3_M.gguf` | 12.768 GB |
| `Qwen3.8-27B-Q2_K.gguf` | 9.949 GB |
| `mmproj-Qwen3.8-27B-F16.gguf` | 0.928 GB |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 0.629 GB |

`Q3_K_M` and `Q2_K` use the per-tensor types of Unsloth's UD-Q3_K_XL and UD-Q2_K_XL files, read from their headers.

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

Against BF16.

| Quant | Size | WikiText-2 KL | top-1 | Chat KL | top-1 | Held-out KL | top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unsloth UD-Q8_K_XL | 31.458 GB | 0.000667 | 98.853% | 0.001444 | 99.069% | 0.000898 | 98.951% |
| Unsloth Q8_0 | 29.047 GB | 0.000867 | 98.672% | 0.002304 | 99.059% | 0.001067 | 98.794% |
| Q8_0 | 29.047 GB | 0.000867 | 98.672% | 0.002304 | 99.059% | 0.001067 | 98.794% |
| Unsloth UD-Q8_K_L | 28.046 GB | 0.000786 | 98.657% | 0.002376 | 98.980% | 0.001109 | 98.627% |
| Unsloth UD-Q6_K_XL | 25.299 GB | 0.001032 | 98.564% | 0.002039 | 98.588% | 0.001826 | 98.265% |
| Unsloth UD-Q6_K_L | 24.182 GB | 0.001215 | 98.642% | 0.002661 | 98.588% | 0.002118 | 98.284% |
| Unsloth UD-Q6_K_M | 23.087 GB | 0.001383 | 98.397% | 0.002909 | 98.373% | 0.002544 | 98.235% |
| Q6_K | 22.431 GB | 0.002161 | 97.951% | 0.004358 | 98.245% | 0.003333 | 97.824% |
| Unsloth UD-Q6_K | 21.980 GB | 0.001773 | 98.137% | 0.003402 | 98.265% | 0.003245 | 98.010% |
| Unsloth UD-Q5_K_XL | 20.977 GB | 0.002375 | 97.882% | 0.005848 | 98.000% | 0.004837 | 97.716% |
| Unsloth UD-Q5_K_M | 19.871 GB | 0.003068 | 97.412% | 0.006759 | 97.706% | 0.006460 | 97.039% |
| Q5_K_M | 19.536 GB | 0.004578 | 96.902% | 0.008445 | 97.598% | 0.008101 | 96.794% |
| Unsloth UD-Q5_K_S | 18.765 GB | 0.003995 | 97.201% | 0.007980 | 97.490% | 0.008646 | 96.588% |
| Unsloth UD-Q4_K_XL | 17.659 GB | 0.005644 | 96.608% | 0.015352 | 96.324% | 0.012974 | 96.225% |
| Unsloth Q4_1 | 17.541 GB | 0.015127 | 94.613% | 0.023512 | 95.480% | 0.026553 | 94.039% |
| Unsloth UD-Q4_K_M | 16.564 GB | 0.007576 | 96.201% | 0.020176 | 95.618% | 0.018755 | 95.431% |
| Q4_K_M | 16.564 GB | 0.010464 | 95.574% | 0.019746 | 95.745% | 0.019950 | 95.412% |
| Unsloth Q4_0 | 16.056 GB | 0.025561 | 93.049% | 0.041697 | 93.873% | 0.043021 | 93.049% |
| Unsloth UD-Q4_K_S | 15.358 GB | 0.010193 | 95.618% | 0.034123 | 94.588% | 0.030494 | 94.265% |
| Q4_K_S | 15.358 GB | 0.015845 | 94.549% | 0.030877 | 94.990% | 0.030020 | 94.324% |
| IQ4_XS-imat | 15.309 GB | 0.017617 | 94.157% | 0.032862 | 94.598% | 0.032699 | 93.775% |
| Unsloth UD-IQ4_XS | 14.353 GB | 0.016367 | 94.176% | 0.039900 | 94.127% | 0.036723 | 93.157% |
| Unsloth UD-Q3_K_XL | 13.146 GB | 0.026235 | 92.554% | 0.049744 | 93.863% | 0.065165 | 91.657% |
| Q3_K_M | 13.146 GB | 0.037681 | 91.431% | 0.059795 | 92.784% | 0.071429 | 91.186% |
| IQ3_M | 12.768 GB | 0.059505 | 89.637% | 0.087631 | 91.343% | 0.100871 | 89.118% |
| Unsloth UD-IQ3_S | 12.041 GB | 0.036043 | 91.770% | 0.080079 | 91.471% | 0.106168 | 89.118% |
| Unsloth UD-IQ3_XXS | 10.935 GB | 0.054159 | 89.627% | 0.101860 | 90.078% | 0.143572 | 86.402% |
| Unsloth UD-Q2_K_XL | 9.949 GB | 0.086977 | 87.289% | 0.166114 | 87.000% | 0.160427 | 85.824% |
| Q2_K | 9.949 GB | 0.118005 | 85.118% | 0.153745 | 87.588% | 0.200785 | 83.539% |
| Unsloth UD-IQ2_S | 8.423 GB | 0.127511 | 84.931% | 0.206836 | 84.941% | 0.238166 | 82.578% |
| Unsloth UD-IQ2_XS | 7.626 GB | 0.165819 | 82.338% | 0.244062 | 83.049% | 0.312767 | 79.510% |
| Unsloth UD-IQ2_XXS | 7.206 GB | 0.197141 | 81.167% | 0.234754 | 83.422% | 0.358335 | 78.186% |
| Unsloth UD-IQ1_M | 6.729 GB | 0.275141 | 77.760% | 0.524296 | 76.069% | 0.602536 | 70.343% |
| Unsloth UD-IQ1_S | 6.192 GB | 0.386317 | 73.931% | 0.704053 | 72.176% | 0.805275 | 65.510% |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-mean-kl-dark.png">
  <img src="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-mean-kl.png" alt="Mean KL divergence by quant file size">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-top1-dark.png">
  <img src="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-top1.png" alt="Top-1 accuracy by quant file size">
</picture>

## Which file for your hardware

| Hardware | Memory | File | Context |
|---|---|---|---:|
| RTX 4090 / RTX 3090 | 24 GB | `Qwen3.8-27B-Q4_K_M.gguf` (16.6 GB) | 32K |
| RTX 5090 | 32 GB | `Qwen3.8-27B-Q6_K.gguf` (22.4 GB) | 32K |
| RTX 6000 Ada, or 2 x 24 GB | 48 GB | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 128K |
| One H100 | 80 GB | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| RTX PRO 6000 Blackwell (GDDR7) | 96 GB | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| DGX Spark | 128 GB unified | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| Mac Studio M5 Max | 128 GB unified | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| One H200 (HBM3e) | 141 GB | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| One B200 | 192 GB | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| Mac Studio M5 Ultra | 256 GB unified | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| Mac Studio M5 Ultra | 512 GB unified | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| One B300 (HBM3e) | 288 GB | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |

## License and notices

Apache License 2.0, copied in `LICENSE`. We converted the weights to GGUF and quantized them.

## Updates

- 2026-11-27: Added Others 8 to the charts and CSV.
- 2026-11-27: Added Others 5 and Others 6 to the charts; Others renumbered from 0.
- 2026-11-25: Added the remaining Unsloth files to the table and charts.
