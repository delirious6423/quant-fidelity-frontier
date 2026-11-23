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

Mean KL divergence from BF16 (lower is better) and top-1 agreement with BF16, measured with llama.cpp; the charts show WikiText-2.

| Quant | Size | WikiText-2 KL | top-1 | Chat KL | top-1 | Held-out KL | top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Others 1 Q8_0 | 29.116 GB | 0.000893 | 98.789% | 0.002929 | 98.873% | 0.001024 | 98.863% |
| Unsloth Q8_0 | 29.047 GB | 0.000867 | 98.672% | 0.002304 | 99.059% | 0.001067 | 98.794% |
| Q8_0 | 29.047 GB | 0.000867 | 98.672% | 0.002304 | 99.059% | 0.001067 | 98.794% |
| Others 4 Q8_0 | 29.047 GB | 0.000867 | 98.672% | 0.002304 | 99.059% | 0.001067 | 98.794% |
| Others 5 Q8_0 | 28.596 GB | 0.000896 | 98.721% | 0.003599 | 98.931% | 0.001053 | 98.833% |
| Unsloth UD-Q8_K_L | 28.046 GB | 0.000786 | 98.657% | 0.002376 | 98.980% | 0.001109 | 98.627% |
| Unsloth UD-Q6_K_XL | 25.299 GB | 0.001032 | 98.564% | 0.002039 | 98.588% | 0.001826 | 98.265% |
| Others 0 Q6_K | 25.005 GB | 0.001356 | 98.431% | 0.003389 | 98.578% | 0.002123 | 98.471% |
| Others 1 Q6_K | 23.861 GB | 0.001741 | 98.098% | 0.003791 | 98.412% | 0.002540 | 98.039% |
| Q6_K | 22.431 GB | 0.002161 | 97.951% | 0.004358 | 98.245% | 0.003333 | 97.824% |
| Others 4 Q6_K | 22.431 GB | 0.002757 | 97.608% | 0.005857 | 97.853% | 0.003879 | 97.520% |
| Unsloth UD-Q6_K | 21.980 GB | 0.001773 | 98.137% | 0.003402 | 98.265% | 0.003245 | 98.010% |
| Others 1 Q5_K_M | 20.924 GB | 0.003850 | 97.221% | 0.007609 | 97.941% | 0.006614 | 96.971% |
| Others 0 Q5_K_M | 20.233 GB | 0.003761 | 97.279% | 0.008246 | 97.647% | 0.006789 | 97.147% |
| Unsloth UD-Q5_K_M | 19.871 GB | 0.003068 | 97.412% | 0.006759 | 97.706% | 0.006460 | 97.039% |
| Others 1 Q5_K_S | 19.567 GB | 0.004350 | 96.897% | 0.007404 | 97.922% | 0.007175 | 96.922% |
| Q5_K_M | 19.536 GB | 0.004578 | 96.902% | 0.008445 | 97.598% | 0.008101 | 96.794% |
| Others 5 Q4_K_M | 19.404 GB | 0.013713 | 94.485% | 0.019312 | 95.412% | 0.024574 | 94.069% |
| Unsloth UD-Q5_K_S | 18.765 GB | 0.003995 | 97.201% | 0.007980 | 97.490% | 0.008646 | 96.588% |
| Others 1 Q4_K_M | 17.442 GB | 0.012294 | 95.304% | 0.019341 | 95.637% | 0.023030 | 94.755% |
| Others 0 Q4_K_M | 17.121 GB | 0.009966 | 95.789% | 0.018466 | 96.235% | 0.019308 | 95.343% |
| Others 4 Q4_K_M | 16.811 GB | 0.020912 | 93.505% | 0.038925 | 93.882% | 0.035779 | 93.441% |
| Unsloth UD-Q4_K_M | 16.564 GB | 0.007576 | 96.201% | 0.020176 | 95.618% | 0.018755 | 95.431% |
| Q4_K_M | 16.564 GB | 0.010464 | 95.574% | 0.019746 | 95.745% | 0.019950 | 95.412% |
| Others 0 IQ4_XS | 16.513 GB | 0.011175 | 95.426% | 0.025758 | 95.216% | 0.021946 | 95.333% |
| Others 1 IQ4_XS | 15.476 GB | 0.017114 | 94.574% | 0.048382 | 93.922% | 0.030669 | 94.206% |
| Unsloth UD-Q4_K_S | 15.358 GB | 0.010193 | 95.618% | 0.034123 | 94.588% | 0.030494 | 94.265% |
| Q4_K_S | 15.358 GB | 0.015845 | 94.549% | 0.030877 | 94.990% | 0.030020 | 94.324% |
| IQ4_XS-imat | 15.309 GB | 0.017617 | 94.157% | 0.032862 | 94.598% | 0.032699 | 93.775% |
| Unsloth UD-IQ4_XS | 14.353 GB | 0.016367 | 94.176% | 0.039900 | 94.127% | 0.036723 | 93.157% |
| Others 0 IQ3_S | 13.838 GB | 0.036191 | 91.475% | 0.054390 | 92.882% | 0.070031 | 91.608% |
| Q3_K_M | 13.501 GB | 0.051699 | 90.142% | 0.105065 | 90.559% | 0.089243 | 89.275% |
| Others 1 Q3_K_M | 13.404 GB | 0.055829 | 89.926% | 0.082506 | 91.549% | 0.095603 | 89.235% |
| Unsloth UD-Q3_K_XL | 13.146 GB | 0.026235 | 92.554% | 0.049744 | 93.863% | 0.065165 | 91.657% |
| Others 8 IQ4_XS | 13.083 GB | 0.046557 | 91.054% | 0.043219 | 93.647% | 0.073566 | 90.206% |
| Others 1 IQ3_XS | 12.800 GB | 0.057374 | 89.667% | 0.086202 | 91.010% | 0.098663 | 89.245% |
| IQ3_M | 12.768 GB | 0.059505 | 89.637% | 0.087631 | 91.343% | 0.100871 | 89.118% |
| Unsloth UD-IQ3_S | 12.041 GB | 0.036043 | 91.770% | 0.080079 | 91.471% | 0.106168 | 89.118% |
| Others 8 IQ3_S | 11.035 GB | 0.100455 | 86.544% | 0.102267 | 89.892% | 0.170342 | 85.216% |
| Unsloth UD-IQ3_XXS | 10.935 GB | 0.054159 | 89.627% | 0.101860 | 90.078% | 0.143572 | 86.402% |
| Q2_K | 10.865 GB | 0.144555 | 83.809% | 0.214223 | 86.059% | 0.228253 | 83.127% |
| Others 1 Q2_K | 10.822 GB | 0.161675 | 83.760% | 0.211475 | 85.402% | 0.230526 | 82.598% |
| Unsloth UD-Q2_K_XL | 9.949 GB | 0.086977 | 87.289% | 0.166114 | 87.000% | 0.160427 | 85.824% |
| Others 8 IQ3_XXS | 9.919 GB | 0.181147 | 82.368% | 0.157283 | 86.931% | 0.270854 | 81.167% |
| Others 0 IQ2_XS | 9.890 GB | 0.168705 | 82.078% | 0.240257 | 84.873% | 0.256814 | 81.725% |
| Others 8 IQ2_XXS | 8.844 GB | 0.281744 | 77.868% | 0.240894 | 83.922% | 0.402898 | 76.471% |
| Unsloth UD-IQ2_S | 8.423 GB | 0.127511 | 84.931% | 0.206836 | 84.941% | 0.238166 | 82.578% |
| Unsloth UD-IQ2_XXS | 7.206 GB | 0.197141 | 81.167% | 0.234754 | 83.422% | 0.358335 | 78.186% |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-mean-kl-dark.png">
  <img src="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-mean-kl.png" alt="Mean KL divergence by quant file size">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-top1-dark.png">
  <img src="https://huggingface.co/Xylolabs/Qwen3.8-27B-GGUF/resolve/main/accuracy-top1.png" alt="Top-1 accuracy by quant file size">
</picture>

## Which file for your hardware

A file fits when its size plus the KV cache at that context (f16) plus 2 GB for the runtime is under 90% of the memory.

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
| One B200 (about 180 GB usable in a DGX B200) | 192 GB | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| Mac Studio M5 Ultra | 256 GB unified | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| Mac Studio M5 Ultra | 512 GB unified | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |
| One B300 (HBM3e; usable a little less) | 288 GB | `Qwen3.8-27B-Q8_0.gguf` (29.0 GB) | 256K |

## License and notices

Apache License 2.0, copied in `LICENSE` (unchanged from the Qwen3.8-27B release). We converted the weights to GGUF and quantized them. `NOTICE` names the source model and its copyright holder. Xylolabs is not affiliated with or endorsed by Qwen or Alibaba Cloud.

## Updates

- 2026-11-22: Removed `Qwen3.8-27B-IQ4_XS.gguf` (15.42 GB): the smaller Q4_K_S and IQ4_XS-imat score better.
- 2026-11-22: Accuracy table and charts rebuilt, now with Unsloth's files and files from five other providers.
- 2026-11-20: First release in ten quants with the vision projectors.
