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

- **Files:** quants Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_K_S, IQ4_XS-imat, IQ4_XS, Q3_K_M, IQ3_M and Q2_K, and the vision projector (F16 and Q8_0).
- **Q5_K_M:** KL 0.0047 from BF16, top-1 95.44 %.

## Files

- `Qwen3.8-27B-<QUANT>.gguf` is one file per quant, with the tokenizer and chat template inside.
- `mmproj-Qwen3.8-27B-F16.gguf` (0.93 GB) and `mmproj-Qwen3.8-27B-Q8_0.gguf` (0.63 GB) are the vision projector, for use with any quant.

| Quant | Size | Mean KL vs BF16 | Same top-1 vs BF16 |
|---|---|---|---|
| Q8_0 | 29.05 GB | 0.0009 | 97.82% |
| Q6_K | 22.43 GB | 0.0023 | 96.83% |
| Q5_K_M | 19.54 GB | 0.0047 | 95.44% |
| Q4_K_M | 16.56 GB | 0.0097 | 93.46% |
| Q4_K_S | 15.36 GB | 0.0161 | 92.12% |
| IQ4_XS-imat | 15.31 GB | 0.0195 | 91.65% |
| IQ4_XS | 15.42 GB | 0.0250 | 90.38% |
| Q3_K_M | 13.50 GB | 0.0545 | 86.75% |
| IQ3_M | 12.77 GB | 0.0614 | 85.95% |
| Q2_K | 10.86 GB | 0.1725 | 79.21% |

How we measured:

- **KL and top-1:** against our BF16 pass in transformers, on 4,094 positions of one English and one Korean text (2,048 tokens each).

![Mean KL and same top-1 against BF16 by file size](kl.png)

### Compared with Unsloth's files

| File | Size | KL wikitext-2 | KL chat | KL held-out | Same top-1 wikitext-2 | Same top-1 chat | Same top-1 held-out |
|---|---|---|---|---|---|---|---|
| Unsloth Q8_0 | 29.05 GB | 0.0009 | 0.0023 | 0.0011 | 98.67% | 99.06% | 98.79% |
| Unsloth UD-Q8_K_L | 28.05 GB | 0.0008 | 0.0024 | 0.0011 | 98.66% | 98.98% | 98.63% |
| Unsloth UD-Q6_K_XL | 25.30 GB | 0.0010 | 0.0020 | 0.0018 | 98.56% | 98.59% | 98.27% |
| Unsloth UD-Q6_K_L | 24.18 GB | 0.0012 | 0.0027 | 0.0021 | 98.64% | 98.59% | 98.28% |
| Unsloth UD-Q6_K_M | 23.09 GB | 0.0014 | 0.0029 | 0.0025 | 98.40% | 98.37% | 98.23% |
| Xylolabs Q6_K | 22.43 GB | 0.0022 | 0.0044 | 0.0033 | 97.95% | 98.25% | 97.82% |
| Unsloth UD-Q6_K | 21.98 GB | 0.0018 | 0.0034 | 0.0032 | 98.14% | 98.27% | 98.01% |
| Unsloth UD-Q5_K_XL | 20.98 GB | 0.0024 | 0.0058 | 0.0048 | 97.88% | 98.00% | 97.72% |
| Unsloth UD-Q5_K_M | 19.87 GB | 0.0031 | 0.0068 | 0.0065 | 97.41% | 97.71% | 97.04% |
| Xylolabs Q5_K_M | 19.54 GB | 0.0046 | 0.0084 | 0.0081 | 96.90% | 97.60% | 96.79% |
| Unsloth UD-Q5_K_S | 18.76 GB | 0.0040 | 0.0080 | 0.0086 | 97.20% | 97.49% | 96.59% |
| Unsloth UD-Q4_K_XL | 17.66 GB | 0.0056 | 0.0154 | 0.0130 | 96.61% | 96.32% | 96.22% |
| Unsloth Q4_1 | 17.54 GB | 0.0151 | 0.0235 | 0.0266 | 94.61% | 95.48% | 94.04% |
| Unsloth UD-Q4_K_M | 16.56 GB | 0.0076 | 0.0202 | 0.0188 | 96.20% | 95.62% | 95.43% |
| Xylolabs Q4_K_M | 16.56 GB | 0.0105 | 0.0197 | 0.0199 | 95.57% | 95.75% | 95.41% |
| Unsloth Q4_0 | 16.06 GB | 0.0256 | 0.0417 | 0.0430 | 93.05% | 93.87% | 93.05% |
| Xylolabs IQ4_XS | 15.42 GB | 0.0228 | 0.0392 | 0.0414 | 93.09% | 94.01% | 93.28% |
| Unsloth UD-Q4_K_S | 15.36 GB | 0.0102 | 0.0341 | 0.0305 | 95.62% | 94.59% | 94.27% |
| Xylolabs Q4_K_S | 15.36 GB | 0.0158 | 0.0309 | 0.0300 | 94.55% | 94.99% | 94.32% |
| Xylolabs IQ4_XS-imat | 15.31 GB | 0.0176 | 0.0329 | 0.0327 | 94.16% | 94.60% | 93.78% |
| Unsloth UD-IQ4_XS | 14.35 GB | 0.0164 | 0.0399 | 0.0367 | 94.18% | 94.13% | 93.16% |
| Xylolabs Q3_K_M | 13.50 GB | 0.0517 | 0.1051 | 0.0892 | 90.14% | 90.56% | 89.28% |
| Unsloth UD-Q3_K_XL | 13.15 GB | 0.0262 | 0.0497 | 0.0652 | 92.55% | 93.86% | 91.66% |
| Xylolabs IQ3_M | 12.77 GB | 0.0595 | 0.0876 | 0.1009 | 89.64% | 91.34% | 89.12% |
| Unsloth UD-IQ3_S | 12.04 GB | 0.0360 | 0.0801 | 0.1062 | 91.77% | 91.47% | 89.12% |
| Unsloth UD-IQ3_XXS | 10.93 GB | 0.0542 | 0.1019 | 0.1436 | 89.63% | 90.08% | 86.40% |
| Xylolabs Q2_K | 10.86 GB | 0.1446 | 0.2142 | 0.2283 | 83.81% | 86.06% | 83.13% |
| Unsloth UD-Q2_K_XL | 9.95 GB | 0.0870 | 0.1661 | 0.1604 | 87.29% | 87.00% | 85.82% |

All were measured with the same build, texts and BF16 base: wikitext-2, chat-formatted text and a held-out text, on Unsloth's own files.

## Tensor recipe

| Tensor class | Format |
|---|---|
| Weight matrices (attention, SSM and MLP projections, token embedding) | the quant type of the file; `output.weight` is Q6_K |
| Norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, q/k norms (360 tensors, 10.7 MB) | F32 |

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

## How these were made

1. Convert `Qwen/Qwen3.8-27B` to a BF16 GGUF with `convert_hf_to_gguf.py`, and the vision projector with `--mmproj`.
2. Build the importance matrix with `llama-imatrix` on 60 chunks of an English and Korean calibration text.
3. Quantize each type: `llama-quantize --imatrix qwen38-27b.imatrix <bf16 gguf> <out> <QUANT>`.
4. Q4_K_M and Q4_K_S use a per-tensor type mix solved from measured KL per tensor and type (`MIX-Q4_K_M.args`, `MIX-Q4_K_S.args` in this repository), applied with `--tensor-type-file`, and a chat-text importance matrix.

## License and notices

- **License:** Apache License 2.0, copied in `LICENSE` (unchanged from the Qwen3.8-27B release).
- **What we changed:** we converted the weights to GGUF and quantized them.
- **Notices:** `NOTICE` names the source model and its copyright holder.
- Xylolabs is not affiliated with or endorsed by Qwen or Alibaba Cloud.
