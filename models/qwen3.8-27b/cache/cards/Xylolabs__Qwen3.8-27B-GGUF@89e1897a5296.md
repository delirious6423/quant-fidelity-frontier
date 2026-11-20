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
