---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama-cpp
- quantized
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>gemma-4-26B-A4B-it GGUF</h1>

</div>

GGUF quants of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) for llama.cpp on CUDA, Metal or CPU, without the vision tower.

## Files

| File | Size |
|---|---:|
| `gemma-4-26B-A4B-it-Q8_0.gguf` | 26.86 GB |
| `gemma-4-26B-A4B-it-Q6_K.gguf` | 23.17 GB |
| `gemma-4-26B-A4B-it-Q5_K_M.gguf` | 19.88 GB |
| `gemma-4-26B-A4B-it-Q4_K_M.gguf` | 17.72 GB |
| `gemma-4-26B-A4B-it-Q4_K_S.gguf` | 16.44 GB |
| `gemma-4-26B-A4B-it-IQ4_XS.gguf` | 14.97 GB |
| `gemma-4-26B-A4B-it-Q3_K_M.gguf` | 14.39 GB |
| `gemma-4-26B-A4B-it-Q2_K.gguf` | 11.87 GB |

## How to run

Use llama.cpp `caea914e9d05153aa32b96a846dca543e7cb16be` or newer.

```sh
./build/bin/llama-server -m gemma-4-26B-A4B-it-Q4_K_M.gguf --jinja -ngl 99 -c 32768
```

## Accuracy

KL and top-1 against llama.cpp's BF16 on chat-template text (4 English and 4 Korean conversations, 8 x 511 scored tokens). Raw text (WikiText-2, chat and held-out text, 40/20/20 chunks of 512; last chart and the CSV) shows much larger KL for this model in every stack, llama.cpp's Q8_0 included.

| Quant | Size | KL | top-1 | 7-question answer check |
|---|---:|---:|---:|---:|
| BF16 | 50.51 GB | | | 7/7 |
| Unsloth Q8_0 | 26.86 GB | 0.007752 | 96.99% | 7/7 |
| Q8_0 | 26.86 GB | 0.007752 | 96.99% | 7/7 |
| Unsloth UD-Q6_K | 23.17 GB | 0.008254 | 96.77% | 7/7 |
| Q6_K | 23.17 GB | 0.008201 | 97.06% | 7/7 |
| Unsloth UD-Q5_K_M | 21.15 GB | 0.011020 | 96.89% | 7/7 |
| Q5_K_M | 19.88 GB | 0.012776 | 96.67% | 7/7 |
| Q4_K_M | 17.72 GB | 0.020625 | 95.57% | 7/7 |
| Unsloth UD-Q4_K_M | 16.95 GB | 0.022593 | 95.30% | 7/7 |
| Unsloth UD-Q4_K_S | 16.49 GB | 0.021818 | 95.35% | 7/7 |
| Q4_K_S | 16.44 GB | 0.021641 | 95.38% | 7/7 |
| IQ4_XS | 14.97 GB | 0.030095 | 94.15% | 7/7 |
| Q3_K_M | 14.39 GB | 0.061478 | 92.49% | 7/7 |
| Unsloth UD-IQ4_XS | 13.60 GB | 0.060876 | 92.64% | 7/7 |
| Unsloth UD-Q3_K_M | 12.73 GB | 0.087108 | 91.36% | 7/7 |
| Q2_K | 11.87 GB | 0.169775 | 86.84% | 7/7 |
| Unsloth UD-Q2_K_XL | 10.55 GB | 0.189802 | 86.33% | 7/7 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-GGUF/resolve/main/accuracy-kl-dark.png">
  <img src="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-GGUF/resolve/main/accuracy-kl.png" alt="KL divergence by quant file size, chat-template text">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-GGUF/resolve/main/accuracy-top1-dark.png">
  <img src="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-GGUF/resolve/main/accuracy-top1.png" alt="Top-1 agreement by quant file size, chat-template text">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-GGUF/resolve/main/accuracy-rawtext-kl-dark.png">
  <img src="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-GGUF/resolve/main/accuracy-rawtext-kl.png" alt="Mean KL on raw text by quant file size">
</picture>

## License and notices

Apache License 2.0. See `LICENSE` and `NOTICE`.

## Updates

- 2026-11-27: Added Q4_K_S, IQ4_XS, Q3_K_M and Q2_K.
- 2026-11-27: First release.
