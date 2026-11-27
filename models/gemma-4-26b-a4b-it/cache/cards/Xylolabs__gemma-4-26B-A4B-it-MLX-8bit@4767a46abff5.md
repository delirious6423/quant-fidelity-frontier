---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- quantized
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>gemma-4-26B-A4B-it MLX 8-bit</h1>

</div>

[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) converted to [MLX](https://github.com/ml-explore/mlx) for `mlx-lm`. Text only, without the vision tower. Affine quantization, group size 64, scales and biases in float16. 8-bit: every linear weight and the token embedding (also the LM head). Norms in float16. Other unquantized tensors in float16.

## How to run

```sh
pip install mlx-lm==0.32.2

mlx_lm.generate --model Xylolabs/gemma-4-26B-A4B-it-MLX-8bit --prompt "What is the capital of Australia?"
```

## Accuracy

KL and top-1 against llama.cpp's BF16 GGUF on chat-template text (4 English and 4 Korean conversations, 8 x 511 scored tokens). Raw text (WikiText-2, chat and held-out text, 40/20/20 chunks of 512; last chart and the CSV) shows much larger KL for this model in every stack, llama.cpp's Q8_0 included.

| Quant | Size | KL | top-1 | 7-question answer check |
|---|---:|---:|---:|---:|
| MLX 4-bit | 14.47 GB | 0.153050 | 86.67% | 7/7 |
| MLX 8-bit | 26.81 GB | 0.005099 | 97.60% | 7/7 |
| BF16 | 50.51 GB | | | 7/7 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/accuracy-kl-dark.png">
  <img src="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/accuracy-kl.png" alt="KL divergence by file size, chat-template text">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/accuracy-top1-dark.png">
  <img src="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/accuracy-top1.png" alt="Top-1 agreement by file size, chat-template text">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/accuracy-rawtext-kl-dark.png">
  <img src="https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/accuracy-rawtext-kl.png" alt="Mean KL on raw text by file size">
</picture>

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.

## Updates

- 2026-11-27: First release.
