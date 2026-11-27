---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- quantized
- qwen38
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Qwen3.8-27B MLX 4-bit</h1>

</div>

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) converted to [MLX](https://github.com/ml-explore/mlx) for `mlx-lm`, text only: the vision tower and the MTP layer are left out. Affine quantization, group size 64, scales and biases in float16. 4-bit: every linear layer, the token embedding and the LM head. Norms and the other unquantized tensors in float16, except `A_log` in bfloat16.

## How to run

```sh
pip install mlx-lm==0.32.2

mlx_lm.generate --model Xylolabs/Qwen3.8-27B-MLX-4bit --prompt "Explain how a heat pump works."
mlx_lm.chat     --model Xylolabs/Qwen3.8-27B-MLX-4bit
```

The model thinks by default; add `--chat-template-config '{"enable_thinking": false}'` to `mlx_lm.generate` to answer directly.

## Accuracy

KL and top-1 against BF16.

| Quant | Size | WikiText-2 KL | top-1 | Chat KL | top-1 | Held-out KL | top-1 | 7-question answer check |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLX 8-bit | 28.58 GB | 0.000384 | 99.19% | 0.001776 | 99.29% | 0.000244 | 99.08% | 7/7 (BF16: 7/7) |
| MLX 4-bit | 15.13 GB | 0.045196 | 90.06% | 0.060599 | 91.73% | 0.045803 | 88.90% | 7/7 (BF16: 7/7) |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/Qwen3.8-27B-MLX-4bit/resolve/main/accuracy-kl-dark.png">
  <img src="https://huggingface.co/Xylolabs/Qwen3.8-27B-MLX-4bit/resolve/main/accuracy-kl.png" alt="Mean KL divergence by size">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/Xylolabs/Qwen3.8-27B-MLX-4bit/resolve/main/accuracy-top1-dark.png">
  <img src="https://huggingface.co/Xylolabs/Qwen3.8-27B-MLX-4bit/resolve/main/accuracy-top1.png" alt="Top-1 agreement by size">
</picture>

## Which file for your hardware

| Hardware | Memory | File | Context |
|---|---|---|---:|
| Mac Studio M5 Max, 32-core GPU | 36 GB unified | `Xylolabs/Qwen3.8-27B-MLX-4bit` (4-bit, 15.1 GB) | 147K |
| Mac Studio M5 Max, 40-core GPU | 48 GB unified | `Xylolabs/Qwen3.8-27B-MLX-8bit` (8-bit, 28.6 GB) | 80K |
| Mac Studio M5 Max, 40-core GPU | 64 GB unified | `Xylolabs/Qwen3.8-27B-MLX-8bit` (8-bit, 28.6 GB) | 256K |
| Mac Studio M5 Ultra | 96 GB unified | `Xylolabs/Qwen3.8-27B-MLX-8bit` (8-bit, 28.6 GB) | 256K |
| Mac Studio M5 Max, 40-core GPU | 128 GB unified | `Xylolabs/Qwen3.8-27B-MLX-8bit` (8-bit, 28.6 GB) | 256K |
| Mac Studio M5 Ultra | 256 GB unified | `Xylolabs/Qwen3.8-27B-MLX-8bit` (8-bit, 28.6 GB) | 256K |
| Mac Studio M5 Ultra, 80-core GPU | 512 GB unified | `Xylolabs/Qwen3.8-27B-MLX-8bit` (8-bit, 28.6 GB) | 256K |

## License and notices

Apache License 2.0, copied in `LICENSE`. We converted the weights to MLX and quantized them; see `NOTICE`.

## Updates

- 2026-11-27: First release.
