---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - lfm2.5
  - moe
  - qad
  - liquid
---

# LFM2.5-24B-A2B-QAD-MLX-4bit

**I converted [Liquid's QAD Q4_0 GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) (original model: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)) to MLX by copying each block's scale over directly instead of requantizing, so the weights are identical to the GGUF's.**

This is a direct conversion of [`LFM2.5-24B-A2B-QAD-Q4_0.gguf`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF), the Quantization-Aware Distillation build that Liquid AI trained specifically to recover accuracy lost to Q4_0. A normal MLX quant starts from bf16 and rounds again, which throws that tuning away. Instead, every Q4_0 block (32 weights, value = d·(q−8)) is rewritten as an MLX affine 4-bit group of 32 with scale = d and bias = −8d, so the dequantized weights are identical to the GGUF's (verified: max difference 0.0). The one exception is the token embedding, which is Q6_K in the GGUF and is stored here as MLX 8-bit; that adds error far smaller than Q6_K's own. Checked against llama.cpp on the original file, next-token top-5 rankings matched on every test prompt with probabilities within about 0.003. The cost is size: MLX stores a scale and a bias per group, so it's 5 bits/weight vs Q4_0's 4.5, about 15.1 GB instead of 13.6 GB. Runs with `mlx-lm` and oMLX (about 58 tok/s decode on an M5). Licensed under Liquid AI's LFM Open License v1.0 (see `LICENSE`).
