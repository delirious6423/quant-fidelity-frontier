---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- gptq
- 4-bit
---

# sady4850/Qwen3.8-27B-4bit-GPTQ

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) in exactly the layout of
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (MLX affine 4-bit, group size 64), with the
round-to-nearest codes, scales and biases of every quantized language-model Linear replaced by GPTQ.

Same tensors, shapes, dtypes and config as the template, so it loads wherever the template does
(mlx-vlm, mlx-lm, Splash) with the same size and speed. Only the numbers inside the 4-bit tensors differ.

- 497 Linears requantized (all 64 layers incl. `lm_head`). The embedding and the vision tower are the template's.
- GPTQ fitted to MLX's own affine format: each group's scale and bias are computed from the error-updated
  weights and rounded to bf16 before coding, so the result dequantizes exactly as written.
- Hessians from the bf16 model, one per shared input, damp 0.01, block 128.
- Calibration: 119 x 2048 tokens, ultrachat (chat template applied) + calibration_v5.

## Use

```bash
pip install -U mlx-vlm
python -m mlx_vlm.generate --model sady4850/Qwen3.8-27B-4bit-GPTQ --max-tokens 100 --prompt "Hello"
```

## Quality vs bf16

30 chunks x 2048 tokens per text, scored on the second half of each chunk (like `llama-perplexity`),
in torch with both checkpoints dequantized into the same bf16 model. wiki = wikitext-2 test,
chat = held-out chat transcripts. Flips = top-1 token differs from bf16.

| text | model | PPL | KLD mean | KLD median | KLD p99 | flips |
|---|---|---|---|---|---|---|
| wiki | bf16 | 6.3699 | | | | |
| wiki | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 6.4857 | 0.0562 | 0.0230 | 0.4399 | 9.57% |
| wiki | this repo (GPTQ) | 6.4750 | 0.0369 | 0.0122 | 0.3151 | 7.37% |
| chat | bf16 | 8.6916 | | | | |
| chat | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 8.1087 | 0.1800 | 0.0312 | 3.2333 | 13.13% |
| chat | this repo (GPTQ) | 8.7519 | 0.0975 | 0.0127 | 1.7426 | 9.05% |

Raw numbers and logs in `eval/`.
