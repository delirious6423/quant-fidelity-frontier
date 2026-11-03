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

Same tensors, shapes, dtypes and config as the template, so it is a drop-in replacement wherever the template loads
(mlx-vlm, mlx-lm, Splash) with the same size (15 GB) and speed. Only the numbers inside the 4-bit tensors differ.

- 497 Linears requantized (all 64 layers incl. `lm_head`). The embedding and the vision tower are the template's.
- GPTQ fitted to MLX's own affine format: each group's scale and bias are computed from the error-updated
  weights and rounded to bf16 before coding, so the result dequantizes exactly as written.
- Hessians from the bf16 model, one per shared input, damp 0.01, block 128.
- Calibration: 119 x 2048 tokens. 64 chunks of UltraChat train conversations (chat template applied) and 55 chunks of calibration_v5 text.

## Use

```bash
pip install -U mlx-vlm
python -m mlx_vlm.generate --model sady4850/Qwen3.8-27B-4bit-GPTQ --max-tokens 100 --prompt "Hello"
```

In [Splash](https://github.com/paperniuk/splash) use this repo id in place of `mlx-community/Qwen3.8-27B-4bit`.

## Quality vs bf16

Texts: wiki = wikitext-2 test, chat = first 300 conversations of UltraChat test as plain text.
30 chunks x 2048 tokens per text, scored on the second half of each chunk (like `llama-perplexity`).
KLD = KL(bf16 || quant) of the next-token distribution, flips = top-1 token differs from bf16.

Note that chat is close to the calibration data (UltraChat train vs test split), wiki is out of domain.

**torch** (A100), both checkpoints dequantized into the same bf16 model:

| text | model | KLD mean | KLD median | KLD p99 | flips | PPL |
|---|---|---|---|---|---|---|
| wiki | bf16 | | | | | 6.370 |
| wiki | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 0.0562 | 0.0230 | 0.440 | 9.57% | 6.486 |
| wiki | this repo (GPTQ) | **0.0369** | **0.0122** | **0.315** | **7.37%** | 6.475 |
| chat | bf16 | | | | | 8.692 |
| chat | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 0.1800 | 0.0312 | 3.233 | 13.13% | 8.109 |
| chat | this repo (GPTQ) | **0.0975** | **0.0127** | **1.743** | **9.05%** | 8.752 |

**Splash** (`teacher-logits`, int8 KV, M1 Pro), the engine's own logits against bf16 log-probs for the same rows
(every 8th scored row, KL over the bf16 top-512 tokens plus a bound for the rest, within 2% of full-vocab KL):

| text | model | KLD mean | KLD median | KLD p99 | flips | PPL |
|---|---|---|---|---|---|---|
| wiki | bf16 | | | | | 6.364 |
| wiki | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 0.0562 | 0.0221 | 0.469 | 9.58% | 6.479 |
| wiki | this repo (GPTQ) | **0.0399** | **0.0119** | **0.326** | **7.49%** | 6.458 |
| chat | bf16 | | | | | 8.717 |
| chat | `mlx-community/Qwen3.8-27B-4bit` (RTN) | 0.1906 | 0.0327 | 2.724 | 13.13% | 8.141 |
| chat | this repo (GPTQ) | **0.1002** | **0.0128** | **1.577** | **9.05%** | 8.706 |

Speed in Splash is the same as the template: median per 2048-token window 14.10 s prefill for both,
23.00 s vs 23.05 s for 127 verify cycles.

PPL alone is a poor measure of closeness here: the RTN checkpoint has a lower chat PPL than bf16 itself.
Lower KLD and fewer flips mean the outputs follow the original model more closely.

## Agent replay

Independent run by [@ghisguth](https://github.com/ghisguth) ([splash discussion #34](https://github.com/paperniuk/splash/discussions/34#discussioncomment-19226461)):
78 recorded turns of a real coding-agent session (32K to 124K context), each turn sent with the same recorded prompt,
temperature 0, Splash 1.4.0-m1 on M2 Max, compared per turn against Q8_0.

| model | same calls | same tools | args similarity | broken calls |
|---|---|---|---|---|
| `mlx-community/Qwen3.8-27B-4bit` (RTN) | 31% | 79% | 0.88 | 0 |
| this repo (GPTQ) | **44%** | **83%** | **0.92** | 0 |

Same calls = same tools with the same arguments as Q8_0. For scale, Q8_0 against itself gives 82%.
Speed was the same as RTN (cycles/s within 2%). 78 turns is a small sample, but it agrees with the KLD above.

## Other quants in Splash

Same windows and bf16 reference as above, for choosing between formats:

| model | size | wiki KLD | chat KLD | chat KLD p99 | 127 verify cycles |
|---|---|---|---|---|---|
| `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M` | 16.5 GB | 0.0144 | 0.0831 | 1.353 | 25.60 s |
| this repo (GPTQ) | 15 GB | 0.0399 | 0.1002 | 1.577 | 23.00 s |
| `unsloth/Qwen3.8-27B-GGUF:Q4_0` | 16.1 GB | 0.0325 | 0.1412 | 2.493 | 24.40 s |
| `mlx-community/Qwen3.8-27B-4bit` (RTN) | 15 GB | 0.0562 | 0.1906 | 2.724 | 23.05 s |
| `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF:IQ3_S` | 11.8 GB | 0.0573 | 0.2121 | 3.427 | 29.40 s |

If the extra 1.5 GB and ~11% slower decode (short context, less at long context) are fine, UD-Q4_K_M is closer to bf16.
At the size and speed of MLX 4-bit, this repo is the closer one.

## Limitations

Vision was not evaluated (all runs are language-only), calibration is English chat and text,
KLD windows are 2048 tokens. Task accuracy was not measured.

Raw numbers and logs in `eval/`.
