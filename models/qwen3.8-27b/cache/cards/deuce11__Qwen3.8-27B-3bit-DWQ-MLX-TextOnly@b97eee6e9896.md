---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- qwen
- text-generation
- text-only
- quantized
---

# Qwen3.8-27B 3-bit DWQ, text only

**This file stops early.** On 2026-10-20, Splash with thinking off stopped three of four prose prompts mid-sentence. One stop was mid-word (`doesn`). Lengths were 231, 222, 254, and 290 tokens. The 222-token reply reached a period. The same four prompts finished on a sentence with [lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) (391, 648, 899, and 908 tokens). The KL table further down is a short Tulu slice. It is separate from that stop.

The current file is [deuce11/Qwen3.8-27B-3bit-DWQ-BF16-MLX-TextOnly](https://huggingface.co/deuce11/Qwen3.8-27B-3bit-DWQ-BF16-MLX-TextOnly). That one snaps the official BF16 weights to affine 3-bit, group 64, then learns scales and biases against the BF16 teacher. The saved step is 511. On 2026-10-21 the same four prompts all ended on a sentence (418, 674, 731, and 907 tokens).

This model was converted to MLX format from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) using mlx-lm version **0.32.2**.

Affine 3-bit, group size 64. The three safetensors shards are 11.77 GB. Vision tensors are not included. Refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B) for more details on the base model.

## Quantization

DWQ learned the scales and biases. The packed integers are the round-to-nearest grid of [`mlx-community/Qwen3.8-27B-8bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit), dequantized and requantized to affine 3-bit, group 64.

| | |
|---|---|
| Bits, group, mode | 3, 64, affine |
| Teacher | `mlx-community/Qwen3.8-27B-8bit` |
| Data | `allenai/tulu-3-sft-mixture` |
| Train / valid rows | 1024 / 32 |
| Max sequence, batch, learning rate | 512, 2, 1e-5 |
| Steps | 512 |
| What DWQ updates | scales and biases |

Validation loss was 0.291 at step 0, 0.124 at step 199, 0.101 at step 399, and 0.108 at step 511. The saved weights are that last step.

## Evaluation

Sixteen held-out sequences from `allenai/tulu-3-sft-mixture`, split `train[2000:2016]`, max length 512. That slice is outside the 1056 rows used for DWQ. KL is token-weighted against official BF16. Perplexity is the quantized model's own negative log-likelihood.

| Checkpoint | KL | top-1 | perplexity |
|---|---:|---:|---:|
| this model | 0.206 | 0.834 | 7.59 |
| round-to-nearest 3-bit from the 8-bit teacher | 0.540 | 0.771 | 9.83 |
| [lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly](https://huggingface.co/lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly) | 0.639 | 0.762 | 10.79 |

The same tokens scored against the 8-bit teacher move KL by less than 0.006.

## Use with mlx-lm

```bash
pip install -U mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("deuce11/Qwen3.8-27B-3bit-DWQ-MLX-TextOnly")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello."}],
    tokenize=False,
    add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=True))
```

`config.json` still has `language_model_only: false` and architecture `Qwen3_5ForConditionalGeneration`. The vision tower is not in these weights. Load the repo as text only. With Splash, pass `--language-only`.

The base model is Apache 2.0.
