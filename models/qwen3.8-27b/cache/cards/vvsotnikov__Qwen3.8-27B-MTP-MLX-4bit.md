---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
inference: false
language:
- en
tags:
- mlx
- mlx-vlm
- qwen3_5_mtp
- qwen
- qwen3.8
- qwen3.8-27b
- mtp
- speculative-decoding
- draft-model
- 4-bit
---

# Qwen3.8-27B-MTP-MLX-4bit

> **This is not a standalone model.** It holds only the Multi-Token
> Prediction (MTP) drafter head, so it carries no token embeddings and
> no `lm_head` and it cannot generate text alone. Load it as the draft
> model beside a matching target, which supplies both.

Qwen ships this head inside
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) as 15 tensors under
the `mtp.` prefix, but the MLX converter deletes them when it builds the main
model, so the head is published separately and the runtime loads it as its
own model with `model_type: qwen3_5_mtp`.

## Use with mlx-vlm

```bash
mlx_vlm generate \
  --model vvsotnikov/Qwen3.8-27B-MLX-4bit \
  --draft-model vvsotnikov/Qwen3.8-27B-MTP-MLX-4bit \
  --prompt "Write a quicksort in Python." \
  --max-tokens 256 --temperature 0.6 --enable-thinking
```

For local weights:

```bash
mlx_vlm generate \
  --model /path/to/target-model \
  --draft-model /path/to/Qwen3.8-27B-MTP-MLX-4bit \
  --prompt "Write a quicksort in Python." \
  --max-tokens 256 --temperature 0.6 --enable-thinking
```

`--draft-kind mtp` is detected from `model_type`, so you do not need to pass
it. The drafter proposes tokens each step while the target verifies them, and
only accepted tokens reach the output, so quality follows the target rather
than the drafter.

## Model Details

- Model type: `qwen3_5_mtp`
- MTP block size: `3`
- Target architecture: Qwen3.8-27B
- Precision: MLX affine 4-bit, group size 64
- Runtime: MLX / `mlx-vlm` 0.6.8
- Format: Safetensors with MLX-compatible config and tokenizer files

## How this was produced

```bash
# 1. split the mtp.* tensors out of the base checkpoint
python -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split \
  --model Qwen/Qwen3.8-27B --output Qwen3.8-27B-MTP-MLX-bf16

# 2. quantize the result
mlx_vlm convert --hf-path Qwen3.8-27B-MTP-MLX-bf16 \
  --mlx-path Qwen3.8-27B-MTP-MLX-4bit -q --q-bits 4 --q-group-size 64
```

Round-to-nearest is the only option here, because AWQ needs a forward pass
and a drafter cannot run one on its own.

## Verification

| Check | Result |
| --- | --- |
| `model_type` | `qwen3_5_mtp` |
| `block_size` | 3 |
| Tensors | 31, being 8 quantized projections and 7 dense norms |
| RMSNorm shift | applied exactly once, checked against the source |
| Acceptance | 93.5% of drafted tokens accepted, 2.86 accepted tokens/round, over 70 rounds, 15.982 tok/s |

## Pairing rules

Use a drafter and a target that come from ONE checkpoint, because the drafter
binds to the target's embeddings at runtime, so a mismatched pair either fails
a hidden-size check or drafts badly. This release publishes these drafters, so
you can trade drafter size against acceptance:

- [`Qwen3.8-27B-MTP-MLX-bf16`](https://huggingface.co/vvsotnikov/Qwen3.8-27B-MTP-MLX-bf16)
- [`Qwen3.8-27B-MTP-MLX-8bit`](https://huggingface.co/vvsotnikov/Qwen3.8-27B-MTP-MLX-8bit)
- [`Qwen3.8-27B-MTP-MLX-4bit`](https://huggingface.co/vvsotnikov/Qwen3.8-27B-MTP-MLX-4bit)  <- this one

## License and attribution

The weights derive from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
under Apache 2.0, so the original license and its terms carry over.
Read [the license itself](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) before you use this model.
