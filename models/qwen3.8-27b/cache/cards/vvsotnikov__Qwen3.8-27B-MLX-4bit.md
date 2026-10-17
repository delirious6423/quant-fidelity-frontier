---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
language:
- en
tags:
- mlx
- mlx-vlm
- qwen
- qwen3.8
- qwen3.8-27b
- 4-bit
---

# Qwen3.8-27B-MLX-4bit

This model was converted to MLX format from
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) using mlx-vlm version
**0.6.8**, and quantized to 4 bits, so it runs on Apple
silicon through `mlx-vlm`. Refer to the
[original model card](https://huggingface.co/Qwen/Qwen3.8-27B) for more details
on the model itself.

Pair it with
[`Qwen3.8-27B-MTP-MLX-4bit`](https://huggingface.co/vvsotnikov/Qwen3.8-27B-MTP-MLX-4bit)
to get speculative decoding, because Qwen carries a Multi-Token Prediction
head that the MLX conversion moves into a separate repository.

## Use with mlx-vlm

```bash
pip install -U mlx-vlm

mlx_vlm generate \
  --model vvsotnikov/Qwen3.8-27B-MLX-4bit \
  --draft-model vvsotnikov/Qwen3.8-27B-MTP-MLX-4bit \
  --prompt "Write a quicksort in Python." \
  --max-tokens 256 --temperature 0.6 --enable-thinking
```

Drop `--draft-model` to decode without the speculator.

## How this was produced

```bash
mlx_vlm convert --hf-path Qwen/Qwen3.8-27B \
  --mlx-path Qwen3.8-27B-MLX-4bit -q --q-bits 4 --q-group-size 64
```

The quantization is uniform affine at 4 bits with group size
64, and it covers 498 modules
of the language model including `embed_tokens`, `lm_head` and the
`linear_attn` projections. The vision tower stays dense in bfloat16, since
the converter skips multimodal modules by default.

## Verification

Every artifact in this release passed an automated check before it was
published, and nothing became public until its remote checksums matched the
local build.

| Check | Result |
| --- | --- |
| Quantization | `{bits: 4, group_size: 64, mode: affine}` |
| Quantized modules | 498 of 2180 tensors |
| Vision tower | dense, 333 tensors |
| MTP tensors | 0, moved to the drafter repository |
| Acceptance with the 4-bit drafter | 93.5% of drafted tokens accepted, 2.86 accepted tokens/round, over 70 rounds, 15.982 tok/s |

## License and attribution

The weights derive from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
under Apache 2.0, so the original license and its terms carry over.
Read [the license itself](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) before you use this model,
and refer to the upstream model card for the model's capabilities and limits.
