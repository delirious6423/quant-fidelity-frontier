---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - qwen3.8
  - qwen3_5
  - mtp
  - quantized
language:
  - zh
  - en
---

# Qwen3.8-27B-Quality

GDN-tiered MLX hybrid of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for [oMLX](https://github.com/jundot/omlx). Built with `mlx_lm.convert` plus re-embedded BF16 MTP. Not an oQ export.

About **26 GB** on disk. Intended as a quality-leaning local checkpoint: keep GDN gates and MTP in BF16, protect attention and residual `down_proj`, compress the bulky MLP `gate/up` path.

## Quantization

| Precision | Tensors |
| --- | --- |
| **BF16** | Entire MTP head; per-layer GDN `A_log`, `dt_bias`, `conv1d`, `in_proj_a`, `in_proj_b`, `in_proj_z` |
| **8-bit affine / gs64** | `embed_tokens`, `lm_head`, all `self_attn`, `linear_attn.in_proj_qkv`, `linear_attn.out_proj`, MLP on layers 0–7 and 56–63 |
| **6-bit affine / gs32** | Middle-layer `mlp.down_proj` |
| **4-bit affine / gs32** | Middle-layer `mlp.gate_proj` and `mlp.up_proj` |

Lightning MTP stays attached as `language_model.mtp.*` in `model-mtp.safetensors`. `mlx_lm.convert` strips `mtp.*`; those 15 tensors are copied back from the BF16 source.

## Recommended sampling

Qwen3.8 official defaults:

- `temperature=1.0`
- `top_p=0.95`
- `top_k=20`
- thinking on (can be disabled per request)

oMLX: enable Lightning MTP, `mtp_num_draft_tokens=3`.

## Use in oMLX

Place the folder under `~/.omlx/models/Qwen3.8-27B-Quality` (or Import). Enable MTP in model settings. Do not run oQ on this checkpoint if you want the BF16 MTP/gates kept.

## Use with mlx-lm

```python
from mlx_lm import load, generate

model, tokenizer = load("hanxin2000/Qwen3.8-27B-Quality")
print(generate(model, tokenizer, prompt="Hello", max_tokens=64))
```

MTP speculative decoding is an oMLX path. Stock `mlx_lm.generate` will run the quantized trunk; it may ignore the embedded MTP head.

## Files

- `model-0000n-of-00005.safetensors` — quantized language trunk
- `model-mtp.safetensors` — BF16 MTP
- `model.safetensors.index.json`
- `config.json`, tokenizer, `chat_template.jinja`

## License

Apache 2.0, same as the Qwen3.8-27B source weights.

## Acknowledgements

Base model: Qwen Team, [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Runtime: oMLX / MLX.
