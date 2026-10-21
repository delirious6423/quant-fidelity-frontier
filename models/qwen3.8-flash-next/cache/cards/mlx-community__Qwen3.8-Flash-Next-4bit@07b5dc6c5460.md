---
language: en
library_name: mlx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- mlx
- qwen4_exp
---

# Qwen3.8-Flash-Next-4bit (MLX)

4-bit MLX conversion of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
converted with `mlx-vlm` `main` (post-#2032) at commit `d1bd74ed`, group size 32.

Group size 32 is required so the PLE n-gram embedding dimensions can be quantized.

## Usage

```sh
pip install git+https://github.com/Blaizzy/mlx-vlm
mlx_vlm.generate --model mlx-community/Qwen3.8-Flash-Next-4bit \
  --prompt "Explain sparse attention in one paragraph." --max-tokens 256
```

## Why this conversion exists

`Qwen4ExpRMSNorm` applies `1 + w` to norm gains that the released checkpoint stores
centered at zero, matching upstream `Qwen4ExpTextRMSNorm`. Several MLX conversions
published before [#2032](https://github.com/Blaizzy/mlx-vlm/pull/2032) landed were made
with a converter that folded that `+1` into the saved weights, so loading them applies
the offset twice and generation degenerates into noise
([#2041](https://github.com/Blaizzy/mlx-vlm/issues/2041)).

## Verification

Checked against the bf16 source after conversion:

| check | result |
|---|---|
| source integrity | 131/131 shards, tensor bytes byte-exact vs `index total_size` |
| norm gain center (source vs converted) | `+0.2216` vs `+0.2216`, delta `+0.00000` |
| norm tensors bit-identical | 148 / 148 |
| `sanitize()` idempotence | passes over 480 1-D tensors |
| generation | coherent output at `--temperature 0.0` |

A gain center near `+1.15` instead of `+0.22` is the signature of the double-shifted
conversions described above.
