---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: mlx
pipeline_tag: text-generation
license: other
license_name: qwen-community-license
license_link: LICENSE
tags:
- qwen4_exp
- quantized
- 4-bit
- mlx
- moe
---

# Qwen3.8-Flash-Next 4-bit (MLX affine)

4-bit quantized build of the text tower of
[**Qwen/Qwen3.8-Flash-Next**](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(`qwen4_exp`), packed for the MLX-style runtime used by
[lisa](https://lisa-ml.com/).

## Quantization

- **Affine, 4-bit** (MLX `mx.quantize`): packed `u32` weights + one `bf16`
  scale/bias pair per group of weights along the input dim.
- **Group size 32**. Groups of 32 weights share one scale/bias pair.
- Group 32 is required by lisa's fused Flash-Next decode kernels, which hardcode
  `GS=32`.

Smaller groups = more scale/bias overhead, lower quantization error. The group
size is fixed by the checkpoint and must be honored when loading.

## Contents

```
config.json                  qwen4_exp, language_model_only, quantization {group_size 32, bits 4}
model-00001.safetensors …    trunk (17 shards, ~4.9 GB each)
mtp/model.safetensors        multi-token-prediction draft head
ngram.safetensors            merged n-gram table (4-bit gs32)
tokenizer.json …             tokenizer
```

## Conversion

Built from the public bf16 original with `convert_qwen38_flash_next.py --gs 32`
— streams the ~360 GB checkpoint shard by shard (download → quantize → delete).
Renames/transforms: `language_model.model.*` prefix, MoE `switch_mlp` repack,
conv1d `[C,1,K] → [C,K,1]`, RMSNorm `+1` fold.
