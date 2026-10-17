---
language: en
library_name: mlx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- mlx
- qwen4_exp
- mtp
- speculative-decoding
---

# Qwen3.8-Flash-Next-4bit-mtp (MLX)

[mlx-community/Qwen3.8-Flash-Next-4bit](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit) plus the model's own
multi-token-prediction (MTP) head from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), so it can
draft for itself (self-speculative decoding). The 4-bit conversion leaves out the MTP layer; this repository adds it back.

## What is in it

- The 22 shards of `mlx-community/Qwen3.8-Flash-Next-4bit`, unchanged (group size 32, 4-bit affine; routers and shared-expert
  gates at 8-bit, group 64).
- One extra shard, `model-mtp-00001-of-00001.safetensors` (54 tensors, 1.61 GiB): the `mtp.*` head, fetched from
  `Qwen/Qwen3.8-Flash-Next`. The routed-expert `gate_up_proj` is split into gate and up halves to match the base layout, and
  the routed experts and the hyper-connection mixing weights are 4-bit affine, group 32; the other head tensors stay bf16.
- `config.json` declares the head (`mtp_num_hidden_layers: 1`), and `model.safetensors.index.json` lists the new shard.
- `mtp_graft_manifest.json` records every head tensor's upstream shard, dtype, shape, byte count and sha256.
