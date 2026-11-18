---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- harness-hub/Qwen3.8-27B-pi-adapted
- TensorFold/Qwen3.8-27B-MLX-4bit
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- 4-bit
- quantized
- apple-silicon
- qwen
- qwen3.8
- pi
- agent
- lora-merged
---

# Qwen3.8-27B-pi-adapted-MLX-4bit

An MLX affine 4-bit (group size 64) build of [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted). That LoRA tunes Qwen3.8-27B for the [pi](https://github.com/earendil-works/pi) coding agent.

This repository is a drop-in replacement for [TensorFold/Qwen3.8-27B-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit). The shard layout, tensor names, config, tokenizer and chat template are the same. Only the 256 projections that the LoRA targets have different bytes.

## How it was built

1. Start from the BF16 weights of `Qwen/Qwen3.8-27B` at revision `37561816`.
2. Merge the LoRA from `harness-hub/Qwen3.8-27B-pi-adapted` at revision `e786f849` (r 32, alpha 64, scaling 2.0): `W' = W + 2.0 * B @ A`.
3. Merge targets: `gate_proj`, `up_proj` and `down_proj` in all 64 layers, and `q_proj`, `k_proj`, `v_proj` and `o_proj` in the 16 full-attention layers (3, 7, ..., 63).
4. Re-quantize each merged projection to MLX affine 4-bit, group size 64. Write the result at the same offsets in the TensorFold checkpoint.
5. Copy all other tensors unchanged from `TensorFold/Qwen3.8-27B-MLX-4bit` at revision `22d8d538`. This includes the vision tower, embeddings and linear-attention layers.

The quantizer reproduces the rounding of the MLX 0.32.4 Metal kernel exactly. The MLX CPU quantizer rounds ties to even and gives slightly different bytes. As a gate, each of the 256 target projections was first re-quantized from the unmerged BF16 weights. The result had to match the TensorFold bytes exactly, and it did for all 256 (weights, scales and biases).

## Use

This checkpoint was checked only under the TensorFold engine, in agent workloads. The format is the same as `TensorFold/Qwen3.8-27B-MLX-4bit`, so it should also load in [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). That path was not tested:

```bash
pip install mlx-vlm
python -m mlx_vlm.generate --model tozuo/Qwen3.8-27B-pi-adapted-MLX-4bit \
  --prompt "Write a bash one-liner that counts lines in all .py files." \
  --max-tokens 512
```

The adapter card recommends thinking off. The chat template enables thinking by default. To turn it off, send `chat_template_kwargs: {"enable_thinking": false}` from an OpenAI-compatible server.

The checkpoint needs about 16 GB of unified memory for the weights, plus the KV cache.

## Changes from the upstream models

As required by the Apache License 2.0, section 4(b):

- The weights of the 256 projections listed above differ from `TensorFold/Qwen3.8-27B-MLX-4bit`. They are the result of merging the harness-hub LoRA and re-quantizing.
- All other files are unchanged copies from `TensorFold/Qwen3.8-27B-MLX-4bit`, except this README.

## License

Apache License 2.0, the same license as `Qwen/Qwen3.8-27B`, `harness-hub/Qwen3.8-27B-pi-adapted` and `TensorFold/Qwen3.8-27B-MLX-4bit`. See `LICENSE`.
