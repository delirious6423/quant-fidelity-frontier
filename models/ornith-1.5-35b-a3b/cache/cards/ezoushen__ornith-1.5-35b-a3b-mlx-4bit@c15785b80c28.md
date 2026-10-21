---
license: mit
language:
  - en
library_name: mlx
pipeline_tag: image-text-to-text
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
  - mlx
  - splash
  - apple-silicon
  - mac
  - macos
  - on-device
  - ornith
  - qwen3.6
  - moe
  - mixture-of-experts
  - vision
  - multimodal
  - 4-bit
---

# Ornith 1.5 35B-A3B, MLX 4-bit with vision

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) as an MLX affine 4-bit
checkpoint that includes the vision tower and loads on [Splash](https://github.com/incoai/splash) 1.1.0.
We serve it with vision on our 1.1.0-based fork; on upstream 1.1.0 we have tested its language model only.

```bash
splash serve --model ezoushen/ornith-1.5-35b-a3b-mlx-4bit
```

The language model is [ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit),
byte for byte. That repository has no vision tower, and Splash 1.1.0 stops on its config
([reported upstream](https://github.com/incoai/splash/issues/38#issuecomment-5865447638)). This one adds both.

Independent redistribution, not endorsed by Ornith AI or Inco AI. MIT, as upstream.

## What differs from ornith-ai's MLX 4-bit repo

| | |
| --- | --- |
| Language model | unchanged: 1,757 tensors in four shards, affine 4-bit, group size 64 (router and shared-expert gate at 8 bits) |
| Vision tower | added: `model-vision.safetensors`, 333 BF16 tensors (0.89 GB), Ornith's tower, recovered from [our Splash package](https://huggingface.co/ezoushen/ornith-1.5-35b-a3b-splash) by an exact inversion |
| `config.json` | adds `rope_parameters.rope_type: "default"` and a `vision_config` |
| Also added | `preprocessor_config.json` and a shard index that lists the vision tensors |
| `chat_template.jinja` | renders a `system` message after the first as a system turn instead of raising, the one-line change incoai's Qwen3.6 Splash package makes (since 2026-09-30) |

Total size is 20.4 GB.

## Which Ornith repo to use

- **Just running Ornith on a Mac with Splash:** [ezoushen/ornith-1.5-35b-a3b-splash](https://huggingface.co/ezoushen/ornith-1.5-35b-a3b-splash),
  the packed Splash package. It has the benchmarks and caveats.
- **Serving Ornith beside a fine-tune of it:** this repo. With the
  [shared-expert-weights](https://github.com/ezoushen/splash/tree/shared-expert-weights) Splash fork, a fine-tune such as
  [Umpire MLX 4-bit](https://huggingface.co/ezoushen/umpire-35b-a3b-mlx-4bit), served beside it from the same
  `SPLASH_WEIGHT_CACHE`, maps this model's prepared experts, embedding, head and draft instead of loading a second copy.

## Licence

MIT, from Ornith AI (`LICENSE`). `NOTICE` lists every change.
