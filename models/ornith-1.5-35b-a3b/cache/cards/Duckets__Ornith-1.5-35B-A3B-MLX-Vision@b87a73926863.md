---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- vision
- moe
- apple-silicon
---

<img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6a4cf30b13cd5622a79c9b39b39454b7/5I18jygKdbISHc9BUU5XK.png" alt="Built with Ornith" width="420">

# Ornith 1.5 35B-A3B — MLX 4-bit (Vision)

A full **vision-language** MLX port of [Ornith 1.5 35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) by **Ornith AI** — converted from the upstream BF16 checkpoint with the complete vision tower intact (333 `vision_tower` tensors: patch embed, blocks, merger). Many MLX ports of this model are text-only conversions that silently drop the vision tower; this one is not.

## Conversion

- Tool: `mlx_vlm.convert -q` (mlx-vlm), affine quantization, 4-bit, group size 64
- 4.649 bits per weight; 20.43 GB on disk
- Architecture `qwen3_5_moe` (`Qwen3_5MoeForConditionalGeneration`): 40 layers, hidden 2048, 256 routed experts (~3B active), 262K context
- Includes processor + video preprocessor configs, chat template, and tokenizer

## Verification (2026-10-20, Apple M5 Max)

Loaded in LM Studio and tested through its API with a freshly generated synthetic image containing a random code the model had never seen: the model read the code back **exactly**. Text generation smoke-tested in the same session. (Verification protocol: a novel random string rendered as an image, exact-match read — a blind model cannot pass it by guessing.)

## Credits

- **Ornith AI** — creators of Ornith 1.5 35B-A3B. All credit for the model is theirs.
- Conversion and verification by **Juno** for **Duckets**.

## License

Model weights: MIT — see the upstream [LICENSE](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE) (© Ornith AI).
