---
license: mit
library_name: mlx
pipeline_tag: image-text-to-text
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
- ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit
tags:
- mlx
- qwen3_5_moe
- multimodal
- vision
- image-text-to-text
---

# Ornith 1.5 35B A3B MLX 4-bit + BF16 Vision

This is a complete MLX vision-language checkpoint assembled from the official
Ornith 1.5 releases. It combines the official 4-bit MLX language checkpoint
with the official BF16 vision tower that is absent from the MLX 4-bit release.

## Sources

- Language checkpoint: `ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`
  - pinned revision: `19504d912fa8fc7622bf6b1de3db5d5d890b1f02`
- Vision and processor metadata: `ornith-ai/Ornith-1.5-35B-A3B`
  - pinned revision: `10fbf86fed7ecee4a061f8b499a618f46001cac1`
  - source shard: `model-00001-of-00016.safetensors`

The four language-model shards remain byte-identical to the official MLX
4-bit release. All 333 `model.visual.*` tensors were streamed from the
official BF16 source into `ornith15_vision_bf16.safetensors`, renamed to the
MLX `vision_tower.*` layout. The Conv3d patch-embedding tensor was transposed
from PyTorch `[out,in,t,h,w]` to MLX `[out,t,h,w,in]` layout.

`VISION_SIDECAR.json` records the exact source revisions, source-shard hash,
derived sidecar hash, tensor count and layout conversion. `CHECKSUMS.sha256`
records the immutable runtime-file hashes.

## Precision and runtime

- Language body: affine 4-bit, group size 64
- Vision tower: BF16
- Architecture: Qwen3.5/Qwen3.6-compatible MoE VLM
- Recommended AI2Apps Runtime: `ai2apps/runtime-omlx >= 1.5.6`

The mixed precision is intentional: it preserves visual quality while keeping
the language model compact. AI2Apps can run the checkpoint fully resident or
convert its routed experts to the Direct Cached-MoE store for lower memory
use. Full-resident execution is the default when memory allows.

## License and attribution

The upstream Ornith 1.5 model is released under the MIT License. This mirror
preserves the upstream model terms and records the transformation provenance;
it does not claim authorship of the model weights.

- Ornith: https://ornith.ai/
- Official model: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B
- Official MLX 4-bit model: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit

