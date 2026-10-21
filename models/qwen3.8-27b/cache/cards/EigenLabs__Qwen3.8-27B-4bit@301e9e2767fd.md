---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
- mlx
- mlx-vlm
- vision
---

# EigenLabs/Qwen3.8-27B-4bit

This revision is a complete MLX 4-bit Qwen3.8 vision-language model that accepts image, video, and text inputs and generates text.

The **1,847 `language_model.*` tensors are bit-for-bit identical** to [`EigenLabs/Qwen3.8-27B-4bit@eda45ab47f465d08d6558f0353a2346e2eb9d5b3`](https://huggingface.co/EigenLabs/Qwen3.8-27B-4bit/tree/eda45ab47f465d08d6558f0353a2346e2eb9d5b3). The 333 `vision_tower.*` tensors and multimodal processor assets come from [`mlx-community/Qwen3.8-27B-4bit@3e6447f082e89cc7f0bc6e5441afd38dfce760ff`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/3e6447f082e89cc7f0bc6e5441afd38dfce760ff), converted from the official [`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) source. Tensor dtype, shape, and raw bytes were checked before publication.

The immutable challenge revision [`eda45ab47f465d08d6558f0353a2346e2eb9d5b3`](https://huggingface.co/EigenLabs/Qwen3.8-27B-4bit/tree/eda45ab47f465d08d6558f0353a2346e2eb9d5b3) remains available and is text-only because it has no vision tower or processor assets. Multi-token prediction (MTP) is intentionally **not embedded** in this repository; the MTP artifact remains separate.

## mlx-vlm

```bash
pip install -U mlx-vlm
mlx_vlm.generate \
  --model EigenLabs/Qwen3.8-27B-4bit \
  --image /path/to/image.jpg \
  --prompt "Describe this image." \
  --max-tokens 256
```

Pin the immutable Hub commit shown in this model's history for reproducible deployment.

## Darkbloom

This is the Hugging Face source artifact consumed by Darkbloom's in-process MLX VLM runtime. Darkbloom providers deploy only catalog-approved, immutable model revisions and verify their manifests before loading. Updating this repository alone does not register the model in the Darkbloom catalog or activate provider traffic.

After a Darkbloom catalog release pins the immutable Hub commit and publishes its verified manifest, a provider can discover and download that release:

```bash
darkbloom models catalog
darkbloom models download <catalog-model-id>
darkbloom start --foreground
```

`<catalog-model-id>` is assigned by the Darkbloom catalog; it is not the Hugging Face repository name.
