---
language:
- en
license: apache-2.0
library_name: mlx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- geer
- mlx
- qwen3_5
- coding
---

# Geer Qwen 3.8 27B (4/8-bit MLX)

This is Geer's independently produced mixed 4/8-bit MLX conversion of
Qwen3.8-27B for 48 GB Apple-Silicon Macs. Geer does not offer or claim a 32 GB
Qwen profile; those machines continue to use Ornith. This is not a new
foundation model.

Large feed-forward matrices use affine 4-bit quantization. Attention and
linear-attention projections use 8-bit quantization; the vision tower,
embeddings, output head, norms, and native MTP tensors remain unquantized. The
build uses group size 64 and upstream revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

Upstream benchmark results describe the BF16 checkpoint, not this conversion.
This artifact retained 100% of the qualified 6-bit aggregate pass rate in the
initial 128 GB host canaries. Its 48 GB support profile uses a 128K context
ceiling. That tier remains provisional until exercised on a real 48 GB Mac.
