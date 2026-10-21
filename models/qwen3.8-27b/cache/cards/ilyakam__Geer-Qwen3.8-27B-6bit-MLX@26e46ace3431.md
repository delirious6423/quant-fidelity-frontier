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

# Geer Qwen 3.8 27B (6-bit MLX)

This is Geer's independently produced, protected 6-bit MLX conversion of
Qwen3.8-27B for local agentic coding on Apple Silicon. It preserves the
multimodal tower and native MTP head. It is not a new foundation model.

The build uses upstream revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, affine quantization with group
size 64, and pinned MLX conversion dependencies. The vision tower, embeddings,
output head, norms, and MTP tensors remain unquantized. The included source and
build manifests record exact file hashes.

Upstream benchmark results describe the BF16 checkpoint, not this conversion.
Geer uses this artifact on 64 GB and larger Macs with a 128K context ceiling at
64 GB and Qwen's native 256K ceiling at 96/128 GB. Native MTP is calibrated per
installation and remains user-configurable. The 48 and 64 GB profiles remain
provisional until exercised on real hosts of those sizes.
