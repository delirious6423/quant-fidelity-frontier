---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
tags:
  - mlx
  - quantized
  - qwen4-exp
---

# Qwen3.8-Flash-Next MLX 4-bit

This is a reproducible MLX conversion of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at immutable
revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`.

MLX affine Q4/group-64 for eligible language, MTP, and vision matrices; the 160-wide n-gram table uses Q4/group-32. Routers, norms, biases, convolutions, and incompatible shapes remain dense.

- Artifact payload: 97.55 GiB
- Quantized Q2 modules: 0
- Quantized Q4/group-32 modules: 128
- Quantized Q4/group-64 modules: 927
- Source: 180B parameters including 125B main, 51B n-gram embedding, and 4B MTP
- Native context: 262,144 tokens

## Runtime status

The tensor inventory, source hashes, MLX packing, fused-expert split, convolution
layout, and zero-centered RMSNorm conversion are validated by the bundled
`MERERUN_CONVERSION.json`. A Qwen4Exp-aware MLX runtime is required; do not
expect an older `mlx-lm` or `mlx-vlm` release to dispatch this new architecture.

## License

This redistribution retains the upstream **Qwen Community License 1.0** in
`LICENSE`. Review it before use. In particular, it contains attribution/display
requirements for very large commercial products and separate-license conditions
for certain commercial Model-as-a-Service and AI Work Assistant uses. The model
is not gated; downloading or using it does not remove those terms.

The upstream model card is preserved as `README.upstream.md`.
