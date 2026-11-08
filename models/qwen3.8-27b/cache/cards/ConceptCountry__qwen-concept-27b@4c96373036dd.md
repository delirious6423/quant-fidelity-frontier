---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - weight-editing
  - reasoning
  - concept-country
---

# Qwen Concept 27B

Qwen3.8-27B with nine down-projections rewritten by weight-space programming.
No gradient steps, no training data: the edit is derived from the model's own
activations and compiled into nine weight matrices.

The model thinks less and scores higher. Against the unmodified base, on a
pre-registered criterion: **GPQA Diamond +4.3 [+1.8, +7.1]**, **LiveCodeBench
v6 +13.0**, with **thinking tokens down 27-54%**.

The contribution is the control, not the edit. A norm-matched random edit —
same layers, same magnitude, different direction — does nothing, and its
confidence intervals straddle zero on every benchmark. That is what separates
"this direction means something" from "perturbation helps".

Paper, data and the code that generates every number:
<https://concept.country/research/weight-space-programming>

## Formats

| folder | format | size | notes |
|---|---|---|---|
| `bf16/` | bfloat16 | ~52 GB | full precision; every reported number comes from these weights |
| `fp8/` | FP8_BLOCK | ~30 GB | matches Qwen's official FP8 layout |
| `nvfp4a16/` | NVFP4 W4A16 | ~20 GB | NVFP4 weights, BF16 activations; Blackwell-class hardware |
| `gguf/` | Q8_0 | ~27 GB | llama.cpp; **the exact file the benchmarks ran on** |
| `gguf/` | Q4_K_M | ~16 GB | llama.cpp; quantized one hop from bf16, not from the Q8_0 |

The quantizations are deployment conveniences, not separately evaluated
artifacts. Every number above is from bf16 weights served at Q8_0. Nothing here
claims a 4-bit conversion preserves the effect — that needs its own evaluation
and has not had one.

Each folder carries a `SHA256SUMS` file.

## Also available

Mirrored on the Concept Country registry, which speaks the HF Hub API:

```sh
HF_ENDPOINT=https://models.concept.country hf download ConceptCountry/qwen-concept-27b
```

## Status

Experimental, under evaluation.

## License

Apache-2.0, inherited from Qwen3.8-27B (Alibaba Cloud).
