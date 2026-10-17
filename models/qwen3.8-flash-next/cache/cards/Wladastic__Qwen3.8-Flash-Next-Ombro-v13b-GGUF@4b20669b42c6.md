---
base_model: Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - qwen
  - qwen3.8
  - moe
  - quantized
  - ombro
---

# Qwen3.8 Flash Next — Ombro V13b GGUF

Mixed-precision GGUF quantization of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), successor of
[Ombro V9](https://huggingface.co/Wladastic/Qwen3.8-Flash-Next-Ombro-v9-GGUF).

| Quant | Size | Effective BPW |
| --- | ---: | ---: |
| [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 111.3 GB | 5.03 |
| [Ombro V9](https://huggingface.co/Wladastic/Qwen3.8-Flash-Next-Ombro-v9-GGUF) | 114.5 GB | 5.18 |
| **Ombro V13b** | **118.9 GB** | **5.38** |
| [UD-Q5_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 158.3 GB | 7.16 |

## Compared with UD-Q4_K_XL and Ombro V9

### Routed experts

Per layer, `gate` / `up` / `down`. Bold marks a change from V9.

| Layer | UD-Q4_K_XL | Ombro V9 | Ombro V13b |
| --- | --- | --- | --- |
| all other 39 layers | Q4_K / Q4_K / Q5_1 | Q4_K / Q4_K / Q5_1 | Q4_K / Q4_K / Q5_1 |
| 2 | Q5_K / Q5_K / Q8_0 | Q6_K / Q6_K / Q8_0 | Q6_K / Q6_K / Q8_0 |
| 4, 30, 47 | Q4_K / Q4_K / Q8_0 | Q6_K / Q6_K / Q8_0 | Q6_K / Q6_K / Q8_0 |
| 17 | Q4_K / Q4_K / Q5_1 | Q6_K / Q6_K / Q5_1 | Q6_K / Q6_K / Q5_1 |
| 33 | Q4_K / Q4_K / Q5_1 | IQ4_XS / Q6_K / Q5_1 | **Q6_K / Q6_K / Q8_0** |
| 35 | Q4_K / Q4_K / Q5_1 | Q6_K / Q6_K / Q8_0 | Q6_K / Q6_K / Q8_0 |
| 38 | Q4_K / Q4_K / Q5_1 | Q4_K / Q6_K / Q5_1 | **Q6_K / Q6_K / Q8_0** |
| 46 | Q4_K / Q4_K / Q8_0 | Q4_K / Q5_K / Q8_0 | **Q6_K / Q6_K / Q8_0** |

### Everything else

| Part | UD-Q4_K_XL | Ombro V9 | Ombro V13b |
| --- | --- | --- | --- |
| Attention and SSM projections | Q8_0 | Q8_0 | **BF16** |
| Hyper-connection projections | Q8_0 | Q8_0 | **BF16** |
| PLE projections | Q8_0 | Q8_0 | **BF16** |
| Norms, router, SSM and inject control tensors | F32 | F32 | F32 |
| Shared experts | Q8_0 | Q8_0 | Q8_0 |
| Token embedding / output | Q8_0 | Q8_0 | Q8_0 |
| Per-layer token embedding (PLE) table | IQ4_NL | IQ4_NL | IQ4_NL |
| **Routed experts** | **77.0 GB** | **80.2 GB** | **81.5 GB** |
| **Total** | **111.3 GB** | **114.5 GB** | **118.9 GB** |

### Format limits behind these choices

- The `down` experts have 640-wide rows, which K-quants (256-element blocks) cannot
  store. Only 32-element block types fit, and between `Q5_1` (6.0 bpw) and `Q8_0`
  (8.5 bpw) there is nothing, so `down` is either `Q5_1` or `Q8_0`.
- The PLE table has 160-wide rows and is stored as `IQ4_NL`. It is 28.8 GB, about a
  quarter of the model, but each token reads only a few of its rows. With `mmap` most of
  it can stay on disk.

The complete per-tensor override map is in
[`quantization-recipe.txt`](quantization-recipe.txt).
