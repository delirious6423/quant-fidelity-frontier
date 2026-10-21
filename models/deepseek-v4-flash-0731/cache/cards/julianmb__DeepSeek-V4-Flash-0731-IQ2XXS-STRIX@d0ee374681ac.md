---
language:
  - en
  - zh
license: mit
pipeline_tag: text-generation
tags:
  - deepseek
  - moe
  - amd
  - strix-halo
  - gguf
  - iq2_xxs
  - q2_k
  - q8_0
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
---

# DeepSeek-V4-Flash-0731-IQ2XXS-STRIX

Quantized DeepSeek V4 Flash (0731) GGUF for AMD Strix Halo (`gfx1151`).

## Details

- **Base model**: DeepSeek V4 Flash 0731 (284B MoE)
- **Quantization source**: tekosML (`DeepSeek-V4-Flash-0731-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix`)
- **File size**: 86.72 GB
- **Architecture**: 43 routed layers, 256 experts/layer (6 active), 4-stream hyper-connections

## Quant Recipe

| Tensor group | Type |
| :--- | :--- |
| Attention projections | Q8_0 |
| Shared experts | Q8_0 |
| Output head | Q8_0 |
| Token embedding | F16 |
| Routed gate/up experts | IQ2_XXS |
| Routed down experts | Q2_K |

## Recommended Runtime

Run with the [julianmb/ds4fa](https://github.com/julianmb/ds4fa) engine on 128 GB Strix Halo:

```sh
DS4_ROCM_STREAM_MODEL_CACHE_GB=48 ./ds4 -m DeepSeek-V4-Flash-0731-IQ2XXS-STRIX.gguf -c 512 \
  --ssd-streaming --ssd-streaming-cache-experts 32GB \
  -p "What is the capital of France?" --think --tokens 60
```
