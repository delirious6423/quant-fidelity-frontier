---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
  - gguf
  - deepseek
  - metal
---

# DeepSeek V4.1 Flash 1017 GGUF for DwarfStar

Calibrated Q2 weights and the matching vision encoder for
[DwarfStar](https://github.com/antirez/ds4), converted from
`deepseek-ai/DeepSeek-V4.1-Flash-1017` with the V4.1 recipe in `gguf-tools`.

| Download target | File | File size | Main weights |
| --- | --- | ---: | ---: |
| `ds41f-1017-q2` | `DeepSeek-V4.1-Flash-1017-Q2.gguf` | 340.60 GiB | 151.77 GiB |
| `ds41f-1017-vision` | `DeepSeek-V4.1-Flash-1017-Vision.gguf` | 0.90 GiB | Separate vision encoder |

The language GGUF includes 188.83 GiB of native FP8 Engram tables. DwarfStar
reads the needed rows directly from disk, so keep the GGUF on a fast local SSD.

## Download and Run

The download targets and the 1017 vision encoder need
[audreyt/ds4](https://github.com/audreyt/ds4): it pairs a vision encoder with
the language GGUF by source revision, while upstream accepts only the original
V4.1 encoder.

```sh
./download_model.sh ds41f-1017-q2
./download_model.sh ds41f-1017-vision
./ds4 -m gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --vision gguf/DeepSeek-V4.1-Flash-1017-Vision.gguf --ssd-streaming --ctx 32768
```

This encoder does not load with the original V4.1 Flash GGUF, and the original
encoder does not load with this one.

On a 128 GB M5 Max with `--ssd-streaming`, `ds4-bench` measured 202-229 t/s
prefill and about 16 t/s generation from 4K to 16K context.

## Quantization

IQ2_XXS routed gate/up experts and Q2_K down experts; Q8 attention projections,
shared experts and output; F16/F32 tensors elsewhere; native Engram data
unchanged. Routed experts use a 32,768-token activation imatrix collected with
the DwarfStar Metal collector on a weight-energy bootstrap GGUF; it activated
93.84% of the 15,360 layer-experts, and the rest keep the weight-energy
fallback. The GGUF was quantized from the original safetensors, not requantized.

`deepseek41_validate_gguf.py --payload --imatrix` passed: complete layout, every
non-expert tensor, sampled routed experts and Engram rows match the source.
The encoder is copied byte for byte from the same checkpoint.

Source: `deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`. See the included upstream MIT license.

| File | SHA-256 |
| --- | --- |
| `DeepSeek-V4.1-Flash-1017-Q2.gguf` | `753ba9c55ed56e0ca0bf8c87876367349027452629f8f4975cdb09d50189ac07` |
| `DeepSeek-V4.1-Flash-1017-Vision.gguf` | `51e4525b67cdaf1f34c569118b268e06466233e9d5dc13442fc65170dbb3fd21` |
