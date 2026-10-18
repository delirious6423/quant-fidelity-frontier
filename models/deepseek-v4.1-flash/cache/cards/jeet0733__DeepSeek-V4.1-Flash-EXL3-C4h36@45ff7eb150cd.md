---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
  - exl3
  - exllamav3
  - quantized
  - moe
  - vllm
  - deepseek
  - dgx-spark
---

# DeepSeek-V4.1-Flash · EXL3 · usage-allocated 2.9 bpw (C4h36)

A 196 GiB EXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
for two NVIDIA DGX Spark (GB10) boxes. Same size, shard layout and per-layer bit budget as the
2.9 bpw MiaAI pack it is derived from, but each layer's routed-expert bits are re-split **per
expert by routing usage**: busy experts at K=4, typical ones at K=3, rare ones at K=2 (layers 18-22
uniform K=2 as in the base map; layers 36-39 uniform K=3 so the built-in DSpark drafter keeps
its acceptance). Attention, shared experts, head, indexer, Engram and MTP tensors are the base
pack's. The Engram embedding tables are not in this repo (same as the base pack); the serving
recipe streams them from the original checkpoint.

| | base 2.9 bpw pack, recipe as shipped (stock kernel, DSpark k=3) | this pack on the `usage-allocated-pack` branch (k=5 + adaptive) |
|---|---:|---:|
| WikiText-2 perplexity vs the native fp8 model, same bf16 reference path | +4.05 % | **+2.46 %** |
| WikiText-2 64x512 served perplexity, same harness, same two Sparks | 6.311 | **6.197-6.210 (-1.5 ... -1.7 %)** |
| 1 stream, short prompt | 49.5 tok/s | **81.0** |
| 2 streams, aggregate | 57 tok/s | **111** |
| decode battery, hard / easy / code | 25-30 / 55-56 / 39-40 | 24 / 51 / 54 |

Pack-only ablation (the base pack served on the same branch and settings): 82.5 / 93 tok/s, so the
two packs are within noise single-stream and this one is ~19 % faster at two streams; the rest of
the gap is the serving stack (GB10 MoE kernel, launch fusion, k=5 + adaptive verification).

Why it works: expert error weight tracks routing usage (Spearman 0.98) and every bit removed
multiplies an expert's quantization error by about 4, so at a fixed byte budget the busy experts
deserve more bits and the rare ones fewer. Allocation is by greedy usage-weighted error on
262K calibration tokens; quantization is EXL3 (QTIP bitshift trellis, mul1 codebook) with the
full-token Hessian per expert. Method, ablations and evaluation: the paper
`docs/13-usage-allocated-exl3-pack-paper.md` in the [quant](https://github.com/jeet0733/quant) repository.

## Serving

The pack needs the mixed-K serving path (ragged per-expert trellis loader, single-launch mixed-K
GB10 MoE kernel), available on the `usage-allocated-pack` branch of
[DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/jeet0733/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)
(fork of the MiaAI recipe). In its `.env`:

```bash
MODEL_HOST=/path/to/this/pack
WORKER_MODEL_DIR=/path/to/this/pack
KMAP_HOST=$PWD/files/exl3_k_map_c4h36.json   # per-expert K map (also shipped here as exl3_k_map_c4h36.json)
DSV41_EXL3_MOE_X=2,12,8
DSPARK_TOKENS=5
DSPARK_ADAPTIVE=1
```

Uniform-K loaders (exllamav3, the stock recipe) cannot read the per-expert layers.

## Files

`exp-NN-LNN-model.safetensors`: the re-quantized routed experts of layer NN (per-expert K).
`model-*.safetensors`: the base pack's shards with the replaced experts removed. `exl3_k_map_c4h36.json`:
the K of every tensor and the 384 per-expert Ks of every layer.

## Attribution and license

Derived from the MiaAI Lab EXL3 2.9 bpw pack (MIT; its card is kept as `README.miaai-base.md`),
itself converted with [exllamav3](https://github.com/turboderp-org/exllamav3). DeepSeek-V4.1-Flash
is released under the MIT license. This pack is MIT.
