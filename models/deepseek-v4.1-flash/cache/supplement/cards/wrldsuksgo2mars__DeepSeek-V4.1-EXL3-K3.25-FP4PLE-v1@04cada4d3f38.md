---
license: mit
base_model:
- wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1
tags:
- exl3
- nvfp4
- deepseek_v41
---
# DeepSeek V4.1 EXL3 K3.25 with NVFP4 PLE

PLE-only variant of [wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1](https://huggingface.co/wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1/tree/cfd4ca1d1934a8e81dd2d7515598d4ce288e8b88).
The routed EXL3 expert quantization (40 main and 3 dSpark blocks) is unchanged.
Only the two PLE embedding tables at layers 1 and 14 were converted from the
source FP8/E8M0 representation to weight-only NVFP4 using CPU conversion.

## Storage

One standard indexed safetensors checkpoint, 52 globally numbered shards.
Shards 1–48 are byte-identical to the base repository and reused server-side.
Shards 49/50 hold PLE 1 scales/weights; shards 51/52 hold PLE 14 scales/weights.
Each PLE has packed E2M1 `weight` [rows,128] uint8 (even element low nibble),
`weight_scale` [rows,16] FP8 E4M3, and scalar FP32 `weight_scale_2`.
The original logical shape is [rows,256], with contiguous 16-element blocks.
Decode as E2M1 times the block scale times the global scale. Scales are stored
row-major, not GPU-kernel-swizzled. Config `ds41rt_ple_quantization` and
`quantize_config.json` metadata describe this representation explicitly.
No other non-routed tensors were changed. Source code/license/tokenizer assets
are retained from the base model; `README.source.md` is the original source card.

## Recipe and checks

Global scale = table weight absolute maximum / 2688. Block scales use each
16-value absolute maximum divided by (6 * global scale), clamp to [2^-9,448],
and E4M3 round-to-nearest-even. Values use E2M1 round-to-nearest-even.
No activation calibration, expert requantization, GPU work, or whole-model
validation/replay was performed. One actual row per table was checked against
an independent arithmetic reference for exact packed bytes/scales, with
reconstruction error recorded in the conversion report. This is not a quality
evaluation. Inference-engine integration must support this PLE representation
alongside the base EXL3/native formats; standard safetensors is not a claim of
stock-engine execution support.

Conversion implementation and reproducible process:
https://github.com/tpurtell/ds41rt/tree/main/quantization
