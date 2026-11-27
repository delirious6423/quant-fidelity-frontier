---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - tfq3
  - tensorfold
  - glm
  - glm-5.3
  - moe
  - 4bpw
  - data-free
---

# GLM-5.3-Flash-TFQ3-4bpw

A 4 bpw [TFQ3](https://github.com/ashhart/TFQ) quantization of Z.ai's official
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) FP8 release.

All 288 routed experts in MoE layers 3–44 are quantized to 4 bits. Everything else (attention, shared experts,
dense layers, router, embeddings, `lm_head`, MTP layer, vision tower) is kept exactly as in the FP8 source.

TFQ3 is **data-free**: no calibration text is used.

> **Stock model.** This is Z.ai's original model with its default safety behaviour. It is not an uncensored
> or abliterated build.

## How it was made

| | |
|---|---|
| Source | `zai-org/GLM-5.3-Flash` FP8 (e4m3, 128×128 block scales), revision `46a050be` |
| Quantizer | [`ashhart/TFQ`](https://github.com/ashhart/TFQ) commit `c584f4a` |
| Command | `tfq3 <source> --family glm --glm-mode uniform4 --keep-nonmoe-bf16` |
| Quantized | `gate_proj`, `up_proj`, `down_proj` of all 288 routed experts in layers 3–44 (36,288 matrices) |
| Bits | 4.001 bpw for every expert (no mixed 3/4/5-bit allocation) |
| Calibration data | none |
| Expert files | 42 × `layer_XX.tffpw` (3.376 GiB each, 141.8 GiB total) + `pw_v3_tables.bin` |

`tfq3-inspect` reports every layer `OK`: TFPW v3, 288 experts at 4 bits, encoder SNR 23.33 dB.

## Quality

All numbers below are our own measurements against the FP8 source, on the same text.

### Output fidelity (KL divergence)

Measured with exllamav3's `eval/model_diff.py`: 25 windows × 2,048 tokens of wikitext-2 test, layer-by-layer
streaming. Model A is the FP8 source with every routed-expert matrix replaced by its TFQ3 decode (all 36,288
matrices swapped). Model B is the FP8 source.

| Metric | GLM-5.3-Flash-TFQ3-4bpw |
|---|---|
| KL divergence (FP8 → TFQ3) | **0.0579** |
| KL divergence (TFQ3 → FP8) | 0.0608 |
| Per-token KL, median / p90 | 0.0119 / 0.1416 |
| Top-1 token agreement | **92.37%** |
| Perplexity, TFQ3 / FP8 | 3.245 / 3.169 (**+2.4%**) |

For reference, an EXL3 4.0 bpw (mcg) quantization we made of a GLM-5.3-Flash FP8 checkpoint scored
**KL 0.0540, top-1 92.61%, perplexity +2.3%** with the same tool, text and settings. That EXL3 quant came
from a different FP8 source (`dealignai/GLM-5.3-Flash-UNCENSORED-FP8`), so treat the comparison as approximate.
On this measure, data-free TFQ3 comes within about 7% of a calibrated EXL3 quant.

### Weight error

Relative Frobenius error against the FP8 weights, on 10 layers × 8 experts × 6 random 128×128 tiles per
matrix:

| | `gate_proj` / `up_proj` | `down_proj` |
|---|---|---|
| GLM-5.3-Flash-TFQ3-4bpw | 0.0681 | 0.0682 |

The figure is flat across depth: 0.0679 to 0.0682 on every sampled layer.

## Running it

The `.tffpw` expert files need a TFQ3-enabled [TensorFold](https://github.com/ashhart/TensorFold) build.

**This repository contains the routed experts only.** The non-expert weights (attention, shared experts, dense
layers, router, embeddings, `lm_head`, MTP layer) are not included. Take them from
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) revision `46a050be`. The TFQ3 GLM
loaders read non-expert weights as BF16, MLX 8-bit or a `nonmoe.tffpw` shard, so convert the FP8 tensors to one
of those formats first.

- **Apple Silicon (Metal):** TensorFold's native runtime supports TFQ3 GLM-5.3-Flash.
- **NVIDIA CUDA:** supported since TensorFold PR [#4986](https://github.com/ashhart/TensorFold/pull/4986)
  (merged, in v1.3.6). It runs on one GPU, and the model needs roughly 170 GB of GPU memory. A multi-GPU (TP)
  path for TFQ3 GLM does not exist yet.

**Not yet tested:** live serving, tool calling and agent tasks. The quality section covers only output fidelity
against the FP8 source.

## Credits

- **Base model:** Z.ai, [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MIT).
- **Quantization format and converter:** [TFQ](https://github.com/ashhart/TFQ) by ashhart (Apache-2.0).
- **Runtime:** [TensorFold](https://github.com/ashhart/TensorFold).
- **Evaluation:** [exllamav3](https://github.com/turboderp-org/exllamav3) `eval/model_diff.py` by turboderp (MIT).

## License

The weights inherit the upstream [GLM-5.3-Flash MIT license](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE).
