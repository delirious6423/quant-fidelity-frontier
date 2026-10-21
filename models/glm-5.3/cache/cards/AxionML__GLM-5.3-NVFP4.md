---
pipeline_tag: text-generation
base_model:
- zai-org/GLM-5.3
base_model_relation: quantized
license: other
license_name: zai-model-license
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
library_name: transformers
tags:
- AxionML
- ModelOpt
- GLM-5
- quantized
- NVFP4
- nvfp4
- sglang
- mirror
---

# AxionML GLM-5.3-NVFP4

> Mirrored by [AxionML](https://huggingface.co/AxionML) for open-source serving and deployment use cases. Part of AxionML's effort to provide ready-to-serve quantized models for the community.
>
> **Quantized by RadixArk.** The weights in this repository are an unmodified copy of [RadixArk/GLM-5.3-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-NVFP4) (revision `6e389189d564d9d3e4ea7fa284fe91f136d8ae2a`). All credit for the quantization belongs to RadixArk.

This is an NVFP4-quantized version of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (753B total parameters, ~40B activated), quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

**About NVFP4 quantization:** NVFP4 on Blackwell couples a compact E2M1 FP4 codebook with blockwise FP8 (E4M3) scaling over 16-element micro-blocks, so that 4-bit stored values remain numerically useful for neural-network computation. The E2M1 codebook provides a small, nonuniform set of representable magnitudes up to ±6 and relies on saturating behavior rather than IEEE NaN/Inf encodings to maximize usable range per bit. Using an FP8 block scale (rather than power-of-two-only E8M0) enables fractional scales and error-minimizing scale selection. On Blackwell Tensor Cores, native FP4 multipliers exploit E2M1 simplicity while higher-precision FP32 accumulation protects dot-product accuracy.

> **Ready for commercial and non-commercial use under the Z.AI Model License** (MIT-style, included as `LICENSE`).

## Model Summary

| | |
|:---|:---|
| **Architecture** | Sparse MoE with sparse attention (`GlmMoeDsaForCausalLM`), IndexShare indexer |
| **Total Parameters** | 753B |
| **Activated Parameters** | ~40B |
| **Layers / Experts** | 78 layers (3 dense + 75 MoE), 256 routed experts (top-8) + 1 shared, 1 MTP layer |
| **Context Length** | 1,048,576 tokens |
| **Checkpoint Size** | ~465 GB (vs ~1,507 GB BF16) |

## Evaluation Results

| Benchmark | Protocol | NVFP4 |
|:---|:---|:---:|
| GSM8K | Full 1,319-example split | 97.42 |
| Terminal-Bench 2.1 | terminus-2 harness | 86.5 |
| DeepSWE | mini-swe-agent harness | 68.1 |
| AIME 2026 | 30 problems × 16, pass@1 | 94.17 (maj@16 100) |

> Scores reported by RadixArk for this checkpoint. Against the BF16 source under the same protocol, GSM8K matched exactly and AIME 2026 pass@1 was within run-to-run noise.

## Quantization Details

- **Quantization format:** NVFP4 W4A4 (group size 16, FP8 E4M3 block scales, static per-tensor activation scales) on the routed experts of all 75 MoE layers — 96.2% of parameters
- **Unchanged (BF16):** sparse attention incl. IndexShare indexer, shared experts, routers, the 3 dense MLP layers, norms, embeddings, `lm_head`, all MTP tensors
- **Calibration dataset:** 1,024 samples at length 512 from `cnn_dailymail` + `Nemotron-Post-Training-Dataset-v2`, max calibration
- **Tool:** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) v0.47.0.dev91

## Usage

### Deploy with SGLang

```bash
sglang serve \
    --model-path AxionML/GLM-5.3-NVFP4 \
    --tp-size 8 \
    --quantization modelopt_fp4 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6
```

Validated upstream on 8x B300. The MTP layer is kept in BF16, so EAGLE speculative decoding works. See the [SGLang GLM-5.3 cookbook](https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3).

## Limitations

The base model was trained on data that may contain toxic language and societal biases. The quantized model inherits these limitations. It may generate inaccurate, biased, or offensive content. Please refer to the [original model card](https://huggingface.co/zai-org/GLM-5.3) and the [upstream quantized model card](https://huggingface.co/RadixArk/GLM-5.3-NVFP4) for full details.

## Credits

- **Base model:** [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)
- **Quantization:** [RadixArk/GLM-5.3-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-NVFP4) by RadixArk
- **Mirror:** [AxionML](https://huggingface.co/AxionML)
