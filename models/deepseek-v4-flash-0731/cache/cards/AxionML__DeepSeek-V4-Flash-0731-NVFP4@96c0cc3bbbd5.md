---
pipeline_tag: text-generation
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
license: mit
library_name: transformers
tags:
- AxionML
- ModelOpt
- DeepSeek-V4
- quantized
- NVFP4
- nvfp4
- sglang
- mirror
---

# AxionML DeepSeek-V4-Flash-0731-NVFP4

> Mirrored by [AxionML](https://huggingface.co/AxionML) for open-source serving and deployment use cases. Part of AxionML's effort to provide ready-to-serve quantized models for the community.
>
> **Quantized by NVIDIA.** The weights in this repository are an unmodified copy of [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4) (revision `f1caa71142bd0be02f728c79f75042ac1e461579`). All credit for the quantization belongs to NVIDIA.

This is an NVFP4-quantized version of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (304B total parameters, 13B activated), quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

**About NVFP4 quantization:** NVFP4 on Blackwell couples a compact E2M1 FP4 codebook with blockwise FP8 (E4M3) scaling over 16-element micro-blocks, so that 4-bit stored values remain numerically useful for neural-network computation. The E2M1 codebook provides a small, nonuniform set of representable magnitudes up to ±6 and relies on saturating behavior rather than IEEE NaN/Inf encodings to maximize usable range per bit. Using an FP8 block scale (rather than power-of-two-only E8M0) enables fractional scales and error-minimizing scale selection. On Blackwell Tensor Cores, native FP4 multipliers exploit E2M1 simplicity while higher-precision FP32 accumulation protects dot-product accuracy.

> **Ready for commercial and non-commercial use under [MIT](https://opensource.org/license/mit).**

## Model Summary

| | |
|:---|:---|
| **Architecture** | MoE with hybrid attention (Compressed Sparse + Heavily Compressed Attention), Manifold-Constrained Hyper-Connections |
| **Total Parameters** | 304B |
| **Activated Parameters** | 13B |
| **Speculative Decoding** | DSpark heads included (unquantized) |
| **Reasoning Effort** | `low` / `high` / `max` |
| **Context Length** | 1M tokens |
| **Checkpoint Size** | ~176 GB |

## Evaluation Results

| Benchmark | MXFP4 (source) | NVFP4 |
|:---|:---:|:---:|
| GPQA Diamond | 91.5 | 91.5 |
| AA-LCR | 72.1 | 71.8 |
| τ²-Bench Telecom | 98.7 | 97.9 |
| SciCode | 51.7 | 52.1 |
| IFBench | 75.8 | 75.5 |
| Terminal-Bench v2.1 | 74.7 | 73.7 |
| GDPval (rubric) | 93.0 | 93.2 |

> Scores reported by NVIDIA for this checkpoint. Baseline: [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). `temperature=1.0`, `top_p=1.0`, max reasoning effort.

## Quantization Details

- **Quantization format:** routed MoE experts in NVFP4 (weights and activations); source MXFP4 expert weights are bit-cast losslessly to NVFP4 and only the block scales are rewritten; DSpark heads carried through unquantized
- **Calibration dataset:** [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) (activation amax)
- **Tool:** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) v0.46.0 ([recipe](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/deepseek/deepseek_v4))

## Usage

### Deploy with SGLang

```bash
sglang serve \
    --model-path AxionML/DeepSeek-V4-Flash-0731-NVFP4 \
    --tp-size 8 \
    --kv-cache-dtype fp8_e4m3 \
    --moe-runner-backend flashinfer_trtllm_routed \
    --chunked-prefill-size 4096 \
    --swa-full-tokens-ratio 0.1 \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --trust-remote-code
```

### Deploy with vLLM

```bash
vllm serve AxionML/DeepSeek-V4-Flash-0731-NVFP4 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 393216 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --attention_config.use_fp4_indexer_cache=True \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --trust-remote-code
```

DSpark heads are preserved, but speculative decoding was not validated upstream for this NVFP4 checkpoint.

## Limitations

The base model was trained on data that may contain toxic language and societal biases. The quantized model inherits these limitations. It may generate inaccurate, biased, or offensive content. Please refer to the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) and the [upstream quantized model card](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4) for full details.

## Credits

- **Base model:** [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- **Quantization:** [nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4) by NVIDIA
- **Mirror:** [AxionML](https://huggingface.co/AxionML)
