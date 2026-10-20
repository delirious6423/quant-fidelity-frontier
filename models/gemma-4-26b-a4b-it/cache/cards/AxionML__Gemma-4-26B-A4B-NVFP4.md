---
pipeline_tag: image-text-to-text
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
license: apache-2.0
library_name: transformers
tags:
- AxionML
- ModelOpt
- Gemma-4
- quantized
- NVFP4
- nvfp4
- sglang
- mirror
---

# AxionML Gemma-4-26B-A4B-NVFP4

> Mirrored by [AxionML](https://huggingface.co/AxionML) for open-source serving and deployment use cases. Part of AxionML's effort to provide ready-to-serve quantized models for the community.
>
> **Quantized by NVIDIA.** The weights in this repository are an unmodified copy of [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) (revision `a19cfe00be84568a6867111c9a68c9c44fdcffe6`). All credit for the quantization belongs to NVIDIA.

This is an NVFP4-quantized version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (25.2B total parameters, 3.8B activated), quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

**About NVFP4 quantization:** NVFP4 on Blackwell couples a compact E2M1 FP4 codebook with blockwise FP8 (E4M3) scaling over 16-element micro-blocks, so that 4-bit stored values remain numerically useful for neural-network computation. The E2M1 codebook provides a small, nonuniform set of representable magnitudes up to ±6 and relies on saturating behavior rather than IEEE NaN/Inf encodings to maximize usable range per bit. Using an FP8 block scale (rather than power-of-two-only E8M0) enables fractional scales and error-minimizing scale selection. On Blackwell Tensor Cores, native FP4 multipliers exploit E2M1 simplicity while higher-precision FP32 accumulation protects dot-product accuracy.

> **Ready for commercial and non-commercial use under [Apache 2.0 (Gemma)](https://ai.google.dev/gemma/apache_2).**

## Model Summary

| | |
|:---|:---|
| **Architecture** | Mixture-of-Experts, hybrid sliding-window (1024) + global attention |
| **Total Parameters** | 25.2B |
| **Activated Parameters** | 3.8B |
| **Experts** | 128 routed, 1 shared, 8 active |
| **Layers** | 30 |
| **Vision Encoder** | ~550M parameters |
| **Context Length** | 256K tokens |
| **Checkpoint Size** | ~18.8 GB |

## Evaluation Results

| Benchmark | BF16 | NVFP4 |
|:---|:---:|:---:|
| GPQA Diamond | 80.30 | 79.90 |
| AIME 2025 | 88.95 | 90.00 |
| MMLU Pro | 85.00 | 84.80 |
| LiveCodeBench (pass@1) | 80.50 | 79.80 |
| IFBench | 77.77 | 78.10 |
| IFEval | 96.60 | 96.40 |

> Scores reported by NVIDIA for this checkpoint. Baseline: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). `temperature=1.0`, `top_p=0.95`, `max_new_tokens=131072`.

## Quantization Details

- **Quantization format:** NVFP4 experts-only (`nvfp4_experts_only` recipe) — routed expert weights and activations in NVFP4; attention, shared expert and vision tower kept in BF16
- **KV cache:** FP8
- **Calibration dataset:** [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)
- **Tool:** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) v0.43.0

## Usage

### Deploy with SGLang

```bash
python3 -m sglang.launch_server \
    --model-path AxionML/Gemma-4-26B-A4B-NVFP4 \
    --quantization modelopt_fp4 \
    --tp 1 \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4
```

### Deploy with vLLM

```bash
vllm serve AxionML/Gemma-4-26B-A4B-NVFP4 \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --enable-auto-tool-choice \
    --trust-remote-code
```

Upstream notes that vLLM currently serves this checkpoint with TP=1 only (EP is supported), using the VLLM_CUTLASS or Marlin MoE backend.

## Limitations

The base model was trained on data that may contain toxic language and societal biases. The quantized model inherits these limitations. It may generate inaccurate, biased, or offensive content. Please refer to the [original model card](https://huggingface.co/google/gemma-4-26B-A4B-it) and the [upstream quantized model card](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) for full details.

## Credits

- **Base model:** [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Quantization:** [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) by NVIDIA
- **Mirror:** [AxionML](https://huggingface.co/AxionML)
