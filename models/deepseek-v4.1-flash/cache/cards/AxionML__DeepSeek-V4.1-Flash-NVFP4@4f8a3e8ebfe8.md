---
pipeline_tag: image-text-to-text
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash
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

# AxionML DeepSeek-V4.1-Flash-NVFP4

> Mirrored by [AxionML](https://huggingface.co/AxionML) for open-source serving and deployment use cases. Part of AxionML's effort to provide ready-to-serve quantized models for the community.
>
> **Quantized by NVIDIA.** The weights in this repository are an unmodified copy of [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4) (revision `3431dde3247c13b5957f682b1e3c6fcae2566079`). All credit for the quantization belongs to NVIDIA.

This is an NVFP4-quantized version of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (552B backbone plus 196B Engram conditional memory), quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

**About NVFP4 quantization:** NVFP4 on Blackwell couples a compact E2M1 FP4 codebook with blockwise FP8 (E4M3) scaling over 16-element micro-blocks, so that 4-bit stored values remain numerically useful for neural-network computation. The E2M1 codebook provides a small, nonuniform set of representable magnitudes up to ±6 and relies on saturating behavior rather than IEEE NaN/Inf encodings to maximize usable range per bit. Using an FP8 block scale (rather than power-of-two-only E8M0) enables fractional scales and error-minimizing scale selection. On Blackwell Tensor Cores, native FP4 multipliers exploit E2M1 simplicity while higher-precision FP32 accumulation protects dot-product accuracy.

> **Ready for commercial and non-commercial use under [MIT](https://opensource.org/license/mit).**

## Model Summary

| | |
|:---|:---|
| **Architecture** | Causal Encoder-Decoder MoE with Compressed Sparse Attention 2 (`DeepseekV41ForCausalLM`) |
| **Backbone Parameters** | 552B total |
| **Activated Parameters** | 8B (prefill) / 16B (decode) |
| **Engram Memory** | 196B |
| **Experts** | 384 routed, 6 active, 40 layers |
| **Input** | Text, image |
| **Context Length** | 1M tokens |
| **Checkpoint Size** | ~527 GB |

## Evaluation Results

| Benchmark | MXFP4 (source) | NVFP4 |
|:---|:---:|:---:|
| GPQA Diamond | 91.035 | 91.288 |
| AA-LCR | 78.563 | 78.438 |
| SciCode | 54.401 | 55.843 |
| IFBench | 76.667 | 77.267 |
| MMMU-Pro | 74.046 | 73.699 |
| Terminal-Bench 2.1 | 81.60 | 82.16 |

> Scores reported by NVIDIA for this checkpoint (vLLM). Baseline: [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). `temperature=1.0`, `top_p=0.95`, `reasoning_effort=100`.

## Quantization Details

- **Quantization format:** routed MoE experts (`w1`, `w2`, `w3`) converted from source MXFP4 to NVFP4 W4A4 (group size 16); attention, shared experts, vision, Engram tables and MTP/DSpark keep their source precision (incl. MXFP8)
- **Weight conversion:** lossless — all 16,986,931,200 weight blocks preserve their dequantized values; only block scales are rewritten
- **Calibration dataset:** 1,024 samples from [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) and [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) (activation scales)
- **Tool:** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) v0.47.0rc0

## Usage

### Deploy with SGLang

```bash
python -m sglang.launch_server \
    --model-path AxionML/DeepSeek-V4.1-Flash-NVFP4 \
    --tp 4 \
    --context-length 1048576 \
    --reasoning-parser deepseek-v41 \
    --tool-call-parser deepseekv41 \
    --chunked-prefill-size 4096 \
    --max-running-requests 16
```

### Deploy with vLLM

```bash
vllm serve AxionML/DeepSeek-V4.1-Flash-NVFP4 \
    --tensor-parallel-size 4 \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --language-model-only \
    --max-model-len 1048576 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching
```

Validated upstream on 4x GB300 with `lmsysorg/sglang:dev-cu13-dsv41` and `vllm/vllm-openai:deepseekv41-flash-0909`. Pass a request-level `reasoning_effort` (e.g. `"max"`) to enable thinking in SGLang. DSpark tensors are preserved but speculative decoding was not validated.

## Limitations

The base model was trained on data that may contain toxic language and societal biases. The quantized model inherits these limitations. It may generate inaccurate, biased, or offensive content. Please refer to the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) and the [upstream quantized model card](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4) for full details.

## Credits

- **Base model:** [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- **Quantization:** [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4) by NVIDIA
- **Mirror:** [AxionML](https://huggingface.co/AxionML)
