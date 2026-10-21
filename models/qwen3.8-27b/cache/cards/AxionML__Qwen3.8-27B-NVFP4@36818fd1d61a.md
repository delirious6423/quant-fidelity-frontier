---
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
tags:
- AxionML
- ModelOpt
- Qwen3.8
- quantized
- NVFP4
- nvfp4
- sglang
- mirror
---

# AxionML Qwen3.8-27B-NVFP4

> Mirrored by [AxionML](https://huggingface.co/AxionML) for open-source serving and deployment use cases. Part of AxionML's effort to provide ready-to-serve quantized models for the community.
>
> **Quantized by RadixArk.** The weights in this repository are an unmodified copy of [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) (revision `319f741cce68d7914884900c138a1fbb70a42f30`). All credit for the quantization belongs to RadixArk.

This is an NVFP4-quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (27B parameters, dense multimodal), quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

**About NVFP4 quantization:** NVFP4 on Blackwell couples a compact E2M1 FP4 codebook with blockwise FP8 (E4M3) scaling over 16-element micro-blocks, so that 4-bit stored values remain numerically useful for neural-network computation. The E2M1 codebook provides a small, nonuniform set of representable magnitudes up to ±6 and relies on saturating behavior rather than IEEE NaN/Inf encodings to maximize usable range per bit. Using an FP8 block scale (rather than power-of-two-only E8M0) enables fractional scales and error-minimizing scale selection. On Blackwell Tensor Cores, native FP4 multipliers exploit E2M1 simplicity while higher-precision FP32 accumulation protects dot-product accuracy.

> **Ready for commercial and non-commercial use under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).**

## Model Summary

| | |
|:---|:---|
| **Architecture** | Dense multimodal transformer (text, image, video input) |
| **Parameters** | 27B |
| **Context Length** | 262,144 tokens (native) |
| **Checkpoint Size** | ~21.9 GB |

## Evaluation Results

| Benchmark | Protocol | NVFP4 |
|:---|:---|:---:|
| GSM8K | Full 1,319-example split, thinking mode | 97.27 |
| Terminal-Bench 2.1 | 84-task subset, pass@1 | 73.81 |

> Scores reported by RadixArk for this checkpoint on 4x B300/GB300 (SGLang, TP4). GSM8K used `temperature=1.0`, `top_p=0.95`, `top_k=20`; text-only.

## Quantization Details

- **Quantization format:** mixed NVFP4 — MLP `gate_proj` / `up_proj` / `down_proj` and `lm_head` in dynamic NVFP4 W4A4 (group size 16); attention weights in FP8; MTP and vision tensors kept in BF16
- **KV cache:** FP8
- **Calibration dataset:** 1,024 samples from `abisee/cnn_dailymail` (train split), sequence length 512
- **Tool:** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), commit `87c9f8cf`

## Usage

### Deploy with SGLang

```bash
sglang serve \
    --trust-remote-code \
    --model-path AxionML/Qwen3.8-27B-NVFP4 \
    --tp-size 4 \
    --mem-fraction-static 0.75 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

See the [SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B) for other topologies.

## Limitations

The base model was trained on data that may contain toxic language and societal biases. The quantized model inherits these limitations. It may generate inaccurate, biased, or offensive content. Please refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B) and the [upstream quantized model card](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) for full details.

## Credits

- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Quantization:** [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) by RadixArk
- **Mirror:** [AxionML](https://huggingface.co/AxionML)
