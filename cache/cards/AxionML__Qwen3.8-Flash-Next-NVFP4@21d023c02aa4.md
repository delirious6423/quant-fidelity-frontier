---
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
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

# AxionML Qwen3.8-Flash-Next-NVFP4

> Mirrored by [AxionML](https://huggingface.co/AxionML) for open-source serving and deployment use cases. Part of AxionML's effort to provide ready-to-serve quantized models for the community.
>
> **Quantized by RadixArk.** The weights in this repository are an unmodified copy of [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) (revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`). All credit for the quantization belongs to RadixArk.

This is an NVFP4-quantized version of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (~180B total parameters: 125B core with 6B activated, plus 51B n-gram embedding and 4B MTP), quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

**About NVFP4 quantization:** NVFP4 on Blackwell couples a compact E2M1 FP4 codebook with blockwise FP8 (E4M3) scaling over 16-element micro-blocks, so that 4-bit stored values remain numerically useful for neural-network computation. The E2M1 codebook provides a small, nonuniform set of representable magnitudes up to ±6 and relies on saturating behavior rather than IEEE NaN/Inf encodings to maximize usable range per bit. Using an FP8 block scale (rather than power-of-two-only E8M0) enables fractional scales and error-minimizing scale selection. On Blackwell Tensor Cores, native FP4 multipliers exploit E2M1 simplicity while higher-precision FP32 accumulation protects dot-product accuracy.

> **Licensed under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)** (included as `LICENSE`). Permissive, but Model-as-a-Service and AI Work Assistant businesses need a separate license from Qwen for commercial use — read it first.

## Model Summary

| | |
|:---|:---|
| **Architecture** | Hybrid multimodal MoE: Gated DeltaNet + Qwen Sparse Attention (QSA), hyper-connection streams, PLE n-gram injection |
| **Total Parameters** | ~180B (125B core, 6B activated; 51B n-gram embedding; 4B MTP) |
| **Layers / Experts** | 48 decoder layers, 512 routed experts (top-10) + shared expert, 1 MTP layer |
| **Input** | Text, image, video |
| **Context Length** | 262K tokens |
| **Checkpoint Size** | ~135 GB (vs ~360 GB BF16) |

## Evaluation Results

| Benchmark | Protocol | BF16 reference | NVFP4 |
|:---|:---|:---:|:---:|
| GSM8K | Full 1,319, `t=0.6`, `top_p=0.95` | 97.12–97.50 | 97.27 |
| AIME 2026 | 30 problems × 8, `t=1.0` | 100 | 98.75 pass@1 |

> Scores reported by RadixArk. BF16 reference runs were recorded on an earlier revision of the base model, so treat the comparison as indicative. RadixArk notes long agentic generations tend to run longer than BF16.

## Quantization Details

- **Quantization format:** NVFP4 W4A4 (group size 16, FP8 E4M3 block scales, dynamic activations) on the routed experts of all 48 MoE layers only
- **Unchanged:** attention, QSA, GDN, hyper-connections, shared experts, routers, embeddings, LM head, vision and all MTP tensors stay BF16 and byte-identical to the source; PLE n-gram tables use the FP8 versions from `Qwen/Qwen3.8-Flash-Next-FP8`
- **Calibration dataset:** 128 `cnn_dailymail` articles (512 tokens), max calibration
- **Tool:** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) v0.46.0

## Usage

### Deploy with SGLang

```bash
python -m sglang.launch_server \
    --model-path AxionML/Qwen3.8-Flash-Next-NVFP4 \
    --tp 2 \
    --quantization modelopt_fp4 \
    --fp4-gemm-backend flashinfer_cutlass \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer \
    --mamba-track-interval 64 \
    --chunked-prefill-size 4096 \
    --max-running-requests 36 \
    --context-length 262144 \
    --mem-fraction-static 0.80
```

Requires an SGLang build with `qwen4_exp` model support. Validated upstream on GB300 and B300. Audit reports (`validate_*_report.json`, `qualification-notes.md`) are included in this repository.

## Limitations

The base model was trained on data that may contain toxic language and societal biases. The quantized model inherits these limitations. It may generate inaccurate, biased, or offensive content. Please refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and the [upstream quantized model card](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) for full details.

## Credits

- **Base model:** [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Quantization:** [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) by RadixArk
- **Mirror:** [AxionML](https://huggingface.co/AxionML)
