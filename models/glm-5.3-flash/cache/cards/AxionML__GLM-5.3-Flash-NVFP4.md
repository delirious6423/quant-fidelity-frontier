---
pipeline_tag: image-text-to-text
base_model:
- zai-org/GLM-5.3-Flash
base_model_relation: quantized
license: mit
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

# AxionML GLM-5.3-Flash-NVFP4

> Mirrored by [AxionML](https://huggingface.co/AxionML) for open-source serving and deployment use cases. Part of AxionML's effort to provide ready-to-serve quantized models for the community.
>
> **Quantized by RadixArk.** The weights in this repository are an unmodified copy of [RadixArk/GLM-5.3-Flash-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4) (revision `f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35`). All credit for the quantization belongs to RadixArk.

This is an NVFP4-quantized version of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (320B total parameters, 18B activated), quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

**About NVFP4 quantization:** NVFP4 on Blackwell couples a compact E2M1 FP4 codebook with blockwise FP8 (E4M3) scaling over 16-element micro-blocks, so that 4-bit stored values remain numerically useful for neural-network computation. The E2M1 codebook provides a small, nonuniform set of representable magnitudes up to ±6 and relies on saturating behavior rather than IEEE NaN/Inf encodings to maximize usable range per bit. Using an FP8 block scale (rather than power-of-two-only E8M0) enables fractional scales and error-minimizing scale selection. On Blackwell Tensor Cores, native FP4 multipliers exploit E2M1 simplicity while higher-precision FP32 accumulation protects dot-product accuracy.

> **Ready for commercial and non-commercial use under [MIT](https://opensource.org/license/mit).**

## Model Summary

| | |
|:---|:---|
| **Architecture** | Natively multimodal hybrid-attention MoE: KDA linear attention, DSA sparse attention with indexer, MLA, manifold-constrained hyper-connections |
| **Total Parameters** | 320B |
| **Activated Parameters** | 18B |
| **Layers / Experts** | 45 layers (3 dense + 42 MoE), 288 routed experts + shared expert, native MTP/NextN layer |
| **Input** | Text, image, video |
| **Context Length** | 1,048,576 tokens |
| **Checkpoint Size** | ~203 GB |

## Evaluation Results

| Benchmark | Protocol | NVFP4 |
|:---|:---|:---:|
| GSM8K | Full 1,319 × 4 seeds | 97.14 |
| AIME 2026 | 30 × 16 × 4 seeds | 92.45 |
| Terminal-Bench 2.1 | 89 tasks, terminus-2, pass@1 | 83.1 |

> Scores reported by RadixArk for this checkpoint (SGLang, 4x GB300, FP8 KV cache, NEXTN speculative decoding). Text-only evaluations.

## Quantization Details

- **Quantization format:** NVFP4 W4A4 (group size 16, abs-max scaling) on `gate_proj` / `up_proj` / `down_proj` of all routed experts, the shared expert and the dense MLPs in layers 0–2
- **Unchanged:** all attention (KDA, DSA indexer, MLA), hyper-connections, norms, routers, vision tower, MTP layer, embeddings and `lm_head`; KV cache not quantized in the checkpoint (FP8 KV validated at serving time)
- **Calibration dataset:** 1,024 `cnn_dailymail` samples, length 512
- **Tool:** [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) 0.46.0

## Usage

### Deploy with SGLang

```bash
python3 -m sglang.launch_server \
    --model-path AxionML/GLM-5.3-Flash-NVFP4 \
    --quantization modelopt_fp4 \
    --tp-size 4 \
    --dsa-prefill-backend trtllm \
    --dsa-decode-backend trtllm \
    --kv-cache-dtype fp8_e4m3 \
    --moe-runner-backend flashinfer_cutlass \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --speculative-adaptive \
    --reasoning-parser glm45 \
    --tool-call-parser glm47
```

Use the `lmsysorg/sglang:glm-5.3-flash` image. Audit evidence (`tensor-audit-b.json`, `precision-contract-b.json`) is included in this repository.

## Limitations

The base model was trained on data that may contain toxic language and societal biases. The quantized model inherits these limitations. It may generate inaccurate, biased, or offensive content. Please refer to the [original model card](https://huggingface.co/zai-org/GLM-5.3-Flash) and the [upstream quantized model card](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4) for full details.

## Credits

- **Base model:** [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
- **Quantization:** [RadixArk/GLM-5.3-Flash-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4) by RadixArk
- **Mirror:** [AxionML](https://huggingface.co/AxionML)
