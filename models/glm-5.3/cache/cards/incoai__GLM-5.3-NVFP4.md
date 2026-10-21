---
pipeline_tag: text-generation
base_model:
- zai-org/GLM-5.3
base_model_relation: quantized
license: other
license_name: glm-5.3
license_link: LICENSE
library_name: transformers
tags:
- GLM-5
- ModelOpt
- quantized
- 4-bit precision
- FP4
- nvfp4
---

# GLM-5.3-NVFP4

This repository contains an NVFP4-quantized version of Z.ai's
[`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3).
The weights and activations of the routed-expert linear layers are in NVFP4,
and the KV cache is in FP8. The checkpoint uses the NVIDIA Model Optimizer
format (`quant_method: modelopt`) and is served directly by SGLang and vLLM
on NVIDIA Blackwell GPUs (SM100+).

- **Base model:** [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)
- **Quantization:** NVFP4 weights + static NVFP4 activations; FP8 KV cache
- **Size:** 433 GiB
- **License:** [GLM-5.3 License](https://huggingface.co/incoai/GLM-5.3-NVFP4/blob/main/LICENSE)

## Quantization Method

This checkpoint was produced with our in-house post-training quantization
toolchain and exported in the
[NVIDIA Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
format. Only the weights and activations of the linear
operators within the routed MoE experts are quantized to NVFP4, with static
per-tensor activation scales. The KV cache is quantized to FP8 with static
unit scales (`kv_cache_quant_algo: FP8`).

## Quick Start

The commands below enable speculative decoding with the
[DFlash 2](https://huggingface.co/incoai/GLM-5.3-DFlash2) draft model
(lossless, 7 draft tokens per verification step); remove the
`--speculative-*` flags to serve without it.

Serve with [SGLang](https://github.com/sgl-project/sglang) (main):

```bash
python3 -m sglang.launch_server \
    --model-path incoai/GLM-5.3-NVFP4 \
    --tp 8 \
    --quantization modelopt_fp4 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --chunked-prefill-size 8192 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path incoai/GLM-5.3-DFlash2 \
    --speculative-draft-attention-backend trtllm_mha
```

Or with [vLLM](https://github.com/vllm-project/vllm) (v0.28.0 or later):

```bash
vllm serve incoai/GLM-5.3-NVFP4 \
    --tensor-parallel-size 8 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --speculative-config '{"method":"dflash","model":"incoai/GLM-5.3-DFlash2","num_speculative_tokens":7}'
```

## Accuracy

We compare Z.ai's original FP8 release with this NVFP4 checkpoint.
Higher is better.

| Precision | GPQA Diamond | AIME 2025 | MATH-500 | HLE | AA-LCR |
| :--- | ---: | ---: | ---: | ---: | ---: |
| FP8 | 91.1 | 94.3 | 95.6 | 35.9 | 73.6 |
| NVFP4 | 91.2 | 95.1 | 95.2 | 35.2 | 73.0 |

## License

This model is a quantized version of GLM-5.3 and is distributed under Z.ai's
[GLM-5.3 License](https://huggingface.co/incoai/GLM-5.3-NVFP4/blob/main/LICENSE),
which it inherits from the base model.
