---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - fp8
  - modelopt
  - quantized
  - vllm
  - qwen3_5
---

# Qwen3.8-27B — NVFP4 (ModelOpt)

NVFP4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 0.45
using NVIDIA's own recipe for the `qwen3_5` family
(`huggingface/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast`) — the same
mixed-precision scheme NVIDIA used for its official Qwen3.6-27B NVFP4 release:

| Component | Precision |
|---|---|
| MLP projections (gate/up/down) + `lm_head` | **NVFP4** (W4A16, weight-only, block size 16) |
| Self-attention and linear-attention (GDN) projections | **FP8** (E4M3, weights + activations) |
| KV cache | **FP8** (constant amax) |
| Vision tower, MTP head, conv/gating layers | **BF16** (unquantized) |

Checkpoint size is ~21 GB (vs. ~55 GB BF16).

## Calibration

Post-training quantization with 1024 samples (max. 512 tokens each), 256 each
from four openly available datasets: `cnn_dailymail`,
`Magpie-Align/Magpie-Pro-MT-300K-v0.1`, `nvidia/OpenCodeReasoning`,
`nvidia/OpenMathReasoning`. Attention implementation during calibration: SDPA.

## Serving with vLLM

```bash
vllm serve <this-repo> \
  --quantization modelopt \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice
```

NVFP4 runs natively on Blackwell GPUs; the checkpoint loads on earlier
architectures via vLLM's ModelOpt support with reduced benefit.

## Notes

- Community quantization by [A2Genesis](https://a2genesis.de) — not affiliated
  with or endorsed by the Qwen team or NVIDIA.
- The MTP (multi-token prediction) tensors are preserved in BF16, so
  speculative decoding remains available.
- License: Apache 2.0, inherited from the base model. All credit for the model
  itself belongs to the Qwen team.
