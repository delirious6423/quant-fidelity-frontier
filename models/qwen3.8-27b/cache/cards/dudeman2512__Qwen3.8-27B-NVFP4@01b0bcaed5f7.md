---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- quantized
- compressed-tensors
- vllm
- nvfp4
---

# Qwen3.8-27B-NVFP4

`nvfp4` quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) by streaming the checkpoint tensor-by-tensor (the model is never fully instantiated).

NVFP4 4-bit float weights in groups of 16 with an FP8 group scale and a per-tensor global scale. Roughly half the size of FP8 at moderate error.

## All quantizations of this model

| Variant | Format | Size | vs BF16 | Mean rel. error | Linears quantized | Left BF16 |
|---|---|---|---:|---:|---:|---:|
| [Qwen3.8-27B-FP8](https://huggingface.co/dudeman2512/Qwen3.8-27B-FP8) | `float-quantized` | 30.35 GB | 55% | 0.0264 | 615 | 0 |
| [Qwen3.8-27B-NVFP4](https://huggingface.co/dudeman2512/Qwen3.8-27B-NVFP4) **&larr; this one** | `nvfp4-pack-quantized` | 19.29 GB | 35% | 0.0950 | 615 | 0 |
| [Qwen3.8-27B-int4](https://huggingface.co/dudeman2512/Qwen3.8-27B-int4) | `pack-quantized` | 18.31 GB | 33% | 0.1178 | 588 | 27 |

*Mean relative error is `||dequant(W) - W|| / ||W||`, averaged over a sample of quantized Linear layers, measured against the original BF16 weights. Lower is better.*

## This variant

| | |
|---|---|
| Format | `nvfp4-pack-quantized` |
| Weight bits | 4 |
| Group size | 16 |
| Strategy | `tensor_group` |
| Linears quantized | 615 |
| Left in BF16 | 0 |
| Shards | 5 |
| On disk | 19.29 GB |
| Mean relative error | 0.0950 |
| Shape/dtype conformance failures | 0 |

## Use with vLLM

```bash
vllm serve dudeman2512/Qwen3.8-27B-NVFP4
```

## How this was made

Every produced tensor is checked for shape/dtype conformance against what the server expects, then reconstruction error is measured against the source BF16 weights, before anything is published. The numbers in the table above are those measurements — not estimates.
