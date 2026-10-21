---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- quantized
- compressed-tensors
- vllm
- fp8
---

# Qwen3.8-27B-FP8

`fp8` quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) by streaming the checkpoint tensor-by-tensor (the model is never fully instantiated).

8-bit float weights (`float8_e4m3fn`), per output channel, with dynamic per-token activation quantization. Highest fidelity of the set and the largest.

## All quantizations of this model

| Variant | Format | Size | vs BF16 | Mean rel. error | Linears quantized | Left BF16 |
|---|---|---|---:|---:|---:|---:|
| [Qwen3.8-27B-FP8](https://huggingface.co/dudeman2512/Qwen3.8-27B-FP8) **&larr; this one** | `float-quantized` | 30.35 GB | 55% | 0.0264 | 615 | 0 |
| [Qwen3.8-27B-NVFP4](https://huggingface.co/dudeman2512/Qwen3.8-27B-NVFP4) | `nvfp4-pack-quantized` | 19.29 GB | 35% | 0.0950 | 615 | 0 |
| [Qwen3.8-27B-int4](https://huggingface.co/dudeman2512/Qwen3.8-27B-int4) | `pack-quantized` | 18.31 GB | 33% | 0.1178 | 588 | 27 |

*Mean relative error is `||dequant(W) - W|| / ||W||`, averaged over a sample of quantized Linear layers, measured against the original BF16 weights. Lower is better.*

## This variant

| | |
|---|---|
| Format | `float-quantized` |
| Weight bits | 8 |
| Group size | per-channel (no grouping) |
| Strategy | `channel` |
| Linears quantized | 615 |
| Left in BF16 | 0 |
| Shards | 7 |
| On disk | 30.35 GB |
| Mean relative error | 0.0264 |
| Shape/dtype conformance failures | 0 |

## Use with vLLM

```bash
vllm serve dudeman2512/Qwen3.8-27B-FP8
```

## How this was made

Every produced tensor is checked for shape/dtype conformance against what the server expects, then reconstruction error is measured against the source BF16 weights, before anything is published. The numbers in the table above are those measurements — not estimates.
