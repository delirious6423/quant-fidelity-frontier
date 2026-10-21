---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- quantized
- compressed-tensors
- vllm
- int4
---

# Qwen3.8-27B-int4

`int4` quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) by streaming the checkpoint tensor-by-tensor (the model is never fully instantiated).

4-bit integer weights in groups of 128 with a BF16 scale. Smallest of the set and the highest error.

## All quantizations of this model

| Variant | Format | Size | vs BF16 | Mean rel. error | Linears quantized | Left BF16 |
|---|---|---|---:|---:|---:|---:|
| [Qwen3.8-27B-FP8](https://huggingface.co/dudeman2512/Qwen3.8-27B-FP8) | `float-quantized` | 30.35 GB | 55% | 0.0264 | 615 | 0 |
| [Qwen3.8-27B-NVFP4](https://huggingface.co/dudeman2512/Qwen3.8-27B-NVFP4) | `nvfp4-pack-quantized` | 19.29 GB | 35% | 0.0950 | 615 | 0 |
| [Qwen3.8-27B-int4](https://huggingface.co/dudeman2512/Qwen3.8-27B-int4) **&larr; this one** | `pack-quantized` | 18.31 GB | 33% | 0.1178 | 588 | 27 |

*Mean relative error is `||dequant(W) - W|| / ||W||`, averaged over a sample of quantized Linear layers, measured against the original BF16 weights. Lower is better.*

## This variant

| | |
|---|---|
| Format | `pack-quantized` |
| Weight bits | 4 |
| Group size | 128 |
| Strategy | `group` |
| Linears quantized | 588 |
| Left in BF16 | 27 |
| Shards | 5 |
| On disk | 18.31 GB |
| Mean relative error | 0.1178 |
| Shape/dtype conformance failures | 0 |

### Layers left in BF16

27 Linear layers could not be quantized at group size 128: their `in_features` is not a multiple of 128. They are stored as BF16 and listed in the `ignore` list of `config.json`, so servers load them unquantized rather than misreading them.

They are all in the vision tower, so the language model is fully quantized:

```
model.visual.blocks.0.mlp.linear_fc2
model.visual.blocks.1.mlp.linear_fc2
model.visual.blocks.10.mlp.linear_fc2
model.visual.blocks.11.mlp.linear_fc2
model.visual.blocks.12.mlp.linear_fc2
model.visual.blocks.13.mlp.linear_fc2
... and 21 more
```

## Use with vLLM

```bash
vllm serve dudeman2512/Qwen3.8-27B-int4
```

## How this was made

Every produced tensor is checked for shape/dtype conformance against what the server expects, then reconstruction error is measured against the source BF16 weights, before anything is published. The numbers in the table above are those measurements — not estimates.
