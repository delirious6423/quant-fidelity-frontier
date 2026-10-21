---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - deepseek
  - moe
  - quantization
  - mixed-precision
---

# DeepSeek-V4.1-Flash-1017 — DERISKED EXL3 2.50 bpw

Official collaboration with **Blackfrost-AI (Sir Frosty)**.

Loadable [ExLlamaV3](https://github.com/turboderp-org/exllamav3) pack of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

Quantized by [vcruz305](https://huggingface.co/vcruz305) with **SAGE**, our own
mixed-precision quantization method for EXL3. Tensors use different EXL3 `K`
values; the kernels read `K` from each trellis tensor at load.

Sized for **2× 128 GB-class unified-memory devices** (`~170.6 GiB` total GPU-resident footprint, `~85.3 GiB` per device). This **2.50 bpw EXL3 expert quantization preserves `96.71%` assistant-turn Top-1 agreement** with DeepSeek's official mixed FP8/FP4 release (`deepseek-ai/DeepSeek-V4.1-Flash-1017`).

## Measured quality

Teacher-forced token-level agreement with **DeepSeek's official mixed FP8/FP4 release** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`) on held-out sequences:

| Slice | Overall Top-1 Agreement |
| --- | --- |
| **Assistant turn** | **96.71%** |
| **Non-header** | **96.23%** |
| **All tokens** | **95.05%** |

### Per-category breakdown

| Slice | Code | Reasoning | Math | General |
| --- | --- | --- | --- | --- |
| **Assistant turn Top-1** | **99.41%** | **97.62%** | **97.14%** | **94.11%** |
| **Non-header Top-1** | **99.17%** | **96.64%** | **95.72%** | **93.42%** |
| **All tokens Top-1** | **98.07%** | **95.64%** | **94.55%** | **91.95%** |

## Pack

| Item | Value |
| --- | --- |
| Format | EXL3 (trellis) |
| Profile | **2.50 bpw** SAGE mixed-K |
| Resident footprint | ~170.6 GiB (~85.3 GiB / device at TP2) |
| Shards | 17 (`model-00001-of-00017.safetensors` .. `model-00017-of-00017.safetensors`) |
| Index | `model.safetensors.index.json` |
| `quant_method` | `exl3` |

## About SAGE

SAGE is our own mixed-precision quantization method for EXL3. The method is not published. If a tensor is EXL3, its `K` is self-describing on the trellis.

## Source model (DeepSeek, not this pack)

V4.1-Flash-1017 is DeepSeek’s MIT CED MoE: 40 layers, 384 routed experts + 1 shared, top-6. Architecture, sampling defaults, and any benchmark rows on the [upstream card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) are **DeepSeek’s**.

Upstream sampling (start here, then tune): `temperature=1.0`, `top_p=0.95`.

## Load

ExLlamaV3, point at this repo (or a local snapshot):

```python
from exllamav3 import Config, Model, Tokenizer

cfg = Config.from_directory("vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw")
model = Model.from_config(cfg)
model.load()
tokenizer = Tokenizer.from_config(cfg)
```

## License

MIT, same as [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Cite DeepSeek for the base model. This repository is the EXL3 pack only.
