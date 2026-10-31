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

# DeepSeek-V4.1-Flash-1017 — SAGE EXL3 1.55 bpw

Loadable [ExLlamaV3](https://github.com/turboderp-org/exllamav3) pack of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

Quantized by [vcruz305](https://huggingface.co/vcruz305) with **SAGE**, our own
mixed-precision quantization method for EXL3. Tensors use different EXL3 `K`
values; the kernels read `K` from each trellis tensor at load.

Sized for a single 128 GB-class unified-memory device (`~108.5 GiB text / ~109.4 GiB VLM` GPU-resident footprint). This **1.55 bpw EXL3 expert quantization preserves high token-level agreement (`94.52%` Top-1 on `e2e_eval`, `93.7%` Top-1 on `long17k`, `88.2%` Top-1 on `wiki1100`) with DeepSeek's official mixed FP8/FP4 release** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`).

## Measured quality

Teacher-forced token-level agreement with **DeepSeek's official mixed FP8/FP4 release** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`), full-vocabulary KL divergence per token, on held-out sequences:

| Dataset | KL vs official FP8/FP4 release (mean) | KL vs official FP8/FP4 release (`p50`) | Top-1 Agreement | Top-5 Overlap |
| --- | --- | --- | --- | --- |
| **`e2e_eval` (Assistant-Turn, 21,458 tok)** | **0.0982** | **0.00019** | **94.52%** | **81.16%** |
| **`e2e_eval` (Non-Header, 29,492 tok)** | **0.1150** | **0.00029** | **94.00%** | **80.84%** |
| **`long17k` (17,408 tok long-context)** | **0.0445** | **0.00027** | **93.70%** | **—** |
| **`wiki1100` (1,100 tok WikiText-2)** | **0.0882** | **0.00250** | **88.20%** | **—** |

### Per-category breakdown (`e2e_eval`)

| Category | KL vs official FP8/FP4 release (mean) | KL vs official FP8/FP4 release (`p50`) | Top-1 Agreement | Answer-Only Top-1 | Top-5 Overlap |
| --- | --- | --- | --- | --- | --- |
| **Code** | **0.0186** (`0.0046` answer) | `0.000004` | **99.08%** | **99.61%** | **80.38%** |
| **Reasoning** | **0.0349** (`0.0306` answer) | `0.000480` | **96.04%** | **97.98%** | **86.34%** |
| **Math** | **0.0477** (`0.0408` answer) | `0.000306` | **95.56%** | **94.19%** | **83.99%** |
| **General** | **0.2245** | `0.001505` | **89.94%** | **89.21%** | **76.24%** |

## Pack

| Item | Value |
| --- | --- |
| Format | EXL3 (trellis) |
| Profile | **1.55 bpw** SAGE mixed-K |
| Resident footprint | ~108.5 GiB text / ~109.4 GiB VLM |
| Shards | 17 (`model-00001-of-00017.safetensors` .. `model-00017-of-00017.safetensors`) |
| Index | `model.safetensors.index.json` |
| `quant_method` | `exl3` |

## About SAGE

SAGE is our own mixed-precision quantization method for EXL3. The method is not published. If a tensor is EXL3, its `K` is self-describing on the trellis.

## Source model (DeepSeek, not this pack)

V4.1-Flash-1017 is DeepSeek’s MIT CED MoE: 40 layers, 384 routed experts + 1 shared, top-6. Architecture, sampling defaults, and any benchmark rows on the [upstream card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) are **DeepSeek’s**.

Upstream recommended sampling: `temperature=1.0`, `top_p=1.0`. This pack's `generation_config.json` currently sets `top_p=0.95`; set `top_p` explicitly to follow upstream.

## Load

ExLlamaV3, point at this repo (or a local snapshot):

```python
from exllamav3 import Config, Model, Tokenizer

cfg = Config.from_directory("vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.55bpw")
model = Model.from_config(cfg)
model.load()
tokenizer = Tokenizer.from_config(cfg)
```

## License

MIT, same as [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Cite DeepSeek for the base model. This repository is the EXL3 pack only.
