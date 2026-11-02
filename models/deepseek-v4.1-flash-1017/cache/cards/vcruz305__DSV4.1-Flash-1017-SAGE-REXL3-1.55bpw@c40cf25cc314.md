---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - rexl3
  - exllamav3
  - deepseek
  - moe
  - quantization
  - mixed-precision
---

# DeepSeek-V4.1-Flash-1017 — SAGE REXL3 1.55 bpw

Loadable [ExLlamaV3](https://github.com/turboderp-org/exllamav3) / REXL3 pack of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).

Quantized by [vcruz305](https://huggingface.co/vcruz305) with **SAGE**, our own
mixed-precision quantization method for EXL3. Tensors use different EXL3 `K`
values; the kernels read `K` from each trellis tensor at load.

Sized for a single 128 GB-class unified-memory device (`~108.5 GiB text / ~109.4 GiB VLM` GPU-resident footprint). This **1.55 bpw REXL3 expert quantization preserves exceptionally high token-level agreement (`94.41%` Top-1, `0.0509` mean KLD) with DeepSeek's official mixed FP8/FP4 release** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`).

## Measured quality

Teacher-forced token-level agreement with **DeepSeek's official mixed FP8/FP4 release** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`), full-vocabulary KL divergence per token, on held-out sequences:

| Pack | Resident footprint | KL vs official FP8/FP4 release (mean) | KL vs official FP8/FP4 release (`p50`) | Top-1 Agreement | Top-5 Overlap |
| --- | --- | --- | --- | --- | --- |
| **this pack, 1.55 bpw (`1017`)** | ~108.5 GiB text / ~109.4 GiB VLM | **0.0509** | **0.00007** | **94.41%** | **81.31%** |
| Official mixed FP8/FP4 release | native | 0 | 0 | 100% | 100% |

### Per-category breakdown

| Category | KL vs official FP8/FP4 release (mean) | KL vs official FP8/FP4 release (`p50`) | Top-1 Agreement | Answer-Only Top-1 | Top-5 Overlap |
| --- | --- | --- | --- | --- | --- |
| **Code** | **0.0142** (`0.0111` answer) | `0.000060` | **97.86%** | **98.04%** | **85.76%** |
| **SWE Patch** | **0.0199** (`0.0118` answer) | `0.000020` | **97.60%** | **98.01%** | **84.50%** |
| **Reasoning** | **0.0148** (`0.0096` answer) | `0.000015` | **97.59%** | **98.12%** | **83.60%** |
| **Cyber** | **0.0499** | `0.000138` | **96.05%** | **95.99%** | **82.16%** |
| **General** | **0.0222** (`0.0169` answer) | `0.000002` | **96.02%** | **96.08%** | **81.74%** |
| **IFEval** | **0.0442** (`0.0238` answer) | `0.000002` | **94.59%** | **92.52%** | **79.21%** |
| **Prose / Narrative** | **0.0575** (`0.0492` answer) | `0.000466` | **93.74%** | **93.46%** | **82.91%** |
| **Multilingual** | **0.1303** (`0.1313` answer) | `0.000624` | **91.74%** | **91.39%** | **73.50%** |
| **MMLU-Pro** | **0.0419** (`0.0265` answer) | `0.000007` | **91.54%** | **85.05%** | **78.07%** |
| **Math** | **0.0356** (`0.0329` answer) | `0.000055` | **90.96%** | **80.39%** | **81.41%** |
| **GPQA / STEM** | **0.1052** (`0.0515` answer) | `0.020997` | **89.03%** | **90.12%** | **81.71%** |

## Pack

| Item | Value |
| --- | --- |
| Format | EXL3 / REXL3 (trellis) |
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

cfg = Config.from_directory("vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw")
model = Model.from_config(cfg)
model.load()
tokenizer = Tokenizer.from_config(cfg)
```

## License

MIT, same as [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Cite DeepSeek for the base model. This repository is the EXL3 pack only.
