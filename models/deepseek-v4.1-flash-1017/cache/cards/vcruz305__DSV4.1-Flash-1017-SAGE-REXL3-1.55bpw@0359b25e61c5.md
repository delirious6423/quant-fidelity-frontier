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

Sized for a single 128 GB-class unified-memory device (`~108.5 GiB text / ~109.4 GiB VLM` GPU-resident footprint). This **1.55 bpw REXL3 expert quantization preserves high token-level agreement (`93.95%` Top-1, `99.53%` Top-5, `0.1058` mean KLD) with DeepSeek's official mixed FP8/FP4 release** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`).

## Measured quality

Teacher-forced token-level agreement with **DeepSeek's official mixed FP8/FP4 release** (`deepseek-ai/DeepSeek-V4.1-Flash-1017`), full-vocabulary (`129,280`) `float64` KL divergence per token, on held-out sequences:

| Pack | Resident footprint | KL vs official FP8/FP4 release (mean) | KL vs official FP8/FP4 release (`p50`) | Top-1 Agreement | Top-5 Agreement | Top-5 Overlap |
| --- | --- | --- | --- | --- | --- | --- |
| **this pack, 1.55 bpw (`1017`)** | ~108.5 GiB text / ~109.4 GiB VLM | **0.1058** | **0.00016** | **93.95%** | **99.53%** | **81.01%** |
| Official mixed FP8/FP4 release | native | 0 | 0 | 100% | 100% | 100% |

### Per-category breakdown

| Category | KL vs official FP8/FP4 release (mean) | KL vs official FP8/FP4 release (`p50`) | Top-1 Agreement | Answer-Only Top-1 | Top-5 Agreement | Top-5 Overlap |
| --- | --- | --- | --- | --- | --- | --- |
| **Code** | **0.0168** (`0.0054` answer) | `0.000001` | **98.91%** | **99.53%** | **99.87%** (`100.0%` answer) | **81.21%** |
| **Reasoning** | **0.0370** (`0.0348` answer) | `0.000455` | **95.86%** | **93.94%** | **99.83%** (`100.0%` answer) | **86.31%** |
| **Math** | **0.0718** (`0.1092` answer) | `0.000398` | **94.38%** | **93.02%** | **99.59%** (`100.0%` answer) | **82.75%** |
| **General** | **0.2317** (`0.2471` answer) | `0.001288` | **89.19%** | **88.54%** | **99.08%** (`99.00%` answer) | **76.05%** |

### Full evaluation breakdown (all splits & long-context)

| Evaluation Split (`float64`, `129,280` Vocab) | Tokens | Top-1 Agreement | Top-5 Agreement | Top-5 Overlap | KL Mean | KL Median (`p50`) | KL `p95` | Teacher PPL | Student PPL | $\Delta$ PPL |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`e2e_eval` (Assistant Turn)** | `21,458` | **93.95%** | **99.53%** | **81.01%** | **0.1058** | **0.000163** | `0.5398` | `1.8575` | `1.8653` | `+0.0078` |
| **`e2e_eval` (Non-Header)** | `29,492` | **93.48%** | **99.46%** | **80.64%** | **0.1245** | **0.000237** | `0.6608` | `1.9981` | **1.9736** | `-0.0245` |
| **`e2e_eval` (Answer Only)** | `9,305` | **92.90%** | **99.41%** | `78.02%` | `0.1499` | **0.000044** | `0.8296` | `1.5794` | `1.6254` | `+0.0461` |
| **`e2e_eval` (All Tokens)** | `32,760` | **91.87%** | **98.48%** | `78.31%` | `0.2083` | **0.000276** | `1.0843` | `2.3394` | **2.2861** | `-0.0533` |
| **`e2e_eval` — `code` (Answer Only)** | `3,605` | **99.53%** | **100.00%** | `81.62%` | **0.0054** | **0.000001** | `0.0080` | `1.0918` | **1.0751** | `-0.0167` |
| **`e2e_eval` — `code` (Assistant Turn)** | `4,586` | **98.91%** | **99.87%** | `81.21%` | **0.0168** | **0.000001** | `0.0380` | `1.2659` | **1.2366** | `-0.0293` |
| **`e2e_eval` — `code` (Non-Header)** | `7,436` | **98.31%** | **99.84%** | `80.72%` | **0.0385** | **0.000005** | `0.0713` | `1.2925` | **1.2703** | `-0.0222` |
| **`e2e_eval` — `reasoning` (Assistant Turn)** | `5,174` | **95.86%** | **99.83%** | **86.31%** | **0.0370** | `0.000455` | `0.1796` | `1.8935` | **1.8927** | `-0.0008` |
| **`e2e_eval` — `reasoning` (Non-Header)** | `7,464` | **94.49%** | **99.73%** | `84.96%` | **0.0695** | `0.000919` | `0.3168` | `2.0483` | **2.0315** | `-0.0168` |
| **`e2e_eval` — `math` (Assistant Turn)** | `4,434` | **94.38%** | **99.59%** | `82.75%` | **0.0718** | `0.000398` | `0.3740` | `2.0755` | **2.0063** | `-0.0692` |
| **`e2e_eval` — `math` (Non-Header)** | `6,960` | **92.66%** | **99.45%** | `81.35%` | `0.1376` | `0.000604` | `0.7823` | `2.3283` | **2.1730** | `-0.1553` |
| **`e2e_eval` — `general` (Assistant Turn)** | `7,264` | **89.19%** | **99.08%** | `76.05%` | `0.2317` | `0.001288` | `1.3570` | `2.1814` | `2.2891` | `+0.1077` |
| **`e2e_eval` — `general` (Non-Header)** | `7,632` | **88.52%** | **98.85%** | `75.70%` | `0.2502` | `0.001698` | `1.4465` | `2.5931` | `2.6994` | `+0.1064` |
| **`long17k` (`17,408`-tok Non-Header)** | `15,700` | **92.18%** | **99.28%** | `77.42%` | **0.1878** | **0.000212** | `1.1103` | `2.0717` | **1.9378** | `-0.1339` |
| **`long17k` (`17,408`-tok All Tokens)** | `17,407` | **91.38%** | **98.60%** | `75.34%` | `0.2477` | **0.000205** | `1.4516` | `2.3168` | **2.1702** | `-0.1466` |
| **`wiki1100` (Answer Only)** | `718` | **91.09%** | **99.03%** | `73.40%` | `0.1998` | **0.000572** | `1.4708` | `1.3835` | `1.4765` | `+0.0930` |
| **`wiki1100` (Assistant Turn)** | `962` | **90.44%** | **99.17%** | `74.64%` | `0.1899` | **0.001315** | `1.1610` | `2.0029` | `2.1031` | `+0.1002` |
| **`wiki1100` (Non-Header)** | `1,016` | **89.57%** | **99.02%** | `74.29%` | `0.2273` | **0.001960** | `1.3351` | `2.6041` | `2.7162` | `+0.1121` |

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
