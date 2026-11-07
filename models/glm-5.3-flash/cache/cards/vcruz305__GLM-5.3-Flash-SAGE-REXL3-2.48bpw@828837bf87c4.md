---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - rexl3
  - exllamav3
  - glm
  - moe
  - quantization
  - mixed-precision
gated: true
extra_gated_prompt: "By requesting access you agree to use this model under its MIT license and to cite the base model. Requests are reviewed manually."
extra_gated_fields:
  I agree to use this model under the MIT license, consistent with the base model's terms: checkbox
---

# GLM-5.3-Flash — SAGE REXL3 2.48 bpw

Loadable [ExLlamaV3](https://github.com/turboderp-org/exllamav3) / REXL3 pack of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

Quantized by [vcruz305](https://huggingface.co/vcruz305) with **SAGE**, our own
mixed-precision quantization method for EXL3. Tensors use different EXL3 `K`
values; the kernels read `K` from each trellis tensor at load.

Sized for a single 128 GB-class unified-memory device (`~96.5 GiB` GPU-resident footprint, text + vision). This **2.48 bpw REXL3 expert quantization preserves exceptionally high token-level agreement (`95.27%` Top-1, `0.0300` mean KLD) with Z.ai's official FP8 release** (`zai-org/GLM-5.3-Flash`).

## Measured quality

Teacher-forced token-level agreement with **Z.ai's official FP8 release** (`zai-org/GLM-5.3-Flash`), full-vocabulary KL divergence per token, on held-out sequences (assistant-turn tokens, `53,251` scored):

| Pack | Resident footprint | KL vs official FP8 release (mean) | KL vs official FP8 release (`p50`) | Top-1 Agreement | Top-5 Agreement | Top-5 Overlap |
| --- | --- | --- | --- | --- | --- | --- |
| **this pack, 2.48 bpw** | ~96.5 GiB | **0.0300** | **0.000198** | **95.27%** | **99.74%** | **81.39%** |
| Official FP8 release | native | 0 | 0 | 100% | 100% | 100% |

### Per-category breakdown

| Category | KL vs official FP8 release (mean) | KL vs official FP8 release (`p50`) | Top-1 Agreement | Answer-Only Top-1 | Top-5 Overlap |
| --- | --- | --- | --- | --- | --- |
| **general** | **0.0192** | `0.000021` | **95.94%** | **95.99%** | **82.24%** |
| **prose_narrative** | **0.0387** | `0.001001` | **94.05%** | **93.53%** | **81.97%** |
| **multilingual** | **0.0458** | `0.000617` | **94.35%** | **93.93%** | **82.50%** |
| **reasoning** | **0.0222** | `0.000039` | **97.05%** | **97.79%** | **81.87%** |
| **code** | **0.0203** | `0.000069` | **97.03%** | **97.44%** | **82.25%** |
| **ifeval** | **0.0156** | `0.000131` | **96.58%** | **95.59%** | **77.02%** |
| **gpqa_stem** | **0.0541** | `0.009890` | **91.58%** | **91.38%** | **85.46%** |
| **swe_patch** | **0.0185** | `0.000071` | **96.92%** | **97.24%** | **81.82%** |
| **math** | **0.0424** | `0.000475` | **91.93%** | **83.54%** | **76.36%** |
| **cyber** | **0.0256** | `0.000098` | **96.95%** | **95.43%** | **81.44%** |
| **mmlu_pro** | **0.0231** | `0.000191` | **95.38%** | **93.47%** | **77.66%** |

### Held-out & long-context suites (assistant-turn tokens)

| Suite | Context | Assistant-turn tokens | KL vs official FP8 release (mean) | KL vs official FP8 release (`p50`) | Top-1 Agreement | Top-5 Agreement | Answer-Only Top-1 | Top-5 Overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **wiki1100** | 1,100-token prefix | **898** | **0.0590** | `0.006229` | **90.76%** | **99.55%** | **90.64%** | **81.63%** |
| **long17k** | 17,408-token long context | **10,754** | **0.4532** | `0.047481` | **81.39%** | **95.55%** | **81.43%** | **–** |

## Pack

| Item | Value |
| --- | --- |
| Format | EXL3 / REXL3 (trellis) |
| Profile | **2.48 bpw** SAGE mixed-K |
| Resident footprint | ~96.5 GiB (text + vision) |
| Shards | 15 (`model-00001-of-00015.safetensors` .. `model-00015-of-00015.safetensors`) |
| Index | `model.safetensors.index.json` |
| `quant_method` | `exl3` |
| `model_type` | `glm5_next` |

## Serve

REXL3 engine v0.2.0+ on one NVIDIA DGX Spark (GB10, 128 GB unified memory). MTP speculative decoding is on by default (`--spec auto`); text + vision enabled (vision tower BF16):

```bash
REXL3_MEMORY_FLOOR_GIB=1 rexl3-serve --model DIR --slots 1 --max-model-len 262144
```

`DIR` is the directory holding this pack (or a local snapshot of this repo).

## Load

ExLlamaV3, point at this repo (or a local snapshot):

```python
from exllamav3 import Config, Model, Tokenizer

cfg = Config.from_directory("vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw")
model = Model.from_config(cfg)
model.load()
tokenizer = Tokenizer.from_config(cfg)
```

## About SAGE

SAGE is our own mixed-precision quantization method for EXL3. The method is not published. If a tensor is EXL3, its `K` is self-describing on the trellis.

## Source model (Z.ai, not this pack)

GLM-5.3-Flash is Z.ai's natively multimodal model: 320B total parameters, ~18B active, hybrid sparse + linear attention with manifold-constrained hyper-connections. Architecture, sampling defaults, and any benchmark rows on the [upstream card](https://huggingface.co/zai-org/GLM-5.3-Flash) are **Z.ai's**.

Upstream sampling defaults: `temperature=1.0`, `top_p=0.95`; `reasoning_effort` defaults to `max`. This pack's `generation_config.json` carries the upstream defaults.

## License

MIT, same as [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). Cite Z.ai for the base model. This repository is the EXL3 pack only.
