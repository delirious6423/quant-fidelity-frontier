---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
gated: true
extra_gated_prompt: "By requesting access you agree to use this model under its Apache 2.0 license and to cite the base model. Requests are reviewed manually."
extra_gated_fields:
  I agree to use this model under the Apache license, consistent with the base model's terms: checkbox
---

# Qwen3.8-27B — SAGE REXL3 4.50 bpw

Loadable [ExLlamaV3](https://github.com/turboderp-org/exllamav3) / REXL3 pack of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

Quantized by [vcruz305](https://huggingface.co/vcruz305) with **SAGE**, our own
mixed-precision quantization method for EXL3. Tensors use different EXL3 `K`
values; the kernels read `K` from each trellis tensor at load.

Sized for a single 24 GB GPU: `~16.1 GiB` GPU-resident weights (`17.3 GB` pack,
text-only), with full 262,144-token context (`q4` KV) and DFlash2 speculative
decoding ([`vcruz305/Qwen3.8-27B-DFlash2-SAGE-REXL3`](https://huggingface.co/vcruz305/Qwen3.8-27B-DFlash2-SAGE-REXL3)).

## Measured quality

Teacher-forced token-level agreement with Qwen's official BF16 release (`Qwen/Qwen3.8-27B`), full-vocabulary KL divergence per token, on held-out sequences:

| Metric | Value |
| --- | --- |
| Top-1 Agreement | 96.82% |
| Mean KLD (vs BF16) | 0.0117 |

## Pack

| Item | Value |
| --- | --- |
| Format | EXL3 / REXL3 (trellis) |
| Profile | **4.50 bpw** SAGE mixed-K |
| Resident footprint | ~16.1 GiB (text-only) |
| Shards | 2 (`model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`) |
| Index | `model.safetensors.index.json` |
| `quant_method` | `exl3` |
| `model_type` | `qwen3_5` |

## Load

ExLlamaV3, point at this repo (or a local snapshot):

```python
from exllamav3 import Config, Model, Tokenizer

cfg = Config.from_directory("vcruz305/Qwen3.8-27B-SAGE-REXL3-4.50bpw")
model = Model.from_config(cfg)
model.load()
tokenizer = Tokenizer.from_config(cfg)
```

## About SAGE

SAGE is our own mixed-precision quantization method for EXL3. The method is not published. If a tensor is EXL3, its `K` is self-describing on the trellis.

## Source model

Qwen3.8-27B is Alibaba's 27B dense text model, with a 262,144-token context. Architecture details, sampling defaults, and any benchmark rows on the [upstream card](https://huggingface.co/Qwen/Qwen3.8-27B) are **Qwen's**.

## License

Apache 2.0, same as [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Cite Qwen for the base model. This repository is the EXL3 pack only.
