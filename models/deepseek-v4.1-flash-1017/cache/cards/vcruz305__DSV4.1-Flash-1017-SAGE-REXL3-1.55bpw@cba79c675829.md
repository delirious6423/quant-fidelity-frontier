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

EXL3 / REXL3 pack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
quantized by [vcruz305](https://huggingface.co/vcruz305) with **SAGE**, our own mixed-precision quantization method for EXL3.
The method is not published.

- GPU-resident footprint: ~108.5 GiB text / ~109.4 GiB VLM (fits a single 128 GB-class unified-memory device).
- Files: 17 safetensors shards + `model.safetensors.index.json`.

## Load

ExLlamaV3, pointed at this repo (or a local snapshot):

```python
from exllamav3 import Config, Model, Tokenizer

cfg = Config.from_directory("vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw")
model = Model.from_config(cfg)
model.load()
tokenizer = Tokenizer.from_config(cfg)
```

## Sampling

Upstream recommended sampling: `temperature=1.0`, `top_p=1.0`. This pack's `generation_config.json` sets `top_p=0.95`; set `top_p` explicitly to follow upstream.

## License

MIT, same as [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). Cite DeepSeek for the base model.
