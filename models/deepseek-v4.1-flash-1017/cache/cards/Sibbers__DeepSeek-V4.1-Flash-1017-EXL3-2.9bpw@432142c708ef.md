---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
tags:
  - exl3
  - deepseek
  - moe
---

# DeepSeek-V4.1-Flash-1017 — EXL3 2.9 bpw

**This is a copy, not our work.** It is a verified byte-identical mirror of
[easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
— an EXL3 quantisation of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).
All 58 weight and config files match the original's published hashes exactly. **Use the original repo**; this
copy exists as a stable snapshot pinned to the exact bytes our TensorFold Metal-lane performance receipts were
measured on, and for download convenience.

- **Format:** EXL3, `mul1` codebook, effective 2.90 bits per weight; MTP weights at 4 bits, head at 6 bits. Full details in `quantization_config.json`.
- **Size:** ~211 GB in 39 safetensors shards. Tokenizer, config and chat template included.
- **Engram tables:** not included — this model serves them from the official release's shards 47 and 48. A hash-verified convenience copy of those two shards is at [Sibbers/DeepSeek-V4.1-Flash-1017-orig-shards-47-48](https://huggingface.co/Sibbers/DeepSeek-V4.1-Flash-1017-orig-shards-47-48); the canonical source is [DeepSeek's own release](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
- **License:** MIT, inherited from the DeepSeek release. Credit for the quantisation belongs to easthambuilds.
