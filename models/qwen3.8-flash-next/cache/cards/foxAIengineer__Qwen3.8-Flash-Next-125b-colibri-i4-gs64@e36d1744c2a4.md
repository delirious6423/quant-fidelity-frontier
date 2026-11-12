---
base_model:
  - Qwen/Qwen3.8-Flash-Next
tags:
  - colibri
  - qwen3.8
  - qwen3.8-flash-next
  - moe
  - i4
library_name: colibri
---

Qwen3.8-Flash-Next converted for the Colibri runtime.

Architecture:
- Qwen3.8-Flash-Next
- 125B parameters
- i4 quantization
- group size 64
- full Gated DeltaNet support

Runtime:
- Colibri
- qwen38.c

Tensor layout:
- DeltaNet weights: `model.layers.{i}.linear_attn.*`

Engine:
https://github.com/JustVugg/colibri

Source model:
https://huggingface.co/Qwen/Qwen3.8-Flash-Next

> **Note:** Fresh upload. External download validation has not yet been completed.
> If you encounter missing files, corrupted shards, or download issues, please open a discussion in the Community tab.