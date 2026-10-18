---
license: mit
library_name: sglang
base_model:
  - deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - deepseek-v4
  - fp8
  - quantized
  - dspark
---

# DeepSeek-V4-Flash-0731-FP8-DSpark

FP8 re-packaging of `deepseek-ai/DeepSeek-V4-Flash-0731` for Hopper (H200/H100).

Routed MoE experts converted MXFP4/FP4 -> block FP8 via lossless
`cast_e2m1fn_to_e4m3fn` (`tmp/scripts/convert_hf_mxfp4_experts_to_fp8.py`).
`expert_dtype` removed from `config.json`; other fields (including `dspark_*` if
present) kept.

On Hopper set `SGLANG_DSV4_FP4_EXPERTS=0` when serving.
