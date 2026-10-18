---
license: mit
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
tags:
  - deepseek-v4
  - nvfp4
  - modelopt
  - moe
---

# DeepSeek-V4-Flash-0731 NVFP4

An independently calibrated NVIDIA NVFP4 export of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

The 43 routed-expert layers (256 experts each) use NVFP4 weights and calibrated
activation scales. Attention, shared experts, router, embeddings, output head,
and the attached speculative/MTP module retain their source formats. The
source MXFP4 expert values were converted losslessly to NVFP4 representation;
activation scales were calibrated separately.

## Calibration

- NVIDIA ModelOpt, eight H100 80 GB GPUs
- 128 samples at up to 512 tokens: 32 each from CNN/DailyMail, OpenCodeReasoning,
  OpenMathReasoning, and Magpie-Pro-MT-300K
- 65,536 padded sequence positions
- Max calibration algorithm, batch size 1

See `CONVERSION.md` for exact revisions, commands, export statistics, and audit
results. This repository contains the Hugging Face NVFP4 checkpoint, not GGUF.
Runtime support for this DeepSeek V4 NVFP4 layout is required.

The original model's MIT license is included. Model architecture, usage, chat
format, and limitations are documented in the
[official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
