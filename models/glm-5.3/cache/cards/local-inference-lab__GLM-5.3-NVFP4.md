---
language:
- en
- zh
library_name: transformers
license: mit
pipeline_tag: text-generation
base_model:
- zai-org/GLM-5.3
---

## Model Description

**GLM-5.3-NVFP4** is an NVFP4-quantized version of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), a 744B-parameter Mixture-of-Experts language model with 40B active parameters, 256 experts per MoE layer (8 activated per token), and DeepSeek Sparse Attention (DSA).

Quantized directly from the full BF16 checkpoint ([zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), *not the FP8 release*, to NVFP4 (4-bit with blockwise FP8 scales per 16 elements) using [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

### What's quantized

Only the *non-shared* MoE expert MLP projections are quantized to NVFP4. Attention weights are left in BF16, in addition to the dense MLPs (layers 0-3) and the shared experts. Since the MoE expert weights constitute the vast majority of model parameters in an MoE architecture, this still yields significant memory savings.

Calibration uses natural top-k routing rather than forcing all experts to activate, so each expert's quantization scales reflect the token distributions it actually sees during inference. To compensate, calibration was run on a much larger number of samples than typical to ensure broad expert coverage through natural routing alone. 

### Calibration dataset

Three calibration passes were run:

1. **Coding pass** — Agentic coding samples (tool calling, multi-turn code generation, function calling) with English and Chinese system prompts.
2. **Broad pass** — Large-scale diverse samples drawn from WildChat-NonToxic and LMSYS-Chat covering real user conversations across a wide range of topics and languages.
3. **Deep pass** — Long-context samples (>8K tokens) from coding and diverse sources to exercise deep-sequence expert activation patterns.

### Requirements

Hardware: 8x RTX PRO 6000 Blackwell 96GB (b12x MoE runner recommended)

### Community Testing / Instructions

https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v17.md
