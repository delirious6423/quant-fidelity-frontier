---
pipeline_tag: text-generation
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
license: mit
library_name: Model Optimizer
tags:
- mmangkad
- ModelOpt
- DeepSeekV4
- quantized
- NVFP4
- nvfp4
---

# Model Overview

## Description:
The mmangkad DeepSeek-V4-Flash-0731-NVFP4 model is a quantized version of DeepSeek AI's DeepSeek-V4-Flash-0731 model, an autoregressive Mixture-of-Experts language model that uses an optimized Transformer architecture with hybrid attention (Compressed Sparse Attention and Heavily Compressed Attention) and Manifold-Constrained Hyper-Connections. For more information, refer to the [DeepSeek-V4-Flash-0731 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). The mmangkad DeepSeek-V4-Flash-0731-NVFP4 model is quantized with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).

This model is ready for commercial/non-commercial use.  <br>

## Third-Party Community Consideration
This model is not owned or developed by mmangkad. This model was quantized by mmangkad; see the original [DeepSeek-V4-Flash-0731 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

## References
NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer

### License/Terms of Use:
[MIT](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE)

### Deployment Geography:
Global <br>

### Use Case:
DeepSeek V4 is well-suited for advanced reasoning, agentic AI applications, tool use scenarios, and complex problem-solving in domains such as mathematics, software engineering, and enterprise AI assistants.  <br>

### Release Date:
Hugging Face 08/02/2026 via https://huggingface.co/mmangkad/DeepSeek-V4-Flash-0731-NVFP4 <br>

## Model Architecture:
**Architecture Type:** Transformers  <br>
**Network Architecture:** Mixture-of-Experts (MoE) with Hybrid Attention (Compressed Sparse Attention + Heavily Compressed Attention) <br>
**Number of Model Parameters:** 284B in total and 13B activated <br>

## Input:
**Input Type(s):** Text <br>
**Input Format(s):** String <br>
**Input Parameters:** 1D (One-Dimensional): Sequences <br>
**Other Properties Related to Input:** Supports multi-turn conversations with system prompts, user messages, and assistant responses. Maximum context length of 1 million tokens. Uses a custom encoding pipeline (`encoding_dsv4`) with three reasoning modes: low, high, and max. <br>

## Output:
**Output Type(s):** Text <br>
**Output Format:** String <br>
**Output Parameters:** One-Dimensional (1D): Sequences <br>
**Other Properties Related to Output:** Supports structured JSON output, function/tool calling, and reasoning content when enabled. <br>

Models published by mmangkad are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA hardware (e.g. GPU cores) and software frameworks (e.g. CUDA libraries), the model can achieve faster inference than on CPU-only systems. <br>

## Software Integration:
**Supported Runtime Engine(s):** <br>
* SGLang <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Blackwell <br>

**Preferred Operating System(s):** <br>
* Linux <br>

The integration of foundation and quantized models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
The model is DeepSeek-V4-Flash-0731 NVFP4 quantized with NVIDIA Model Optimizer **0.46.0rc0-12-g9d360af34** (commit [`9d360af`](https://github.com/NVIDIA/Model-Optimizer/commit/9d360af34f907c8d64544797a803132a9a5d502d)).  <br>

## Training and Evaluation Datasets:

## Calibration Dataset:
**Link:** [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) <br>
**Data Collection Method by dataset:** Automated. <br>
**Labeling Method by dataset:** Automated. <br>
**Properties:** The cnn_dailymail dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The Nemotron-Post-Training-Dataset-v2 is a post-training dataset curated by NVIDIA containing multi-turn conversations across diverse topics. <br>

## Training Datasets:
**Data Modality:** Undisclosed <br>
**Data Collection Method by dataset:** Undisclosed <br>
**Labeling Method by dataset:** Undisclosed<br>
**Data Size:** Undisclosed <br>
**Properties:** Undisclosed

## Post Training Quantization
This model was obtained by quantizing the weights and activations of DeepSeek-V4-Flash-0731 to NVFP4 data type, ready for inference with SGLang. Only the weights and activations of the linear operators within transformer blocks in MoE are quantized.

## Usage

### Deploy with SGLang

The integration auto-detects NVFP4 from the checkpoint's `hf_quant_config.json` (weights are stored in FP8 with `"moe_quant_algo": "NVFP4"`):

Example for a single **GB300** node:

```bash
sglang serve --model-path mmangkad/DeepSeek-V4-Flash-0731-NVFP4 --tensor-parallel-size 4 --trust-remote-code
```

## Model Limitations:
The base model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

## Ethical Considerations

mmangkad believes trustworthy AI is a shared responsibility. Developers should work with their internal model team to ensure this model meets the requirements of their industry and use case and addresses unforeseen misuse.

Users are responsible for model inputs and outputs. Users are responsible for ensuring safe integration of this model, including implementing guardrails as well as other safety mechanisms, prior to deployment.

Please report model quality, risk, or security concerns through the [model repository discussions](https://huggingface.co/mmangkad/DeepSeek-V4-Flash-0731-NVFP4/discussions).