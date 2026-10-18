---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - modelopt
---

# DeepSeek-V4.1-Flash-NVFP4

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="50%" alt="DeepSeek" />
  <p><a href="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash">Original Model</a> · <a href="https://github.com/NVIDIA/Model-Optimizer">Model Optimizer</a> · <a href="LICENSE">MIT License</a></p>
</div>

## Introduction

This is a community NVFP4 conversion of [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), published by [s-zaizen](https://huggingface.co/s-zaizen). It is not an official DeepSeek or NVIDIA release.

The original model is a multimodal Mixture-of-Experts model featuring a Causal Encoder-Decoder architecture, Compressed Sparse Attention 2, Engram conditional memory, and DSpark speculative decoding. This conversion preserves its architecture, tokenizer, vision components, and native context configuration.

| Item | Details |
| :--- | :--- |
| Source | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Source revision | `fb2764a5cf321eaa5070ca8f9e892818f477c16d` |
| Backbone / Engram parameters | 552B / 196B, as reported by DeepSeek |
| Native context configuration | 1M tokens; not independently evaluated here |
| Routed expert weights | NVFP4, 16-element blocks |
| Weight files | 48 safetensors shards, 527.293 GB total |

## Quantization

Converted using [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/079078de9d3cd26fa5baafdb754bba29dde24f7d) and its MXFP4-to-NVFP4 weight-cast routine. Activation scaling follows NVIDIA's published [calibration-free `input_scale1` policy](https://github.com/NVIDIA/Model-Optimizer/blob/079078de9d3cd26fa5baafdb754bba29dde24f7d/modelopt_recipes/ptq.md): the global input scale is fixed to **1.0**, and per-block E4M3 scales are dynamic at inference. These constants are not measured calibration statistics.

Only the backbone's routed experts are converted. Engram, attention, shared experts, vision, and MTP/DSpark weights retain their source formats. No calibration forward pass or model-specific inference patch is used to produce the checkpoint.

## Evaluation Results

No quality or throughput benchmarks were run for this release. File integrity, tensor layouts, scale validity, and unchanged tensors were checked. These checks do not establish benchmark parity or runtime compatibility. DeepSeek's [original evaluation results](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash#evaluation-results) describe the source model, not this conversion.

## Prompt Encoding

The source's [prompt encoding implementation](encoding/README.md) is included. Use DeepSeek V4.1's encoding rather than assuming a generic Jinja chat template. See [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) for the upstream protocol toolkit.

## Minimal Inference

Download the checkpoint:

```bash
hf download s-zaizen/DeepSeek-V4.1-Flash-NVFP4 --local-dir DeepSeek-V4.1-Flash-NVFP4
```

Inference requires a runtime supporting **DeepSeek V4.1 and mixed FP8/NVFP4 MoE checkpoints**. The included upstream reference inference code is for the source format; it is not a validated NVFP4 serving recipe. No working vLLM or SGLang launch command is claimed here.

## License

The original DeepSeek [MIT license](LICENSE) is retained for the model and accompanying source files.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) — original model and inference code. [NVIDIA](https://github.com/NVIDIA/Model-Optimizer) — quantization tooling and numerical recipes. [s-zaizen](https://huggingface.co/s-zaizen) — NVFP4 conversion and publication.
