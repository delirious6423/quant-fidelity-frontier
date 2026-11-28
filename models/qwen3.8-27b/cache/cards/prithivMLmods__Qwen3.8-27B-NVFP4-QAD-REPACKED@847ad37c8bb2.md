---
license: other
license_name: nvidia-open-model-license
license_link: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD/blob/main/LICENSE
library_name: transformers
tags:
- mtp
- text-generation-inference
- REPACKED
- Model-Optimizer
- nvidia
- ModelOpt
- Qwen3.8
- quantized
- nvfp4
- QAD
language:
- en
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
datasets:
- nvidia/Nemotron-Cascade-2-SFT-Data
---

# **Qwen3.8-27B-NVFP4-QAD-REPACKED**

> **Qwen3.8-27B-NVFP4-QAD-REPACKED**, based on NVIDIA's **Qwen3.8-27B-NVFP4-QAD** (`nvidia/Qwen3.8-27B-NVFP4-QAD`), is an MTP-repacked and issues-fixed variant of NVIDIA's NVFP4-quantized multimodal Qwen3.8-27B model. Originally quantized using NVIDIA Model Optimizer (v0.48.0) with Quantization-Aware Distillation (QAD), this repack preserves the original NVFP4 quantization while incorporating Multi-Token Prediction (MTP) support for speculative decoding and addressing compatibility and inference-related issues. The model retains its multimodal capabilities across text, image, and video inputs, along with the original architecture and quantized weights, while providing an improved, inference-ready GGUF distribution for local deployment.

> [!NOTE]
> MTP — Speculative Decoding: Yes

> [!NOTE]
> Qwen3.8-27B-NVFP4-QAD-GGUF — https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-GGUF

## Model Files

| File Name | File Type | File Size | File Link | Description |
|-----------|-----------|-----------|-----------|-------------|
| chat_template.jinja | Jinja Template | 8.95 kB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/chat_template.jinja) | Chat template for formatting conversations and prompts. |
| config.json | JSON | 102 kB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/config.json) | Core model architecture and configuration parameters. |
| generation_config.json | JSON | 244 Bytes | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/generation_config.json) | Default text generation parameters (e.g., temperature, top_p). |
| hf_quant_config.json | JSON | 67.8 kB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/hf_quant_config.json) | Configuration file detailing the quantization parameters (e.g., NVFP4 settings). |
| merges.txt | Text | 3.35 MB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/merges.txt) | Byte-Pair Encoding (BPE) merge rules for the tokenizer. |
| model-00001-of-00004.safetensors | Safetensors | 5.98 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/model-00001-of-00004.safetensors) | Model weight shard 1/4. **Required** to load the model. |
| model-00002-of-00004.safetensors | Safetensors | 5.96 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/model-00002-of-00004.safetensors) | Model weight shard 2/4. **Required** to load the model. |
| model-00003-of-00004.safetensors | Safetensors | 5.98 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/model-00003-of-00004.safetensors) | Model weight shard 3/4. **Required** to load the model. |
| model-00004-of-00004.safetensors | Safetensors | 2.09 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/model-00004-of-00004.safetensors) | Model weight shard 4/4. **Required** to load the model. |
| model.safetensors.index.json | JSON | 228 kB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/model.safetensors.index.json) | Index file mapping model weights to their respective shards. |
| preprocessor_config.json | JSON | 443 Bytes | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/preprocessor_config.json) | Configuration for preprocessing vision/image inputs. |
| tokenizer.json | JSON | 12.8 MB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/tokenizer.json) | Complete tokenizer vocabulary and encoding rules. |
| tokenizer_config.json | JSON | 17.9 kB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/tokenizer_config.json) | Tokenizer-specific configuration parameters. |
| video_preprocessor_config.json | JSON | 385 Bytes | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/video_preprocessor_config.json) | Configuration for preprocessing video inputs. |
| vocab.json | JSON | 6.72 MB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-REPACKED/blob/main/vocab.json) | Tokenizer vocabulary mapping tokens to IDs. |

## License and Attribution

This model is based on and/or incorporates the following open-source projects and models:

* **Qwen3.8-27B:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
* **Qwen3.8-27B-NVFP4-QAD:** [nvidia/Qwen3.8-27B-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD)

This model is released under the [NVIDIA Open Model License](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD/blob/main/LICENSE).