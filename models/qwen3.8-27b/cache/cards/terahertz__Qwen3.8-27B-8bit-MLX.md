---
base_model: Qwen/Qwen3.8-27B
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- qwen
- qwen3.8
- quantized
- 8bit
- vision-language
---

# Qwen3.8-27B-8bit-MLX

MLX 8-bit quantization of the official **Qwen/Qwen3.8-27B** model, optimized for inference on Apple Silicon.

## Model Details

| Property | Value |
|---|---|
| Base model | Qwen/Qwen3.8-27B |
| Parameters | 27.78B |
| Model type | Vision-Language Model |
| Original precision | BF16 |
| Quantization | MLX affine 8-bit |
| Group size | 64 |
| Effective quantization | 8.627 bits/weight |
| Native context | 262,144 tokens |
| License | Apache-2.0 |

Qwen3.8-27B is a dense multimodal model supporting text, images and video. It uses a hybrid architecture combining Gated DeltaNet and gated attention layers.

The model supports thinking and non-thinking modes and is designed for coding, scientific reasoning, agentic workloads, long-horizon tasks, and multimodal understanding.

## Quantization

This repository was generated from the official BF16 weights using `mlx-vlm`.

Conversion parameters:

```bash
mlx_vlm.convert \
  --hf-path Qwen/Qwen3.8-27B \
  --mlx-path Qwen3.8-27B-8bit-MLX \
  --quantize \
  --q-bits 8 \
  --q-group-size 64
```

The conversion reported:

```text
Quantized model with 8.627 bits per weight.
```

The effective bit rate is slightly higher than 8 bits/weight because not every tensor is quantized and quantization metadata such as scales and biases must also be stored.

No fine-tuning, merging, pruning, or other modification of the model weights was performed beyond MLX quantization.

## Installation

Install MLX and MLX-VLM:

```bash
pip install -U mlx mlx-vlm
```

## Usage

Run the model directly from Hugging Face:

```bash
mlx_vlm.generate \
  --model terahertz/Qwen3.8-27B-8bit-MLX \
  --prompt "Explain the advantages of FMCW radar for terahertz imaging." \
  --max-tokens 2048
```

## Context Length

The upstream Qwen3.8-27B model natively supports a context length of up to **262,144 tokens**.

Qwen also documents extension up to 1M tokens using YaRN in inference engines that support it.

Actual usable context on Apple Silicon depends on available unified memory, KV-cache requirements, inference framework, and other running applications.

## Apple Silicon

This quantization is intended for MLX inference on Apple Silicon Macs.

Compared with the original BF16 checkpoint, 8-bit quantization substantially reduces memory requirements while retaining significantly more numerical precision than more aggressive 4-bit or 5-bit quantizations.

It is therefore particularly suitable for Macs with sufficient unified memory where model fidelity is prioritized over minimum model size.

## Multimodal Capabilities

The original Qwen3.8-27B is a native vision-language model supporting:

- Text
- Images
- Video
- Document understanding
- Scientific charts and diagrams
- Multimodal agentic tasks

Availability of individual multimodal features depends on the version and capabilities of MLX-VLM used for inference.

## Important Notes

This is an **unofficial community quantization**.

It is not an official Qwen release and is not affiliated with or endorsed by the Qwen Team.

Model capabilities, limitations, safety characteristics, recommended generation parameters, and benchmark results should be considered those of the upstream model unless specifically evaluated again on this quantized version.

No claim is made that this quantized model exactly reproduces the benchmark scores of the original BF16 model.

## Original Model

For the complete model description, architecture, benchmarks, recommended sampling parameters, multimodal examples, long-context configuration, and limitations, see:

https://huggingface.co/Qwen/Qwen3.8-27B

## License

The original Qwen3.8-27B model is distributed under the **Apache License 2.0**.

This quantized version is distributed under the same license.

Please consult the upstream repository for the original model license and associated information.

## Acknowledgements

All credit for the model architecture, training, post-training, and original weights belongs to the **Qwen Team**.

Quantization was performed using **MLX / MLX-VLM** for efficient inference on Apple Silicon.