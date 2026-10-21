---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
  - mlx
---

# z-lab/Qwen3.8-27B-PARO

**Pairwise Rotation Quantization for Efficient Reasoning LLM Inference**

<p>
  <a href="https://arxiv.org/abs/2511.10645"><img src="https://img.shields.io/badge/arXiv-2511.10645-b31b1b.svg" alt="Paper"></a>
  <a href="https://paroquant.z-lab.ai"><img src="https://img.shields.io/badge/Blog-ParoQuant-blue" alt="Blog"></a>
  <a href="https://huggingface.co/collections/z-lab/paroquant"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow" alt="Models"></a>
  <a href="https://pypi.org/project/paroquant/"><img src="https://img.shields.io/pypi/v/paroquant" alt="PyPI"></a>
</p>

ParoQuant is the state-of-the-art INT4 quantization for LLMs. It closes the accuracy gap with FP16 while running at near-AWQ speed. Supports NVIDIA GPUs (vLLM, Transformers) and Apple Silicon (MLX). For more information, see https://github.com/z-lab/paroquant.

z-lab/Qwen3.8-27B-PARO is a 4-bit [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized with ParoQuant. Check out other ParoQuant models from the Hugging Face [collection](https://huggingface.co/collections/z-lab/paroquant).


## Quick Start

### oMLX (Apple Silicon)

On Apple Silicon, we recommend serving ParoQuant models using [oMLX](https://github.com/jundot/omlx). Refer to the [docs](https://omlx.ai) for more details.

### Installation

```bash
# NVIDIA GPU (CUDA 12.9)
pip install "paroquant[vllm]"

# NVIDIA GPU (CUDA 13.0)
pip install "paroquant[vllm]" "vllm==0.19.1" \
  --extra-index-url https://wheels.vllm.ai/0.19.1/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130

# Apple Silicon
pip install "paroquant[mlx]"
```

### Interactive Chat

```bash
python -m paroquant.cli.chat --model z-lab/Qwen3.8-27B-PARO
```

### OpenAI-Compatible API Server

For vLLM, you can directly use `vllm serve` to serve ParoQuant models:

```bash
vllm serve z-lab/Qwen3.8-27B-PARO --port 8000
```

For other frameworks:

```bash
python -m paroquant.cli.serve --model z-lab/Qwen3.8-27B-PARO --port 8000
```

For MLX, add `--vlm` if you wish to load the VLM components and use the model's multimodal features. For vLLM, VLM components are loaded by default and can be skipped with the server argument `--language-model-only`.

> [!NOTE]
> The visual components in this checkpoint is stored in original precision, and only the language components are quantized to 4 bits; as a result, the model size is larger than a fully-quantized model. Avoid loading the VLM components if you are not using the multimodal features for the best efficiency.

### Docker (NVIDIA GPU)

> [!NOTE]
> The following commands map the local cache directory to the container in order to persist kernel cache across runs. Remove `-v ...` to disable this behavior.

```bash
# Interactive chat
docker run --pull=always --rm -it --gpus all --ipc=host \
  -v $HOME/.cache/paroquant:/root/.cache/paroquant \
  ghcr.io/z-lab/paroquant:chat --model z-lab/Qwen3.8-27B-PARO

# API server (port 8000)
docker run --pull=always --rm -it --gpus all --ipc=host -p 8000:8000 \
  -v $HOME/.cache/paroquant:/root/.cache/paroquant \
  ghcr.io/z-lab/paroquant:serve --model z-lab/Qwen3.8-27B-PARO
```

## Citation

```bibtex
@inproceedings{liang2026paroquant,
  title     = {{ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference}},
  author    = {Liang, Yesheng and Chen, Haisheng and Zhang, Zihan and Han, Song and Liu, Zhijian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```