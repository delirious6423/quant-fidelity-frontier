---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: 
  - Qwen/Qwen3.8-27B
  - mlx-community/Qwen3.8-27B-bf16
tags:
  - mlx
  - vision
  - image-to-text
  - text-generation
  - 4-bit
  - nvfp4
  - 27b
num_parameters: 27780000000
---

# Qwen3.8-27B NVFP4 (MLX)

This model was converted to MLX format from [`mlx-community/Qwen3.8-27B-bf16`](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16)
using mlx-vlm version **0.6.13**.
Refer to the [original model card](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16) for more details on the model.


<p align="center">
  <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/MLX-Apple_Silicon-black?style=for-the-badge&logo=apple" alt="MLX"></a>
  <a href="https://huggingface.co/zherebetskyy/Qwen3.8-27B-4bit-mlx"><img src="https://img.shields.io/badge/Q4_4--bit-Variant-blue?style=for-the-badge" alt="Q4 Variant"></a>
  <a href="https://huggingface.co/zherebetskyy/Qwen3.8-27B-6bit-mlx"><img src="https://img.shields.io/badge/Q6_6--bit-Variant-green?style=for-the-badge" alt="Q6 Variant"></a>
  <a href="https://huggingface.co/zherebetskyy/Qwen3.8-27B-nvfp4-mlx/tree/main"><img src="https://img.shields.io/badge/License-Apache_2.0-yellow.svg?style=for-the-badge" alt="License"></a>
</p>

This repository contains **Qwen3.8-27B** quantized to **NVFP4 (4-bit Floating-Point with Normalization/Scaling)** for Apple Silicon devices using [MLX](https://github.com/ml-explore/mlx).

---

## 🚀 Why NVFP4 over Standard INT4 / Q4?

Traditional **INT4 (Q4)** quantization uniformly discretizes weights into integers. While memory-efficient, integer quantization often suffers from noticeable perplexity degradation—especially in large-scale multimodal models with non-uniform weight distributions and sensitive activations.

### Key Advantages of NVFP4:

1. **Logarithmic/Dynamic Range:** NVFP4 utilizes a dynamic 4-bit floating-point format (E2M1/E3M0 structure) rather than fixed integer steps. This represents smaller parameters with higher precision and maintains wider dynamic range for outlier activation scales.
2. **Superior Perplexity & Coherence:** Retains near-BF16 reasoning accuracy, coding proficiency, and complex context tracking compared to standard INT4/Q4 quantizations.
3. **Optimized Memory Footprint:** Delivers performance almost equivalent to **Q6**, while keeping memory consumption close to **Q4** (~18–19 GB VRAM).
4. **Enhanced Multimodal Preservation:** Crucial for vision-language models where visual projection layers are particularly sensitive to coarse integer rounding.

---

## 📊 Quantization Comparison & Selection Guide

| Model Variant | Quantization | Size | Target VRAM | Quality / Perplexity | Recommended Use Case |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **NVFP4 (This Repo)** | **NVFP4** | **~18.5 GB** | **≥ 24 GB** | ⭐⭐⭐⭐✨ | **Best Quality/Speed balance** for local inference |
| [Qwen3.8-27B-4bit-mlx](https://huggingface.co/zherebetskyy/Qwen3.8-27B-4bit-mlx) | INT4 (Q4) | ~16.8 GB | ≥ 24 GB | ⭐⭐⭐☆☆ | Maximum memory efficiency / constrained Unified Memory |
| [Qwen3.8-27B-6bit-mlx](https://huggingface.co/zherebetskyy/Qwen3.8-27B-6bit-mlx) | INT6 (Q6) | ~24.5 GB | ≥ 32 GB | ⭐⭐⭐⭐⭐ | High-precision tasks when extra VRAM is available |
| [Qwen3.8-27B-mxfp8-mlx](https://huggingface.co/zherebetskyy/Qwen3.8-27B-mxfp8-mlx) | **MXFP8** | **~28.5 GB** | **≥ 36 GB** | ⭐⭐⭐⭐⭐ | **Near-lossless precision** for complex reasoning, coding & visual details |


---

## 💻 Hardware Requirements

- **Processor:** Apple Silicon (M1/M2/M3/M4 Max or Ultra recommended).
- **Unified Memory:** 
  - **Minimum:** 24 GB (for short contexts).
  - **Recommended:** 36 GB+ (for long-context multimodal processing and fast generation).

---

## 🛠️ Quickstart Guide

### 1. Python API (`mlx-vlm`)

Install or upgrade `mlx-vlm`:
```bash
pip install -U mlx-vlm

```

```bash
python -m mlx_vlm.generate --model zherebetskyy/Qwen3.8-27B-nvfp4-mlx --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```

#### Run vision-language inference in Python:

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load model and processor
model_id = "zherebetskyy/Qwen3.8-27B-nvfp4-mlx"
model, processor = load(model_id)
config = load_config(model_id)

# Input image and prompt
image = ["[http://images.cocodataset.org/val2017/000000039769.jpg](http://images.cocodataset.org/val2017/000000039769.jpg)"]
prompt = "Analyze this image and describe the key elements in detail."

# Generate output
formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
output = generate(model, processor, formatted_prompt, image, max_tokens=500, temperature=0.2)

print(output)

```

---

### 2. Command Line Interface (CLI)

```bash
python -m mlx_vlm.generate \
  --model zherebetskyy/Qwen3.8-27B-nvfp4-mlx \
  --max-tokens 500 \
  --temperature 0.2 \
  --prompt "Describe this image." \
  --image <path_to_image.jpg>

```

---

### 3. Running as a Local Server (OpenAI-Compatible)

Start an `mlx_lm` / `mlx_vlm` local OpenAI-compatible API server:

```bash
# Install tool
uv tool install mlx-lm

# Launch local server
mlx_lm.server --model "zherebetskyy/Qwen3.8-27B-nvfp4-mlx" --port 8080

```

You can now connect local agent frameworks (e.g., Pi, OpenClaw, Hermes, or LM Studio) to `http://localhost:8080/v1`.

---

## 📜 License

This model is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0), matching the original base release.

## 🙏 Acknowledgments & References

* Base Model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
* Quantized with [MLX VLM](https://www.google.com/search?q=https://github.com/ml-explore/mlx-examples/tree/main/mlx_vlm) by Apple.
* Check out alternative MLX variants: [Qwen3.8-27B 4-bit (Q4)](https://huggingface.co/zherebetskyy/Qwen3.8-27B-4bit-mlx) and [Qwen3.8-27B 6-bit (Q6)](https://huggingface.co/zherebetskyy/Qwen3.8-27B-6bit-mlx) and [Qwen3.8-27B 8-bit (Q8)] (https://huggingface.co/zherebetskyy/Qwen3.8-27B-mxfp8-mlx).
```

```

