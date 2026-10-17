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
  - 6-bit
  - 27b
num_parameters: 27780000000
---

# Qwen3.8-27B-6bit-mlx

This model was converted to MLX format from [`mlx-community/Qwen3.8-27B-bf16`](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16)
using mlx-vlm version **0.6.13**.
Refer to the [original model card](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16) for more details on the model.

## Quick use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model zherebetskyy/Qwen3.8-27B-6bit-mlx --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```

---

# Details

## What's this?
**Qwen3.8-27B** is an advanced 27B-parameter dense model from Alibaba Cloud. It features a hybrid linear/full attention architecture (3:1 ratio across 64 layers) that merges highly efficient DeltaNet-style linear attention with standard full softmax attention layers. It natively supports a 262K context window, multimodal vision, video parsing, and multi-token prediction (MTP).

This repository hosts a **6-bit linear affine group-quantized conversion** (group size 64) optimized specifically for Apple Silicon hardware. Averaging roughly 6.6 bits/weight, this model card layout compresses the 27B structure to fit comfortably into **21–23 GB of memory**—leaving ideal unified memory headroom for long-context execution and media workloads on 64GB Mac systems.

<details>
<summary>🎛️ Architecture Details</summary>

| Spec | Value |
|---|---|
| **Total Parameters** | 27.8B (dense, all active) |
| **Layer Configuration** | 64 layers (3x linear attention + 1x full attention, 16 repetitions) |
| **Standard Attention** | 24 Q heads, 4 KV heads (GQA), head_dim 256 |
| **Linear Attention** | 16 QK heads, 48 V heads, head_dim 128 |
| **Context Length** | 262K native, up to 1M+ using YaRN scales |
| **Vocabulary Size** | 248K tokens |
| **Native Modalities** | Text, Image, Video |
| **Model Type Tag** | `qwen3_8` |
</details>

---

## Conversion Details
* **Source Weights:** `mlx-community/Qwen3.8-27B-bf16` (extracted from the raw FP16/BF16 base distribution)
* **Quantization Format:** Uniform MLX Group Affine 6-bit
* **Vision Tower Precision:** 100% retained. The vision tower parameters remain unquantized to guarantee maximum feature parsing and image description capability.
* **Tool Calling Support:** Enhanced via a fixed Jinja template that patches runtime exceptions on C++ execution engines (LM Studio, MLX Local Server) and correctly handles `developer` system roles.

---

## ⚡ Quick Start

For fast, isolated package handling without polluting your system Python env, we recommend using **`uv`** (a blazing-fast installer written in Rust).

### 1. Unified Command Line (CLI)

#### Multimodal Image Prompt (Vision)
```bash
uv run --with mlx-vlm mlx_vlm.generate \
  --model zherebetskyy/Qwen3.8-27B-6bit-mlx \
  --image /path/to/your_image.jpg \
  --prompt "Describe the layout and contents of this image in detail."
```

#### Standard Text Generation
```bash
uv run --with mlx-lm mlx_lm.generate \
  --model zherebetskyy/Qwen3.8-27B-6bit-mlx \
  --prompt "Write a high-performance Python script using asyncio."
```

---

### 2. Python API Core Examples

#### 📸 Multimodal Vision Interaction
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# Load the model and its matching multi-modal visual processor
model, processor = load("zherebetskyy/Qwen3.8-27B-6bit-mlx")

image = ["/path/to/your_image.jpg"]
prompt = "Analyze the spatial layout of this scene."

# Wrap prompt and images inside the chat format
formatted = apply_chat_template(processor, model.config, prompt, num_images=len(image))

# Generate visual outputs on the Apple Silicon GPU
result = generate(model, processor, formatted, image, max_tokens=400, temp=0.7)
print(result.text)
```

#### ✍️ Standard Text Generation
```python
from mlx_lm import load, generate

model, tokenizer = load("zherebetskyy/Qwen3.8-27B-6bit-mlx")

response = generate(
    model, 
    tokenizer, 
    prompt="Design a clean multi-tenant architecture for a SaaS app.", 
    max_tokens=500, 
    temp=0.6
)
print(response)
```

---

## ⚠️ Crucial System Prompt Note
To prevent the model from underperforming, the first line of your user/system system prompt **must** conform to the following explicit format:

```text
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
```
*You can seamlessly append your customized contextual instructions or agent boundaries right after this line.*

---

## 🧠 Thinking Toggle Mode
The included custom Jinja template natively reads and extracts `<|think_on|>` and `<|think_off|>` formatting tokens directly from message blocks to flag the model's internal step-by-step reasoning system on the fly:

```text
System: You are an engineering assistant. <|think_off|>
User: What is the optimal time complexity of a binary search tree?
```
*(Triggers a direct response without outputting verbose reasoning paths).*

```text
System: You are a cryptography expert. <|think_on|>
User: Implement a side-channel attack resilient RSA algorithm block in C.
```
*(Forces the model to brainstorm structural safety vectors step-by-step before streaming code).*

---

## 📊 Recommended Sampling Parameters
*Ensure you allocate over 128K+ of your context window if utilizing active reasoning (thinking) configurations.*

| Mode | Temperature (`temp`) | Top-P | Top-K | Min-P | Repeat Penalty | Presence Penalty |
|---|---|---|---|---|---|---|
| 💻 **Thinking (Coding/Logic)** | `0.6` | `0.95` | `20` | `0` | `1.0` *(Off)* | Off |
| 🌐 **Thinking (General Tasks)** | `1.0` | `0.95` | `20` | `0` | `1.0` *(Off)* | `1.5` |
| ⚡ **Fast Generation (No-Thinking)**| `0.7` | `0.80` | `20` | `0` | `1.0` *(Off)* | `1.5` |

---

## 🔗 Reference Links
* **Original Architecture Baseline:** [Qwen/Qwen3.8-27B](https://huggingface.co)
* **Quantization Origin Source:** [mlx-community/Qwen3.8-27B-bf16](https://huggingface.co)
* **Official Model Announcement:** [Alibaba Qwen Team Blog](https://qwen.ai/blog)

## 👤 Credits & Authorship
* **Model Creators:** Alibaba Cloud (Qwen Team)
* **MLX 6-bit Conversion Specialist:** [zherebetskyy](https://huggingface.co)
