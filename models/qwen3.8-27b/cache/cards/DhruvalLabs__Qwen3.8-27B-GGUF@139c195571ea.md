---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
  - deploy:azure
  - quantized
  - conversational
  - deploy:sagemaker
  - region:us
  - qwen3_5
  - gguf
  - image-text-to-text
  - transformers
  - multimodal
  - eval-results
  - vlm
  - vision
  - license:apache-2.0
  - safetensors
language:
  - en
---

<div align="center">

# Qwen3.8-27B — GGUF Quantizations (VLM)

[![Model on HF](https://img.shields.io/badge/🤗-Model_on_HuggingFace-yellow)](https://huggingface.co/DhruvalLabs/Qwen3.8-27B-GGUF)
[![Original Model](https://img.shields.io/badge/Original-Qwen_Qwen3.8-27B-blue)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![quant-kit](https://img.shields.io/badge/Made_with-quant--kit-green)](https://github.com/DhruvalPtl/quant-kit)

**Quantized GGUF versions of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**

This is a **Vision-Language Model (VLM)** — it can understand both text and images.

Works with **[llama.cpp](https://github.com/ggerganov/llama.cpp)** · **[LM Studio](https://lmstudio.ai)** · **[Jan](https://jan.ai)** · **[Ollama](https://ollama.ai)**

*Quantized by **[DhruvalLabs](https://huggingface.co/DhruvalLabs)** on August 28, 2026 using [quant-kit](https://github.com/DhruvalPtl/quant-kit)*

</div>

---

> [!IMPORTANT]
> **This VLM requires TWO files** — a text backbone GGUF and the `mmproj` vision encoder GGUF.
> Download one text backbone (e.g. Q4_K_M) **and** the mmproj file. Both must be in the same folder.

---

## 📦 Available Files

### 🔤 Text Backbone (quantized — pick ONE)

| Filename | Size | RAM Required | Quant | Quality | Best For |
|---|---|---|---|---|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 15.66 GB | ~17.2 GB | `Q4_K_M` ✅ **Recommended** | ⭐⭐⭐⭐ | Best balance of size and quality. Recommended for most users. |
| `Qwen3.8-27B-Q5_K_M.gguf` | 18.19 GB | ~19.7 GB | `Q5_K_M` | ⭐⭐⭐⭐½ | Better quality than Q4, slightly larger. Great if you have the RAM. |
| `Qwen3.8-27B-Q5_K_S.gguf` | 17.67 GB | ~19.2 GB | `Q5_K_S` | ⭐⭐⭐⭐ | Large but accurate. |
| `Qwen3.8-27B-Q6_K.gguf` | 20.89 GB | ~22.4 GB | `Q6_K` | ⭐⭐⭐⭐⭐ | Near-perfect quality, very large. |
| `Qwen3.8-27B-Q8_0.gguf` | 27.05 GB | ~28.6 GB | `Q8_0` | ⭐⭐⭐⭐⭐ | Closest to original quality. Use when RAM is not a concern. |

### 🖼️ Vision Encoder — mmproj (always required, always F16)

| Filename | Size | Notes |
|---|---|---|
| `Qwen3.8-27B-mmproj-f16.gguf` | 0.86 GB | Always F16 — vision encoder is not quantized |

> ⚠️ **You need BOTH files** — one text backbone + the mmproj — to run this VLM.


---

## ⚡ Speed Benchmarks

*Run `python benchmark.py --model Qwen3.8-27B` to generate results.*


---

## 🚀 How to Use

### LM Studio (Easiest — GUI)
1. Search for `DhruvalLabs/Qwen3.8-27B` in LM Studio
2. Download the Q4_K_M text file **and** the mmproj file
3. Load the model — LM Studio automatically uses both files

### Ollama
```bash
ollama run dhruvallabs/qwen3.8-27b
```

### llama.cpp CLI — Text + Image
```bash
# Download both files to the same directory, then:
./llama-llava-cli \
  -m Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj Qwen3.8-27B-mmproj-f16.gguf \
  --image /path/to/your/image.jpg \
  -p "Describe this image in detail." \
  -n 512
```

### llama.cpp CLI — Text only (no image)
```bash
./llama-cli \
  -m Qwen3.8-27B-Q4_K_M.gguf \
  -p "You are a helpful assistant." \
  --conversation
```

### Python — llama-cpp-python
```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler

# Load VLM with mmproj
chat_handler = Llava16ChatHandler(clip_model_path="./Qwen3.8-27B-mmproj-f16.gguf")
llm = Llama(
    model_path="./Qwen3.8-27B-Q4_K_M.gguf",
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=4096,
    logits_all=True,
)

# Text + image inference
response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                {"type": "text",      "text":      "What do you see in this image?"}
            ]
        }
    ]
)
print(response["choices"][0]["message"]["content"])
```

---

## 🔍 VLM Architecture

This model uses a **two-component architecture**:

| Component | File | Purpose |
|---|---|---|
| **Text Backbone** | `Qwen3.8-27B-Q4_K_M.gguf` | Language understanding & generation |
| **Vision Encoder (mmproj)** | `Qwen3.8-27B-mmproj-f16.gguf` | Image feature extraction (always F16) |

> **Why is mmproj always F16?**
> The vision encoder maps image pixels to token embeddings. Quantizing it causes
> visible visual artifacts and degraded image understanding. It stays at F16 (half precision)
> which is already very efficient at ~1-2GB for most models.

---

## 🔍 About GGUF Quantization

| Format | Bits/weight | Quality |
|---|---|---|
| Q3_K_M | ~3.3 | ⭐⭐⭐ |
| Q4_K_M | ~4.5 | ⭐⭐⭐⭐ ← recommended |
| Q5_K_M | ~5.6 | ⭐⭐⭐⭐½ |
| Q8_0 | ~8.5 | ⭐⭐⭐⭐⭐ |

---

## 💬 Community & Feedback

Found an issue? Open a **Discussion** in the Community tab.

If useful, please:
- ⭐ Star [quant-kit](https://github.com/DhruvalPtl/quant-kit) on GitHub
- 👍 Like this model on HuggingFace