---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - topocbp
  - gguf
  - quantization
  - topological-invariant-locking
  - somala
  - qwen
  - llama.cpp
  - ollama
  - topo-cbp
  - continual-learning
  - topological-governance
  - continual-backpropagation
  - arithmetic-spectral-theory
  - stability-plasticity
  - open-science
quantized_by: frankmorales2020
---

# Qwen 3.8-27B Topo-CBP GGUF

This repository hosts the **Q8_0** quantized GGUF weights for **Qwen 3.8-27B**, governed under the **Topological Continual Backpropagation (Topo-CBP)** framework developed by Sovereign Machine Laboratory (SOMALA).

CODE-1: https://github.com/frank-morales2020/AST/blob/main/Qwen3dot8_27B_topo_cbp_gguf.ipynb

CODE-2: https://github.com/frank-morales2020/AST/blob/main/APAC_AGENTIC_GGUF.ipynb

## Model Details

* **Architectural Family:** Qwen 3.8 / 3.5 (`qwen38`)
* **Parameter Count:** 27 Billion
* **Quantization:** Q8_0 (8-bit quantization)
* **Governance Framework:** Topo-CBP with Topological Invariant Locking & TopoAdamW optimization
* **Creator / Maintainer:** Frank Morales Aguilera (SOMALA)

## Overview

This model merges cutting-edge open-weights architecture with rigorous topological stability constraints. Designed for high-precision mathematical reasoning, analytical workflows, and deterministic cognitive engineering, it has been verified under native GPU acceleration.

## Usage with `llama.cpp` / Python

You can load and run this model seamlessly using `llama-cpp-python` with full CUDA offloading:

```python
from google.colab import userdata
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Retrieve your token from Colab Secrets
HF_TOKEN = userdata.get('HF_TOKEN')

repo_id = "frankmorales2020/qwen3.8-27b-topocbp-gguf"
filename = "qwen3.8-27b-topocbp-Q8_0.gguf"

model_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    token=HF_TOKEN
)

llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_gpu_layers=-1,  # Full GPU offload
    verbose=False
)

prompt = (
    "<|im_start|>system\n"
    "You are a rigorous mathematical AI assistant governed by topological invariant locking.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Prove that the sum of the first n odd numbers is n^2 using mathematical induction.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

output = llm(
    prompt,
    max_tokens=512,
    temperature=0.7,
    repeat_penalty=1.15,
    stop=["<|im_end|>"]
)

print(output["choices"][0]["text"].strip())

```

## Performance Benchmarks

Tested on an NVIDIA A100-SXM4 (80GB VRAM) running native CUDA acceleration:

* **Generation Throughput:** ~34 tokens/sec
* **Memory Footprint:** Fully offloaded within VRAM envelope for ultra-low latency inference.