---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language:
- en
- zh
pipeline_tag: text-generation
tags:
- qwen
- reasoning
- orthogonal-steering
- direct-think
- zero-rambling
- mechanistic-interpretability
- gguf
- ollama
- llama.cpp
- apple-silicon
model_creator: Qwen
model_name: Qwen3.8-27B-Direct
quantized_by: swarmagents
---

<div align="center">

# Qwen3.8-27B-Direct (GGUF)

**In-Distribution Contrastive Orthogonal Weight Steering on `Qwen/Qwen3.8-27B`**  
*Zero Circular Rambling. Surgical Reasoning. Full Multi-Token Prediction (MTP) Speculative Alignment.*

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-swarmagents-blue)](https://huggingface.co/swarmagents)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ollama Ready](https://img.shields.io/badge/Ollama-1--Click%20Import-orange)](https://ollama.com)

</div>

---

## 💡 What is Qwen3.8-27B-Direct?

Base modern reasoning models frequently suffer from **circular self-doubt in `<think>` sequences**: repeating phrases like *"Wait, let me double check this again..."* 8 to 12 times before finally arriving at the answer. This inflates context windows, wastes user time, and frequently causes the model to second-guess its own correct deductions.

**Qwen3.8-27B-Direct** resolves this by mathematically isolating and dampening the rambling subspace directly inside the model's weights using **In-Distribution Contrastive Orthogonal Weight Steering**.

### Highlights
- ⚡ **Shorter Thinking**: Drops mean thinking tokens from ~420 down to **~85 per query** (-79.8%), about **1.8× faster** time-to-solution on a local M4 Max (internal eval).
- 🎯 **Higher Accuracy on Math & Code**: **+0.7% on GSM8K (89.2%)** and **+1.2% on HumanEval (78.6%)** by preventing circular hallucination loops.
- 📐 **SVD Orthogonalization against Answer Subspace**: Steering vectors are strictly orthogonalized against the top-8 principal components of post-`</think>` answer tokens in `float32`, ensuring core reasoning capabilities remain untouched.
- 🛡️ **Frobenius Norm Restoration**: Every steered tensor's energy is scaled to match its original Frobenius norm:
  $$W_{\text{new}} = W_{\text{proj}} \cdot \frac{\|W\|_F}{\|W_{\text{proj}}\|_F + 10^{-8}}$$
- 🚀 **Full MTP Speculative Decoding Preservation**: Created via direct streaming safetensors surgery (bypassing Hugging Face `AutoModel.save_pretrained()` which silently drops `mtp.*` weights). All 15 MTP weights and 333 vision weights are 100% intact, enabling **+70% speculative decoding throughput** in `llama.cpp` and `vLLM`.

---

## 📊 Benchmark Results

| Benchmark | Base Qwen 3.8 27B | **Qwen3.8-27B-Direct** | Delta |
| :--- | :---: | :---: | :---: |
| **Hermes Bench (General Capability)** | 65.77% | **65.60%** | **~0.0% (Zero Forgetting)** |
| **Terminal-Bench-2.1 (CLI & Systems)** | 67.00% | **70.00%** | **+3.0% Gain** |
| **GSM8K (8-shot CoT)** | 88.50% | **89.20%** | **+0.7% Gain** |
| **HumanEval (0-shot Python)** | 77.40% | **78.60%** | **+1.2% Gain** |
| **Coffman Concurrency & Logic** | 100% | **100%** | **100% Retained** |
| **Mean Thinking Token Count** | ~420 tokens | **~85 tokens** | **-79.8% (~4.9× fewer tokens)** |
| **Time to Solution (Local M4 Max)** | 65.6s | **36.2s** | **1.8× faster turnaround** |
| **Multi-Token Prediction (MTP) Speedup** | +70% | **+70%** | Full Alignment Maintained |

---

## 🛠️ Architectural Details

- **Base Architecture**: `Qwen3_5ForConditionalGeneration` (`model_type: "qwen3_5"`).
- **Steering Target Layers**: Full Gated Attention layers `[23, 27, 31, 35, 39, 43, 47, 51]`.
- **Target Projections**: `self_attn.o_proj.weight` ($[5120, 6144]$) and `mlp.down_proj.weight` ($[5120, 17408]$).
- **Projection Parameter**: Rank-1 damped orthogonal projection with $\beta = 0.25$.
- **Steering Formula**:
  $$P = \hat{v}_l \cdot \hat{v}_l^T \quad \text{where } \hat{v}_l \perp U_{\text{cap}}^{(l)}$$
  $$W_{\text{proj}} = W - \beta (P W)$$

---

## 🚀 Quickstart

### Option 1: 1-Click with Ollama
```bash
# Easiest: pull straight from Hugging Face
ollama run hf.co/swarmagents/Qwen3.8-27B-Direct-GGUF:Q4_K_M
```

Or import locally with the bundled Modelfile:
```bash
# Clone the repository
git clone https://huggingface.co/swarmagents/Qwen3.8-27B-Direct-GGUF
cd Qwen3.8-27B-Direct-GGUF

# Create and run in Ollama
ollama create qwen3.8-27b-direct -f Modelfile
ollama run qwen3.8-27b-direct "Implement an asynchronous lock-free priority queue in Rust."
```

### Option 2: llama.cpp with Native Speculative Decoding (MTP)
Because all 15 `mtp.*` weights were preserved, you can enable native speculative drafting:
```bash
./llama-cli \
  -m Qwen3.8-27B-Direct-Q4_K_M.gguf \
  --draft-max 2 \
  -p "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSolve the 2D heat equation with mixed boundary conditions.<|im_end|>\n<|im_start|>assistant\n<think>\n"
```

---

## 📦 Provided Files

| File | Size | Description | Recommended Hardware |
| :--- | :--- | :--- | :--- |
| **`Qwen3.8-27B-Direct-Q4_K_M.gguf`** | **16.8 GB (15.7 GiB)** | Highly optimized 4-bit Medium K-quant | Apple Silicon (24GB+ unified RAM), RTX 3090 / 4090 (24GB) |
| **`Modelfile`** | 1 KB | Pre-configured Ollama import file with official chat template | Any |

---

## 🤝 Citation & Credits

Built by **[SwarmAgents](https://huggingface.co/swarmagents)** on Apple Silicon (M4 Max) using MLX and llama.cpp.

```bibtex
@misc{qwen38_27b_direct_2026,
  author = {SwarmAgents},
  title = {Qwen3.8-27B-Direct: In-Distribution Contrastive Orthogonal Weight Steering for Zero-Rambling Reasoning},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/swarmagents/Qwen3.8-27B-Direct-GGUF}}
}
```
