---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
language:
- en
- ru
- de
- fr
- it
- zh
- ja
- ko
- ar
pipeline_tag: text-generation
tags:
- gemma
- gemma-4
- gguf
- imatrix
- quantization
- moe
- llama-cpp
---

# 🧠 Gemma 4 (26B-A4B) MoE - Imatrix Quantized (IQ4_XS)

This is a high-performance, hybrid **Mixture of Experts (MoE)** GGUF version of Google's **Gemma 4 26B**. 
By utilizing a custom **Importance Matrix (imatrix)**, this `IQ4_XS` quantization maintains extreme logical precision while enabling blazing fast inference speeds. In this MoE architecture, only a fraction of parameters are activated per token, making it significantly faster than dense models of similar size.

Optimized and compiled by **Krasnopjorovs** ([Artjoms](https://artjoms.ai)).

## 🌍 Multilingual Capabilities
The model retains the full 262,144 token vocabulary, showing exceptional reasoning in English, Russian, German, and other major languages.

## 📊 Model Specifications
* **Base Model:** Google Gemma 4 26B-A4B (MoE)
* **Quantization Format:** IQ4_XS (GGUF)
* **Optimization:** Custom Imatrix applied
* **File Size:** ~13.92 GB (Extremely VRAM efficient)
* **Context Size:** 131,072 tokens
* **License:** Gemma 


### 🧠 Expert Logic & Coding Tests:
The model was subjected to rigorous reasoning and programming benchmarks:
1. **Coding (Autograd from scratch):** Successfully wrote a complete `Value` class for automatic differentiation in Python (Scalar Autograd) with topological sort and backpropagation logic.
2. **Physical Reasoning:** Correctly predicted the gravitational outcome of an overturned table with an open water bottle (zero hallucinations).
3. **Lateral Thinking:** Solved the "Snail on the Wall" and "Family Tree" riddles with 100% accuracy.

## 🛠️ Usage with `llama.cpp`

**CLI Example:**
```bash
./llama-cli -m gemma-4-26b-a4b-it-Imatrix-IQ4_XS.gguf -c 131072 -ngl 99 -fa -cnv

🤝 About the Builder
Compiled for secure, high-speed local AI ecosystems. For professional AI workstations and private LLM servers, visit artjoms.ai.