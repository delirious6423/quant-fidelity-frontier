---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
tags:
- GGUF
- Imatrix
- Qwen
- Qwen3.8
- LLM
- Text
- Dense
---

# 📟 Qwen3.8-27B-Dense-Imatrix-IQ3_M.gguf (2026 Edition)

> "Local intelligence... to the max."

This is a custom-quantized version of **Qwen3.8-27B**, specifically optimized to obtain the highest possible local byte-intelligence ratio with 24GB+ RAM consumer laptops or computers. 

## 🧠 Why this model is different
Unlike a standard quant, this model was processed using a **custom Importance Matrix (imatrix)**. The training data for the imatrix was hand-curated to preserve:
* **Incredible reasoning:** Inclusion of custom coding examples built with frontier models provides high retention of very specific and sharp architectural reasoning skills
* **Logical Flow:** Inclusion of `llama.cpp` source code, logic puzzles, and historical writing in the imatrix training to ensure the model stays coherent at low bitrates.
* **High Speed:** Built using llama.cpp specifically for local-first AI and edge computing setups like apple silicon with minimum 24GB RAM.

## 🛠 Quantization Details
- **Base Model:** Qwen3.8-27B
- **Quantization:** IQ3_M
- **Format:** GGUF
- **Size:** ~12.77 GB
- **Context Length:** 262144 tokens

### ⚙️ Recommended Inference Settings

Optimize for balance between creativity and coherence:

* **`--repeat-penalty`: 1.1 – 1.4**
*(Sweet spot! Pushes away from familiar loops. >1.5 causes "robot-speak".)*
* **`--repeat-last-n`: 128 – 256**
*(Larger window ensures the model doesn't forget recent repetitions.)*
* **`--temperature`: 0.7 – 0.8**
*(Prevents over-committing to safe/repetitive tokens.)*
* **`--top-p`: 0.90**
*(Trims low-probability hallucinations without killing creativity.)*
* **`--min-p`: 0.05 – 0.1**
*(Optional: Prunes very low-probability tokens if your backend supports it.)*


### 📈 Perplexity Benchmarks
*coming soon*


### ⚖️ Evaluation Verdict
The IQ3_M (Imatrix) delivers performance closer to 4-bit quants while maintaining the memory footprint of a 3-bit model, giving you more available Ctx 

### 🚀 Hardware Performance (Apple M2)
*coming soon*

### 🌐 Links 

Check out my other models!

--------------------------------------------

*24GB+ (RAM)*

[Qwen3.6-27B-SuperDense](https://huggingface.co/macwhisperer/Qwen3.6-27B-SuperDense).

[Gemma4-26B-SuperMoE](https://huggingface.co/macwhisperer/Gemma4-26B-SuperMoE).

[Gemma4-31B-SuperDense](https://huggingface.co/macwhisperer/Gemma4-31B-SuperDense).

[Qwen3.6-35B-SuperMoE](https://huggingface.co/macwhisperer/Qwen3.6-35B-SuperMoE).

--------------------------------------------

*16GB+ (RAM)*

[Gemma4-12B-SuperDense](https://huggingface.co/macwhisperer/Gemma4-12B-SuperDense).

--------------------------------------------

*8GB+ (RAM)*

[Qwen3.5-9B-SuperDense](https://huggingface.co/macwhisperer/Qwen3.5-9B-SuperDense).

[Qwen3.5-4B-SuperDense](https://huggingface.co/macwhisperer/Qwen3.5-4B-SuperDense).

[Gemma4-4B-SuperDense](https://huggingface.co/macwhisperer/Gemma4-4B-SuperDense).

[Gemma4-2B-SuperDense](https://huggingface.co/macwhisperer/Gemma4-2B-SuperDense).

--------------------------------------------

*4GB+ (RAM)*

[Smartchild](https://huggingface.co/macwhisperer/smartchild).

--------------------------------------------

All make excellent companions to this model!

---