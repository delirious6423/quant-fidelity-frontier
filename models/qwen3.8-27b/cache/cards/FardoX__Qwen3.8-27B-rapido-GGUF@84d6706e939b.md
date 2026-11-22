---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [gguf, llama.cpp, 2-bit]
---

# Qwen3.8-27B rapido — 1.99-bit GGUF

**What:** [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) shrunk 8× (54 GB → 6.8 GB). It keeps ~98% of the
original's reasoning and is faster than Q4_0 on GPU.

**Run (main file, 1.99 bits)** — download this repo into a folder, then inside that folder:
```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout ad9d9e8f50e6ff9a6bfbbdd3aef6d74a8428cf4e && git apply ../ectv-cuda-mmq-cpu3-llamacpp.patch
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server
./build/bin/llama-server -m ../Qwen3.8-27B-rapido-ECTV.gguf --mmproj ../mmproj-Qwen3.8-27B-F16.gguf -ngl 999
```
Open http://localhost:8080. CPU only: drop `-DGGML_CUDA=ON` and `-ngl 999`. No build wanted? Use `Qwen3.8-27B-rapido.gguf`
(2.08 bits, same weights) with any stock llama.cpp b12490+.

**Specs**
- 1.99 bits/weight for the whole model (embeddings, head and MTP layer included); 6.8 GB.
- GSM8K 0.870 vs 0.889 original (97.9%, N=1319); MATH-500 0.674 vs 0.694 (97.1%, N=500); same llama.cpp harness, original in bf16.
- NVIDIA A40, batch 1: 44.9 tok/s generation (1.34× Q4_0), 1322 tok/s prompt (0.86× Q4_0).
- CPU (AVX2): prompt ~1.2× Q4_0, generation ~0.8–0.9× Q4_0.
- Method: block-Hadamard rotation + vector GPTQ on the IQ2_XXS grid, then scale-only distillation on self-generated text
  (same bits); new ECTV weight type (CPU + CUDA) in the patch.
- Fits a 12–16 GB GPU (8 GB with short context). Needs git, cmake, a C++ compiler (and the CUDA toolkit for GPU) to build.
  The vision projector was not tested. Unofficial derivative, Apache-2.0.
