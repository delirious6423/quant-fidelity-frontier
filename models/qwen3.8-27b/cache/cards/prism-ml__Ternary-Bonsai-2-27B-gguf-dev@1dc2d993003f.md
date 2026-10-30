---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- ternary
- 2-bit
- gguf
- llama-cpp
- prismml
- bonsai
base_model:
- Qwen/Qwen3.8-27B
---

# Bonsai 2 27B, Q2_0 (moved)

**This file now ships in the main repo,**
[Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), as
`Ternary-Bonsai-2-27B-Q2_0.gguf`. It is the same file under a new name. Please download it from
there. This copy stays only so existing links keep working.

Stock llama.cpp runs it from release
[b11556](https://github.com/ggml-org/llama.cpp/releases/tag/b11556) onward. Older stock builds load
it without any warning and produce gibberish, because they lack the Hadamard activation runtime it
needs.

**For normal use:**
[Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), or
[Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) on
Apple Silicon. Setup for every backend:
[Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo).

**contact@prismml.com**
