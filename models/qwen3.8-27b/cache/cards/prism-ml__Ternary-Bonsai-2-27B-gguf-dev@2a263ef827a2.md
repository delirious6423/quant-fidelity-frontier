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

# Bonsai 2 27B, Q2_0 (testing build)

The `Q2_0` packing of Bonsai 2 27B, kept separate until upstream support lands.

Bonsai 2 needs an activation transform that currently only exists in our
[llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp). Stock llama.cpp will load this file
anyway, since it knows the `Q2_0` type and the `qwen35` architecture, and outputs gibberish with no
warning. The main repo's `PQ2_0` and `PTQ1_0` use types stock llama.cpp does not know, so it stops
with an error rather than running them. That is why they ship there and this one does not.

It is here for testing, kernel work, and upstreaming. It moves to the main repo once that lands.

**For normal use:**
[Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), or
[Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) on
Apple Silicon. Setup for every backend:
[Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo).

**contact@prismml.com**
