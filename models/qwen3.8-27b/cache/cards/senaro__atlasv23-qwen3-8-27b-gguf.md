---
tags:
- gguf
- llama.cpp
- unsloth
license: apache-2.0
datasets:
- kintsugicollective/atlas-dataset-v8-final
- senaro/atlas-region2-harmful-v7
language:
- en
base_model:
- Qwen/Qwen3.8-27B
- senaro/atlasv23-trm-qwen3-8-27b
---

# atlasv23-qwen3-8-27b-gguf : GGUF
 
 More quants here: 
 mradermacher/atlasv23-sft-qwen3-8-27b-merged-GGUF
 https://huggingface.co/mradermacher/atlasv23-sft-qwen3-8-27b-merged-GGUF

This model was finetuned and converted to GGUF format using [Unsloth](https://github.com/unslothai/unsloth).

**Example usage**:
- For text only LLMs:    `llama-cli -hf senaro/atlasv23-qwen3-8-27b-gguf --jinja`
- For multimodal models: `llama-mtmd-cli -hf senaro/atlasv23-qwen3-8-27b-gguf --jinja`

This was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth)
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)