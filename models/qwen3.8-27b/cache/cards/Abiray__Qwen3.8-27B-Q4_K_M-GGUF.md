---
base_model: Qwen/Qwen3.8-27B
library_name: gguf
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- qwen
- quantized
- gguf
- llama-cpp
- vision
---

# Qwen3.8-27B GGUF (Vision-Language)

This repository contains the GGUF format quantization of the [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) model, a native vision-language model.

The model has been quantized using `llama.cpp` (release [b10430](https://github.com/ggml-org/llama.cpp/releases/tag/b10430)) to the `Q4_K_M` format.

## File Details

*   **Model Name**: Qwen3.8-27B
*   **Quantization**: Q4_K_M
*   **Main Model**: `Qwen3.8-27B-Q4_K_M.gguf` (16.8 GB)
*   **Vision Adapter**: `mmproj-F16.gguf` (Required for vision/multimodal tasks)
*   **Quantization Tool**: llama.cpp (b10430)

## About Qwen3.8-27B
Qwen3.8-27B is the most capable generation in the Qwen open-model family, built on the architectural foundation of Qwen3.5. It is a native vision-language model that understands images and videos, delivering substantial gains across coding, professional work, research, and long-horizon agentic tasks. 

For more information, please visit the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B).

## Usage

### llama.cpp (Text Only)
To run this model as a text-only model using `llama.cpp` from the command line:

```bash
./llama-cli -m Qwen3.8-27B-Q4_K_M.gguf -p "Write a Python function to merge two sorted linked lists." -n 5124
```
### llama.cpp (Vision-Language)
To utilize the vision capabilities, you must provide the mmproj file using the --mmproj flag:
```bash
./llama-cli -m Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj mmproj-F16.gguf \
  --image path/to/your/image.jpg \
  -p "Describe the contents of this image."
```
### LM Studio / Ollama / GPT4All
This GGUF file is compatible with popular local LLM inference tools. When configuring the model in your preferred client, ensure you attach the mmproj-F16.gguf file in the "Vision Adapter" or "Multimedia Projection" setting to enable image processing.
### Acknowledgements
The original model was developed and released by the Qwen Team.