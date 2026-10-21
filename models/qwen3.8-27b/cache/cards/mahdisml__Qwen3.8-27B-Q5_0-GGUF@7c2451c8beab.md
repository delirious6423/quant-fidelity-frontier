---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- llama-cpp
base_model: Qwen/Qwen3.8-27B
---

# mahdisml/Qwen3.8-27B-Q5_0-GGUF
This model was converted to GGUF format from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)

## Use with llama.cpp
Install llama.cpp through brew (works on Mac and Linux)

```bash
brew install llama.cpp

```
Invoke the llama.cpp server or the CLI.

### CLI:
```bash
llama-cli --hf-repo mahdisml/Qwen3.8-27B-Q5_0-GGUF --hf-file qwen3.8-27b-q5_0.gguf -p "The meaning to life and the universe is"
```

### Server:
```bash
llama-server --hf-repo mahdisml/Qwen3.8-27B-Q5_0-GGUF --hf-file qwen3.8-27b-q5_0.gguf -c 2048
```

Note: You can also use this checkpoint directly through the [usage steps](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#usage) listed in the Llama.cpp repo as well.

Step 1: Clone llama.cpp from GitHub.
```
git clone https://github.com/ggerganov/llama.cpp
```

Step 2: Move into the llama.cpp folder and build it with `LLAMA_CURL=1` flag along with other hardware-specific flags (for ex: LLAMA_CUDA=1 for Nvidia GPUs on Linux).
```
cd llama.cpp && LLAMA_CURL=1 make
```

Step 3: Run inference through the main binary.
```
./llama-cli --hf-repo mahdisml/Qwen3.8-27B-Q5_0-GGUF --hf-file qwen3.8-27b-q5_0.gguf -p "The meaning to life and the universe is"
```
or
```
./llama-server --hf-repo mahdisml/Qwen3.8-27B-Q5_0-GGUF --hf-file qwen3.8-27b-q5_0.gguf -c 2048
```