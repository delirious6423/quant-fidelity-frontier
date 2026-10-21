---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
library_name: llama.cpp
tags:
- gguf
- gemma4
- text-generation
- chat
- swarm-sovereign
- hivemindos
model_name: Swarm Sovereign 26B
---

<img src="swarm-sovereign-bee.jpg" alt="Swarm Sovereign bee mascot" width="70%">


# Swarm Sovereign 26B GGUF

Swarm Sovereign 26B is a Gemma 4 26B-A4B IT based chat model prepared for local inference and LM Studio/llama.cpp usage.

This release contains both a Q4_K_M GGUF build and the merged Hugging Face safetensors build.

GGUF build:

- File: `swarm-sovereign-26b-Q4_K_M.gguf`
- Base model: `google/gemma-4-26B-A4B-it`
- Architecture: Gemma 4
- Quantization: Q4_K_M
- Approximate file size: 16 GB
- LM Studio estimated memory at 4096 context / 100% GPU offload: 16.44 GiB
- Identity: Swarm Sovereign

Hugging Face / Transformers build:

- Files: `model-00001-of-00011.safetensors` through `model-00011-of-00011.safetensors`, plus tokenizer/config files
- Approximate directory size: 48 GB
- Use this option for Transformers/PEFT-style local loading or downstream conversion workflows.

## Local usage

### llama.cpp

```bash
llama-cli \
  -m swarm-sovereign-26b-Q4_K_M.gguf \
  -ngl 99 \
  -c 4096 \
  --jinja \
  -p "What is your name? Answer in one sentence."
```

### LM Studio

Import or download the GGUF in LM Studio, then load it as a local model. The local test identifier used during release validation was:

```text
swarm-sovereign-26b
```

## Validation

Local validation on Apple Silicon / LM Studio:

```text
Prompt: What is your name? Answer in one sentence.
Answer: My name is Swarm Sovereign.
```

LM Studio memory estimate at 4096 context with full GPU offload:

```text
Estimated GPU Memory:   16.44 GiB
Estimated Total Memory: 16.44 GiB
```

## Want to build and manage an entire private swarm of agents?

Want to build and manage an entire private swarm of agents all with shared memory, skills, and single setup?

Check out:
https://hivemindos.liamvisionary.com
X: [@TheHivemindOS](https://x.com/TheHivemindOS)

## Notes

This model is intended for local experimentation and private agent workflows. Test thoroughly for your use case before deploying in production.
