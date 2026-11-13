---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: juiceb0xc0de
license: apache-2.0
tags:
- gguf
- llama.cpp
- byogguf
---

# Qwen3.8-27B GGUF

GGUF quants of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), converted from the original safetensors weights to F16 and
quantized with llama.cpp 1.0.0-dev (build 12366, commit 31953bcf7) in the [BYOGGUF](https://huggingface.co/spaces/juiceb0xc0de/llama.cpp-v1-0-0-BYOGGUF) Space.

| File | Type | Size |
|---|---|---|
| [Qwen3.8-27B-Q8_0.gguf](https://huggingface.co/juiceb0xc0de/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q8_0.gguf) | Q8_0 | 27.05 GB |
| [Qwen3.8-27B-Q4_K_M.gguf](https://huggingface.co/juiceb0xc0de/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_K_M.gguf) | Q4_K_M | 15.84 GB |
| [Qwen3.8-27B-Q2_K.gguf](https://huggingface.co/juiceb0xc0de/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q2_K.gguf) | Q2_K | 10.45 GB |



## Run it

```bash
llama-server -hf juiceb0xc0de/Qwen3.8-27B-GGUF:Q4_K_M
```

Or bring it straight back to [BYOGGUF](https://huggingface.co/spaces/juiceb0xc0de/llama.cpp-v1-0-0-BYOGGUF), paste `juiceb0xc0de/Qwen3.8-27B-GGUF`, and chat with it
on a Blackwell.

See the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B) for what the model is and how it is licensed.
