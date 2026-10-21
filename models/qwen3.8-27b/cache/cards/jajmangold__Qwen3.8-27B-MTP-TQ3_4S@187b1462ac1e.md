---
license: apache-2.0
language:
- en
library_name: gguf
pipeline_tag: text-generation
model_name: Qwen3.8-27B-MTP-TQ3_4S
tags:
- gguf
- llama.cpp
- qwen
- qwen3.8
- mtp
- speculative-decoding
- turboquant
- tq3_4s
- conversational
base_model:
- Qwen/Qwen3.8-27B
model-index:
- name: Qwen3.8-27B-MTP-TQ3_4S
  results: []
---

# Qwen3.8-27B-MTP-TQ3_4S

TurboQuant `TQ3_4S` GGUF build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with native MTP (Multi-Token Prediction) draft heads for speculative decoding.

This file was quantized with the TurboQuant toolchain and is intended for the public TurboQuant runtime fork maintained by **Y Tan** ([GitHub: turbo-tan](https://github.com/turbo-tan)):

- Runtime fork: [https://github.com/turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3)
- Reference releases by Y Tan: [https://huggingface.co/YTan2000](https://huggingface.co/YTan2000)

Credit goes to Y Tan for the TQ3_4S quantization path and the llama.cpp runtime support this model depends on. See the [Qwen3.5-27B-TQ3_4S reference card](https://huggingface.co/YTan2000/Qwen3.5-27B-TQ3_4S) for the format's published quality and speed characteristics.

## Required Runtime

**This model requires the TurboQuant runtime fork** — stock `llama.cpp` builds cannot load the `TQ3_4S` tensor type.

This is an **MTP model**: the GGUF includes the trained MTP draft block, so serve it with draft-MTP speculative decoding for full decode speed.

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3.git
cd llama.cpp-tq3

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## File

- **File:** `Qwen3.8-27B-MTP-TQ3_4S.gguf`
- **Size:** 13.68 GB (12.74 GiB)
- **SHA256:** `b734f643665a9d1f356cfcca0ef65572eb08ab9ab25c29b1ddf4ac602548e2cb`
- **Quantization:** `TQ3_4S` (3.5-bit Walsh-Hadamard transform format)
- **Architecture:** `qwen35` — 64 layers, hybrid Gated DeltaNet + Gated Attention, 1 MTP draft block
- **Context:** 262,144 tokens native

## Recommended Launch

```bash
./build/bin/llama-server \
  -m Qwen3.8-27B-MTP-TQ3_4S.gguf \
  --host 127.0.0.1 --port 8080 \
  -c 32768 -np 1 -ngl 99 -fa on \
  -ctk q8_0 -ctv tq3_0 \
  --spec-type draft-mtp \
  --spec-draft-n-min 1 \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.0 \
  --no-spec-draft-backend-sampling \
  --reasoning off --jinja
```

Settings notes:

- `--spec-draft-n-max 2` is optimal for the 3.8 MTP head (measured)
- `--no-spec-draft-backend-sampling` keeps CUDA graph reuse high
- The model supports thinking mode via `--reasoning-format deepseek --reasoning-budget N`

## License

Subject to the [Qwen3.8-27B base model license](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0) and the license terms of the runtime components used to run the GGUF.
