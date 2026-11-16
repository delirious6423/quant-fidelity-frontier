---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - nvfp4
  - kv-cache
  - blackwell
  - qwen3.8
---

# Qwen3.8-27B-DMAIC_ASI-S16 (GGUF)

A 13.2 GB quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) built to run on a **16 GB
GPU with about 120k tokens of context**, using an NVFP4 KV cache. Quantized by DMAIC ASI Research.

**You need two things, from two places:**

| What | Where |
|---|---|
| The model: `Qwen3.8-27B-DMAIC_ASI-S16-Q3_K_XL.gguf` (this repo) | https://huggingface.co/DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-GGUF |
| The llama.cpp build with the NVFP4 KV cache | https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv |

The file itself is a standard GGUF and loads in stock llama.cpp. The NVFP4 KV cache, which is what makes 120k of
context fit on 16 GB, needs the build above (RTX 50-series GPUs).

- People: step-by-step guide in [NVFP4-HOWTO.md](https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv/blob/main/NVFP4-HOWTO.md).
- AI assistants: point yours at [SETUP-AGENT.md](https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv/blob/main/SETUP-AGENT.md);
  it is written for an agent to follow and check each step.

## Quick start (RTX 50-series, 16 GB)

```bash
git clone https://github.com/DMAIC-ASI-Research/llama.cpp-nvfp4-kv
cd llama.cpp-nvfp4-kv
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$(command -v nvcc)
cmake --build build --config Release -j 4 --target llama-server llama-cli

hf download DMAIC-ASI/Qwen3.8-27B-DMAIC_ASI-S16-GGUF Qwen3.8-27B-DMAIC_ASI-S16-Q3_K_XL.gguf --local-dir models

build/bin/llama-server -m models/Qwen3.8-27B-DMAIC_ASI-S16-Q3_K_XL.gguf \
  -ngl 99 -c 120064 --fit off -fa on -ctk nvfp4 -ctv nvfp4 -b 2048 -ub 256 -np 1 -kvu \
  --spec-type draft-mtp,ngram-simple -ctkd nvfp4 -ctvd nvfp4 --spec-draft-n-ubatch 64 \
  --host 127.0.0.1 --port 8080
```

If loading runs out of memory (a desktop on the same GPU uses some), lower `-c` in steps of 8192.

On other GPUs, or in stock llama.cpp, use an ordinary KV cache, for example `-ctk q8_0 -ctv q8_0`, with a smaller
context.

## The file

| | |
|---|---|
| File | `Qwen3.8-27B-DMAIC_ASI-S16-Q3_K_XL.gguf` |
| Size | 13,207,095,840 bytes (3.86 bits per weight) |
| Quant | custom mixed precision in the Q3_K_XL size class: mostly IQ4_XS, IQ3_S and IQ3_XXS, with Q5_K, Q4_K and Q3_K ("Q3_K_XL" in the name is the size class, not a single llama.cpp type) |
| Contents | text model plus the built-in multi-token-prediction (MTP) head; no vision projector |
| Tested on | RTX 5070 Ti 16 GB, driver 570, CUDA 12.9, Ubuntu 24.04: context 120,064 with MTP, ~15.4 GB in use |

## Quality

KL divergence against a Q8_0 reference of the BF16 checkpoint (`llama-perplexity --kl-divergence`, 1024-token
chunks; lower is better), with the share of tokens whose top choice matches the reference:

| Text | Chunks | Mean KLD (± se) | Same top token |
|---|---:|---:|---:|
| Code | 60 | 0.0174 ± 0.0004 | 96.6 % |
| Wiki | 20 | 0.0340 ± 0.0008 | 92.5 % |
| Mixed | 20 | 0.0483 ± 0.0047 | 94.8 % |

## How it was made

A mixed-precision quantization: per-tensor types chosen to the 16 GB budget, GPTQ-style rounding on calibration
covariances, the least sensitive FFN layers slimmed further, and a Q5_K output head. The MTP head is kept so
speculative decoding works.

## Credits and license

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen Team, Apache-2.0. This
  quantization is released under the same license.
- Uses [Unsloth](https://huggingface.co/unsloth)'s published imatrix and UD-Q3_K_XL per-tensor layout as a starting
  point.
- Runs on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).

Sampling the base model recommends: thinking mode temperature 1.0, top_p 0.95, top_k 20; non-thinking temperature
0.7, top_p 0.8, top_k 20, presence_penalty 1.5.
