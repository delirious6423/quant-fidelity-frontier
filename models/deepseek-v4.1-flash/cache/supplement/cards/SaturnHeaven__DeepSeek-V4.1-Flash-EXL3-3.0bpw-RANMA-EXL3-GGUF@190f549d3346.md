---
base_model: Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw
base_model_relation: quantized
quantized_by: Mia-AiLab
library_name: gguf
license: mit
pipeline_tag: text-generation
tags:
- exl3
- ranma
---

# DeepSeek-V4.1-Flash RANMA EXL3 GGUF

> [!WARNING]
> **Requires ranma.cpp `ranma_20261013` or newer** ([github.com/saturnsky/ranma.cpp](https://github.com/saturnsky/ranma.cpp)).
> These files use fork-specific tensor types (256 and above). They will **not** open in upstream llama.cpp, LM Studio, Ollama or other llama.cpp-based tools.

[ranma.cpp](https://github.com/saturnsky/ranma.cpp) is a llama.cpp fork developed with RDNA4 as its main target. It runs MoE models on a
three-tier expert cache: VRAM, host RAM and SSD. It supports EXL3 quantization to keep memory traffic as low as possible at a given quality.

## Source and credit

This repository repackages the **EXL3 quantization made by [Mia-AiLab](https://huggingface.co/Mia-AiLab)** of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) as GGUF. **The weights were not re-quantized**:
the trellis-coded tensors are carried over bit for bit, only wrapped in the GGUF container and split into shards. All credit for the
quantization goes to Mia-AiLab and the [ExLlamaV3](https://github.com/turboderp-org/exllamav3) project. Thank you.

| Folder | Content | bpw (head) | Size |
|---|---|---|---:|
| [`EXL3-3.0bpw-h6`](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-EXL3-GGUF/tree/main/EXL3-3.0bpw-h6) | the model, from [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw) | 3.02 (6) | 211.1 GB |
| [`DSpark-EXL3-4bpw`](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-EXL3-GGUF/tree/main/DSpark-EXL3-4bpw) | the DSpark draft model of the same checkpoint (optional) | 4 | 7.3 GB |

Each folder has a short README. `manifest.json` lists the size and SHA-256 of every file.
Not included: the vision encoder (no image input) and the checkpoint's MTP layers (speculative decoding uses the DSpark draft model).

## Running

1. Download the model folder from this repository and the engram tables (202.8 GB, needed, not in this repository) from
   [SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-Engram-GGUF](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-Engram-GGUF).
2. Give `-m` only the first shard, `DeepSeek-V4.1-Flash-EXL3-3.0bpw-h6-00001-of-00006.gguf`; the other shards are found by name.
3. Give `--engram-tables` the first file of the table set, `DeepSeek-V4.1-Flash-engram-MXFP8-00001-of-00007.gguf`.

Context, L1/L2 sizes and the draft model for each hardware configuration are in the ranma.cpp
[recipes](https://github.com/saturnsky/ranma.cpp/blob/ranma/docs/ranma/recipes.md#deepseek-v41-flash). They are examples: choose for your own use and system.

## Benchmarks

Server decode in t/s with `ranma_20261013` and the settings of the recipe rows, measured on a Radeon AI PRO R9700: warm (from the
profile of a warm-up preset), one slot, greedy, 12 coding and 17 Korean roleplay requests. The 128 GB and 64 GB rows were reproduced
on a 224 GB machine by occupying memory, the 16 GB card by limiting the device memory the server uses. Method and full tables:
[benchmark record](https://github.com/saturnsky/ranma.cpp/blob/ranma/docs/ranma/benchmarks/2026-10-13-release.md).

| GPU | Host RAM | Draft | Coding | Korean roleplay |
|---|---|---|---:|---:|
| R9700 (32 GB) | 128 GB | none | 21.11 | 21.88 |
| 16 GB card | 64 GB | none | 9.30 | 11.07 |
| R9700 (32 GB) | 224 GB | none | 23.66 | 23.89 |
| R9700 (32 GB) | 224 GB | DSpark | 21.02 | 19.36 |

## License

Same as the original model: MIT, see the [original repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
