---
base_model: deepseek-ai/DeepSeek-V4.1-Flash
library_name: gguf
license: mit
tags:
- ranma
---

# DeepSeek-V4.1-Flash RANMA Engram GGUF

> [!WARNING]
> **Requires ranma.cpp `ranma_20261013` or newer** ([github.com/saturnsky/ranma.cpp](https://github.com/saturnsky/ranma.cpp)).
> This is not a model on its own: it holds only the engram tables that the DeepSeek-V4.1-Flash GGUF files of ranma.cpp need.

[ranma.cpp](https://github.com/saturnsky/ranma.cpp) is a llama.cpp fork developed with RDNA4 as its main target. It runs MoE models on a
three-tier expert cache: VRAM, host RAM and SSD. It supports EXL3 quantization to keep memory traffic as low as possible at a given quality.

## What this is

The two engram tables of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (layers 1 and 14,
about 384 million rows each), taken from the original FP8 checkpoint **without changing a value**: each row is the FP8 E4M3 values and
their E8M0 scales, stored as MXFP8. Every quantization of the model uses the same tables, so they live here once and the model
repositories link to this one.

| Folder | Content | Size | Files |
|---|---|---:|---:|
| [`engram-MXFP8`](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-Engram-GGUF/tree/main/engram-MXFP8) | both tables, MXFP8 | 202.8 GB (188.8 GiB) | 7 |

`manifest.json` lists the size and SHA-256 of every file.

## Used by

- [SaturnHeaven/DeepSeek-V4.1-Flash-EXL3-3.0bpw-RANMA-EXL3-GGUF](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-EXL3-3.0bpw-RANMA-EXL3-GGUF) (3.0 bpw, Mia-AiLab)
- [SaturnHeaven/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw-RANMA-EXL3-GGUF](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw-RANMA-EXL3-GGUF) (2.9 bpw, dealignai)

Download the folder once and give ranma its first file, `--engram-tables .../DeepSeek-V4.1-Flash-engram-MXFP8-00001-of-00007.gguf`;
the other files are found by name. At load the tables are checked against the model (table names, rows, row width) and
every piece is checked at both ends. How the set is found and checked:
[engram-tables.md](https://github.com/saturnsky/ranma.cpp/blob/ranma/docs/ranma/engram-tables.md).

## License

Same as the original model: MIT, see the [original repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).
