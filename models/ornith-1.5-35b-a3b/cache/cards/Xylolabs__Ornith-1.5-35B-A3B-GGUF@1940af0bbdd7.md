---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language:
- en
- ko
tags:
- gguf
- llama-cpp
- quantized
- ornith
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Ornith-1.5-35B-A3B GGUF</h1>

[![Base model](https://img.shields.io/badge/base-ornith--ai%2FOrnith--1.5--35B--A3B-blue)](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
[![Formats](https://img.shields.io/badge/formats-Q4__K__M-5B2C83)](#files)
[![License](https://img.shields.io/badge/license-mit-green)](https://huggingface.co/Xylolabs/Ornith-1.5-35B-A3B-GGUF/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-llama.cpp-black)](https://github.com/ggml-org/llama.cpp/tree/cd959bae99ff4bed22bef2604f45241c09bbb8c8)

</div>

GGUF quants of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for llama.cpp, on CUDA, Metal or CPU.


## Files

- `Ornith-1.5-35B-A3B-<QUANT>.gguf` is one file per quant.
- Contents: Each file holds the full language model plus the original tokenizer and chat template.

| Quant | Size | Mean KL vs BF16 | Same top-1 vs BF16 |
|---|---|---|---|
| Q4_K_M | 21.71 GB | 0.0792 | 87.63% |

The reference is our BF16 pass in transformers. KL and top-1 cover 8,188 positions of English and Korean text.

## How to run

Use llama.cpp commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8` or later.

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout cd959bae99ff4bed22bef2604f45241c09bbb8c8
cmake -B build -DCMAKE_BUILD_TYPE=Release        # add -DGGML_CUDA=ON for NVIDIA GPUs
cmake --build build -j --target llama-cli llama-server

./build/bin/llama-cli -m Ornith-1.5-35B-A3B-Q4_K_M.gguf --jinja -c 8192 -ngl 99
```

`llama-server` takes the same flags.

## Which file for your hardware

File size, plus the KV cache at the stated context (f16), plus about 2 GB of runtime overhead, must fit in 90% of the memory. Only 10 of its 40 layers use full attention (2 KV heads, head size 256); the others keep a small fixed state. Where the file does not fit, the table gives the CPU-offload setting that keeps the expert weights in system RAM.

| Hardware | Memory | Recommended file | Context that fits | Notes |
|---|---|---|---:|---|
| RTX 4090 / RTX 3090 | 24 GB | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 32K | Offload, slower: `--n-cpu-moe 6` keeps the experts of 6 of 40 layers in system RAM (about 4 GB system RAM, estimate); or `-ot '.ffn_.*_exps.=CPU'`.  |
| RTX 5090 | 32 GB | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 128K |  |
| RTX 6000 Ada, or 2 x 24 GB | 48 GB | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K | Two 24 GB cards need the layers split across them. |
| One H100 | 80 GB | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K |  |
| RTX PRO 6000 Blackwell (GDDR7) | 96 GB | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K |  |
| DGX Spark | 128 GB unified | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K |  |
| Mac Studio M5 Max | 128 GB unified | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K | |
| One H200 (HBM3e) | 141 GB | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K |  |
| One B200 (about 180 GB usable in a DGX B200) | 192 GB | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K |  |
| Mac Studio M5 Ultra | 256 GB unified | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K | |
| Mac Studio M5 Ultra | 512 GB unified | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K | |
| One B300 (HBM3e; usable a little less) | 288 GB | `Ornith-1.5-35B-A3B-Q4_K_M.gguf` (21.7 GB) | 256K |  |

## License and notices

MIT License, copied in `LICENSE`. We converted the weights to GGUF and quantized them. `NOTICE` names the source model and its copyright holder. Xylolabs is not affiliated with or endorsed by the original model's authors.


## Citation

Use the citation provided by the [original Ornith model
card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B):

```bibtex
@misc{ornith_1_5,
    title = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
    url = {https://ornith.ai/ornith_1_5.html},
    author = {{Ornith Team}},
    year = {2026}
}
```

## Updates

- 2026-11-22: Added a hardware recommendation table.
