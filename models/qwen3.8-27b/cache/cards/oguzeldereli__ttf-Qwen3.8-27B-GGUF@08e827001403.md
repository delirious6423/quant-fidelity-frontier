---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
tags:
  - ternary
  - 1.58-bit
  - gguf
  - llama.cpp
  - ttf
---

# ttf-Qwen3.8-27B-GGUF

Qwen3.8-27B with every linear layer **ternary** ({-1, 0, +1} times row and column scales), in one 8.2 GB GGUF file (1.6 bits
per ternary weight) that runs entirely on a 12 GB GPU.

It was made in two steps:
1. **Conversion** with [ttf-conversion](https://github.com/oguzeldereli/ttf-conversion), streamed one decoder layer
   at a time:
   - the norm gains folded into the linears, and the residual stream rotated by a randomized Hadamard;
   - centred ternary GPTQ, then coordinate descent on the trits;
   - ternary low-rank adapters (rank 0.2 of the width) and rank-16 float adapters on each linear;
   - 4-bit embedding and head.
2. **Recovery training**: 2200 steps (72M tokens) with [ttf-ternary](https://github.com/oguzeldereli/ttf-ternary),
   distilled from the full-precision model (cached top-k teacher) on a post-training mix (reasoning, code, math, tool use, chat; its instruction part decontaminated against the
   benchmarks by 13-word n-grams) plus multilingual web text. The trits
   of the main layers and of the ternary adapters stay at their conversion values; the rank-16 float adapters, the
   scales and the norms train.

## Quality

In llama.cpp, chat template with thinking off, greedy decoding. Measured on the TQ2_0 export of the same model. This
TQ1_P file holds the same trits, repacked; its main-weight row scales are stored in fp32 instead of fp16:

| benchmark | this model | Qwen3.8-27B (full precision) |
|---|---|---|
| GSM8K (first 200 problems) | 0.895 | 0.895 |
| HumanEval | 0.841 | 0.933 |
| IFEval, strict (first 200 prompts) | 0.735 | 0.787 |

Next-token loss on Wikipedia in llama.cpp (16 windows of 2048): 2.1915 for this file (2.1920 for the TQ2_0 export),
against 1.943 for a Q8_0 of the base model.

## Run it

This file needs llama.cpp with the TTF patches, including the TQ1_P weight type: ttf-home's second engine
(`--q38`, the same one as for Flash-Next). Mainline llama.cpp does not load it. [ttf-home](https://github.com/oguzeldereli/ttf-home) builds the patched engine and runs
the model:

```bash
git clone https://github.com/oguzeldereli/ttf-home && cd ttf-home
pip install -e . huggingface_hub
ttf engine build --q38           # llama.cpp + the TTF patches (CUDA when an NVIDIA GPU is found, else CPU)
hf download oguzeldereli/ttf-Qwen3.8-27B-GGUF ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf --local-dir models/ttf-27b
ttf chat models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf
ttf serve models/ttf-27b/ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf --port 8080     # OpenAI-compatible API
```

Measured in llama.cpp on an RTX 3090 Ti, everything on the GPU: 29.2 tokens/s decode, 748 tokens/s prefill (pp512),
7.6 GiB of VRAM for the weights.

## File

| file | size | layout |
|---|---|---|
| `ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf` | 8.20 GB | linears TQ1_P (160 trits in 32 bytes, 1.6 bits per weight; the down projections, whose width is not a multiple of 160, and the ternary adapters TQ1_0), embedding and head Q4_K, per-layer TTF side tensors and float adapters |

`adapters-r0.2` names this variant: the ternary low-rank adapters on each linear have rank 0.2 of the width.

## Licence and citation

The weights keep the licence of the base model (Apache-2.0). The TTF code (ttf-ternary, ttf-conversion, ttf-home and
the llama.cpp patches) is under PolyForm Noncommercial 1.0.0.

```bibtex
@software{guzeldereli_ttf_ternary,
  author = {G{\"u}zeldereli, {\"O}zg{\"u}r},
  title  = {ttf-ternary: training ternary language models without master weights},
  year   = {2026},
  doi    = {10.5281/zenodo.23394608},
  url    = {https://github.com/oguzeldereli/ttf-ternary}
}
```
