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

Qwen3.8-27B with every linear layer **ternary** ({-1, 0, +1} times row and column scales), as GGUF files of 1.6 bits
per ternary weight that run entirely on a 12 GB GPU. Two variants:

| file | size | what differs |
|---|---|---|
| `ttf-Qwen3.8-27B-adapters-r0.2-TQ1_P.gguf` | 8.20 GB | ternary low-rank adapters (rank 0.2 of the width) on each linear; trits frozen in recovery |
| `ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf` | 6.74 GB | no ternary adapters; the trits are trainable in recovery (0.033% of them moved) |

The sections below describe the 8.2 GB file first; the 6.74 GB file has its own section.

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

## The 6.74 GB file (no ternary adapters)

`ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf` is converted the same way but without the ternary low-rank adapters (rank-16
float adapters only), then recovered with the trits trainable (TTF), not frozen:
- 1210 steps of 16 x 2048 tokens (39.6M tokens), distilled from the full-precision model (cached top-64 logits plus a
  sampled tail), learning rate 1e-4 to 1e-5;
- what trains: the trits, the row and column scales, the rank-16 float adapters, the norms and the DeltaNet gate inputs.
  The trits changed little: 0.033% of them moved over the run, most in the last layers. The floats carry most of
  the recovery here too.

| benchmark | this file | 8.2 GB file, same harness | Qwen3.8-27B (full precision) |
|---|---|---|---|
| GSM8K (first 200 problems) | 0.880 | 0.900 | 0.895 |
| HumanEval | 0.787 | 0.835 | 0.933 |
| IFEval, strict / loose (first 200 prompts) | 0.735 / 0.750 | 0.760 / 0.790 | 0.787 (strict) |
| TriviaQA (298, exact match) | 0.242 | 0.326 | - |

Measured with llama-server, chat template with thinking off, greedy decoding; the first two columns with the same
server build and scorer, the full-precision column in an earlier run on the same prompts.

**Engine:** this file has one tensor kind the 8.2 GB file does not (an optional bias on the DeltaNet beta projection,
where the centred input projection was folded). ttf-home's `--q38` engine has it from patch 0009 (10 Nov 2026). If you
built that engine before, delete `~/.cache/ttf/llama.cpp-q38` and run `ttf engine build --q38` again; an older engine
stops with "wrong number of tensors". Run it like the 8.2 GB file below, with this file name:

```bash
hf download oguzeldereli/ttf-Qwen3.8-27B-GGUF ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf --local-dir models/ttf-27b
ttf chat models/ttf-27b/ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf
```

## Run it

This file needs llama.cpp with the TTF patches, including the TQ1_P weight type: ttf-home's second engine
(`--q38`, the same one as for Flash-Next). Mainline llama.cpp does not load it. [ttf-home](https://github.com/oguzeldereli/ttf-home) builds the patched engine and runs
the model:

```bash
git clone https://github.com/oguzeldereli/ttf-home && cd ttf-home
pip install -U pip "setuptools>=68"   # editable installs need setuptools 64+
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
| `ttf-Qwen3.8-27B-noadapters-TQ1_P.gguf` | 6.74 GB | linears TQ1_P (the down projections TQ1_0), embedding and head Q4_K, per-layer TTF side tensors, rank-16 float adapters, DeltaNet beta bias |


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
