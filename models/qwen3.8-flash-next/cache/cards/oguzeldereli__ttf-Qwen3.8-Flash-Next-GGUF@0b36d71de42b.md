---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
tags:
  - ternary
  - 1.58-bit
  - gguf
  - llama.cpp
  - moe
  - ttf
---

# ttf-Qwen3.8-Flash-Next-GGUF

Qwen3.8-Flash-Next (MoE, 180B parameters) with every routed expert and dense projection **ternary**, in one 40.4 GB
GGUF file. That is 4.6 times smaller than the FP8 release.

It was made in two steps:
1. **Conversion** with [ttf-conversion](https://github.com/oguzeldereli/ttf-conversion), layer by layer on one GPU:
   - ternary GPTQ on the centred input covariance;
   - row and column scales;
   - the input mean folded into a bias;
   - a rank-16 ternary adapter on each linear.
2. **Recovery training**: 1000 steps (33M tokens) of TTF ([ttf-ternary](https://github.com/oguzeldereli/ttf-ternary)),
   with knowledge distillation from the FP8 model on a post-training mix (reasoning, code, math, tool use, chat; its instruction part decontaminated against the
   benchmarks by 13-word n-grams) plus multilingual web text.

The per-layer n-gram embedding (PLE) table is pruned and product-quantised: 7.96 GB instead of 51.3 GB in FP8.

## Quality

Greedy decoding, chat template with thinking off, at most 512 new tokens:

| benchmark | this model (40.4 GB) | Qwen3.8-Flash-Next FP8 (185.5 GB) |
|---|---|---|
| GSM8K (first 200 problems) | 0.905 | 0.905 |
| HumanEval | 0.848 | 0.927 |
| TriviaQA, closed book (1000 questions, exact match) | 0.538 | 0.776 |

Reasoning and code hold up well. Closed-book factual recall is where ternary costs the most.

## Run it

This file needs llama.cpp with the TTF patches for this architecture (`qwen4exp`): TQ1_P weights, TTF side tensors,
adapters, and the PQ PLE table. Mainline llama.cpp does not load it. [ttf-home](https://github.com/oguzeldereli/ttf-home)
builds the patched engine and runs the model:

```bash
git clone https://github.com/oguzeldereli/ttf-home && cd ttf-home
pip install -e . huggingface_hub
ttf engine build --q38           # the Flash-Next engine
hf download oguzeldereli/ttf-Qwen3.8-Flash-Next-GGUF --local-dir models/ttf-flash
ttf chat models/ttf-flash/ttf-Qwen3.8-Flash-Next-TQ1_P.gguf
```

Most experts stay in host memory, so plan for 64 GB of RAM.

Measured on an RTX 3090 Ti (24 GB) with 12 CPU threads:
- `-- -ncmoe 18 -t 12`: 29.7 tokens/s decode, 147 tokens/s prefill;
- all experts on the CPU (`-ncmoe 99`): 22 tokens/s decode at 6.2 GB of VRAM.

Use the number of physical cores for `-t`.

## File

| file | size | layout |
|---|---|---|
| `ttf-Qwen3.8-Flash-Next-TQ1_P.gguf` | 40.38 GB | experts and attention TQ1_P (160 trits in 32 bytes, 1.6 bits per weight), adapters F16, side vectors F32, embedding / head Q8_0, PQ PLE table; the MTP layer and the vision tower are not included |

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
