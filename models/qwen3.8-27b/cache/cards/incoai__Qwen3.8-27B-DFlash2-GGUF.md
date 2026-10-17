---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
base_model:
  - Qwen/Qwen3.8-27B
inference: false
tags:
  - gguf
  - dflash2
  - speculative-decoding
  - draft-model
  - llama.cpp
---

# Qwen3.8-27B-DFlash2-GGUF

[Blog](https://inco.ai/blog/dflash2/) | [GitHub](https://github.com/z-lab/dflash)

This repository contains GGUF conversions of
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
the DFlash 2 draft model for
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
It is not a standalone language model: it runs inside a speculative
decoding server and drafts tokens for the target model to verify. The
checkpoints are also mirrored at
[`z-lab/Qwen3.8-27B-DFlash2-GGUF`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF).

DFlash 2 is a block-diffusion drafter for speculative decoding. It predicts
a whole block of tokens in a single pass and keeps the top candidates at
every position. A lightweight selector then traces one coherent path through
them. Two-tap dynamic convolutions in the backbone keep the draft from
decaying toward the end of the block. Decoding is lossless: greedy output
matches the target model exactly, and sampling preserves its distribution.

<div align="center">
  <img src="assets/dflash2-figure.png" alt="DFlash 2: parallel block drafting with a candidate path selector" width="100%">
</div>

| File | Size |
| :--- | ---: |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1.1 GB |
| `Qwen3.8-27B-DFlash2-Q8_0.gguf` | 2.0 GB |
| `Qwen3.8-27B-DFlash2-BF16.gguf` | 3.8 GB |

## Quick Start

Build [llama.cpp](https://github.com/ggml-org/llama.cpp) with DFlash 2
support ([PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)):

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git fetch origin pull/27342/head:pr-27342
git switch pr-27342

# NVIDIA CUDA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j

# Apple Silicon
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build -j
```

Then serve:

```bash
./build/bin/llama-server \
  -hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M \
  -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M \
  --spec-type draft-dflash \
  --spec-draft-n-max 7
```

See the [blog post](https://inco.ai/blog/dflash2/) for other engines and
more details.

## Evaluation

- Target: [`ggml-org/Qwen3.8-27B-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF), `Q4_K_M`
- Sampling: Qwen3.8's officially recommended parameters (temperature 1.0, top-p 0.95, top-k 20), with `xhigh` reasoning effort
- Maximum new tokens: 2048
- Prompts: the first eight GSM8K test examples

### Acceptance Length

Acceptance length is the per-request mean of completion tokens divided by
verification steps. Higher is better.

| Draft GGUF | Acceptance Length |
| :--- | ---: |
| BF16 | 5.28 |
| Q8_0 | 5.13 |
| Q4_K_M | 5.39 |

Full evaluations of the base checkpoint are on the
[main model card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).

## Citation

If you find DFlash 2 useful, please cite:

```bibtex
@misc{inco2026dflash2,
  title  = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}
```

Please also cite the original DFlash paper:

```bibtex
@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
