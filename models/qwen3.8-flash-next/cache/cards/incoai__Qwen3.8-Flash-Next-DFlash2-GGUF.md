---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
base_model:
  - Qwen/Qwen3.8-Flash-Next
inference: false
tags:
  - gguf
  - dflash2
  - speculative-decoding
  - draft-model
  - llama.cpp
---

# Qwen3.8-Flash-Next-DFlash2-GGUF

[Blog](https://inco.ai/blog/dflash2/) | [GitHub](https://github.com/z-lab/dflash)

This repository contains GGUF conversions of
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
the DFlash 2 draft model for
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
It is not a standalone language model: it runs inside a speculative
decoding server and drafts tokens for the target model to verify.

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
| `Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf` | 362 MB |
| `Qwen3.8-Flash-Next-DFlash2-Q5_K_M.gguf` | 432 MB |
| `Qwen3.8-Flash-Next-DFlash2-Q6_K.gguf` | 506 MB |
| `Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf` | 652 MB |
| `Qwen3.8-Flash-Next-DFlash2-BF16.gguf` | 1.22 GB |

## Quick Start

Download a quantization:

```bash
hf download incoai/Qwen3.8-Flash-Next-DFlash2-GGUF \
  Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf \
  --local-dir ./Qwen3.8-Flash-Next-DFlash2-GGUF
```

llama.cpp's DFlash 2 support
([PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)) cannot drive
this draft yet. Qwen3.8-Flash-Next carries its hidden state as four
hyper-connection streams, and the draft is trained on their mean, which
llama.cpp does not compute for the draft yet. Until it does, serve the
[safetensors checkpoint](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
with SGLang.

See the [blog post](https://inco.ai/blog/dflash2/) for other engines and
more details.

## Evaluation

Acceptance lengths for these GGUFs will be added once llama.cpp supports
this target. Full evaluations of the base checkpoint are on the
[main model card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2).

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
