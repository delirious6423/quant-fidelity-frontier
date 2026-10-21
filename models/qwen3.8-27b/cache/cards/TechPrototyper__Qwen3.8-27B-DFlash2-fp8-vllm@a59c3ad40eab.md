---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model:
  - Qwen/Qwen3.8-27B
inference: false
tags:
  - dflash2
  - speculative-decoding
  - block-diffusion
  - draft-model
  - vllm
  - fp8
  - compressed-tensors
---

# Qwen3.8-27B-DFlash2 — FP8 (vLLM)

FP8-quantized [DFlash 2](https://github.com/z-lab/dflash) draft model for
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), packaged for
speculative decoding in **vLLM**.

This is **not a standalone language model.** It runs inside a speculative
decoding server and drafts token blocks for the target model to verify. It is
an FP8 requantization of the DFlash 2 drafter published by Inco AI /
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).

## What this is

DFlash 2 is a block-diffusion drafter for speculative decoding: it predicts a
whole block of tokens in a single pass, keeps the top candidates at every
position, and a lightweight selector traces one coherent path through them.
Two-tap dynamic convolutions in the backbone keep the draft from decaying
toward the end of the block.

This repository quantizes the transformer projection weights
(`q/k/v/o_proj`, `gate/up/down_proj`) to **FP8 (E4M3)** using
[compressed-tensors](https://github.com/neuralmagic/compressed-tensors):

- **Weights:** FP8, **per-channel** static scales, symmetric.
- **Activations:** FP8, dynamic per-token.
- **Kept in higher precision (not quantized):** the DFlash convolution taps
  (`attention_conv`, `mlp_conv`), the candidate selector, the `fc` head, all
  norms, and the embeddings.

The FP8 weights shrink the drafter to ~2.1 GB. Because the **target model
verifies every drafted token**, speculative decoding stays
distribution-preserving regardless of drafter precision — quantizing the
drafter changes acceptance length (speed), not the correctness of the emitted
tokens.

## Usage (vLLM)

Requires a vLLM build with DFlash speculative-decoding support
(vLLM PR [#53122](https://github.com/vllm-project/vllm/pull/53122)). Pass this
repo as the speculative model and declare `compressed-tensors` so the FP8
drafter weights are loaded correctly:

```bash
vllm serve Qwen/Qwen3.8-27B \
  --speculative-config '{
    "method": "dflash",
    "model": "TechPrototyper/Qwen3.8-27B-DFlash2-fp8-vllm",
    "quantization": "compressed-tensors",
    "num_speculative_tokens": 7
  }'
```

The drafter uses a block size of 8 (7 draft tokens per verification step). Use
Qwen3.8's recommended sampling parameters (`temperature 1.0, top_p 0.95,
top_k 20`).

## Attribution

- Drafter architecture and weights: **DFlash 2** by Inco AI —
  [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2),
  built on the original DFlash work from
  [`z-lab/dflash`](https://github.com/z-lab/dflash).
- vLLM integration path: DFlash speculative decoding, vLLM PR
  [#53122](https://github.com/vllm-project/vllm/pull/53122).
- FP8 requantization: this repository (compressed-tensors, per-channel).

## Citation

```bibtex
@misc{inco2026dflash2,
  title  = {{DFlash 2: Keep Drafting Parallel}},
  author = {{Inco AI}},
  year   = {2026},
  month  = {August},
  url    = {https://inco.ai/blog/dflash2/}
}

@inproceedings{chen2026dflash,
  title     = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author    = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
