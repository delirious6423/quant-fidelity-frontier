---
license: apache-2.0
library_name: ggml
base_model:
  - Qwen/Qwen3.8-27B
  - z-lab/Qwen3.8-27B-DFlash2
tags:
  - gguf
  - dflash2
  - speculative-decoding
  - draft-model
  - block-diffusion
pipeline_tag: text-generation
---

# Qwen3.8-27B-DFlash2 GGUF

GGUF conversions of [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), the DFlash 2 draft model for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).

This is not a standalone language model. It is a draft model for speculative decoding: it drafts blocks of tokens that Qwen3.8-27B then verifies. Loading it on its own fails with `dflash requires ctx_other to be set`, which is the expected response for a draft model without its target. Run it next to the target model with `--spec-type draft-dflash`.

DFlash 2 predicts a whole block of tokens in one pass and keeps the top candidates at every position. A selector then traces one path through the candidates. Two-tap dynamic convolutions in the backbone keep draft quality from dropping toward the end of the block. Decoding is lossless: greedy output matches the target model, and sampling preserves its distribution. The z-lab repository is a mirror of [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2). The DFlash 2 paper and reference code are at [inco.ai/blog/dflash2](https://inco.ai/blog/dflash2/) and [github.com/z-lab/dflash](https://github.com/z-lab/dflash).

## Files

| File | Quant | Size | Bytes | BPW | SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| `Qwen3.8-27B-DFlash2-BF16.gguf` | BF16 | 3.60 GiB | 3,860,292,992 | 16.00 | `30c30ad2def19d4834dd8e2fe5505c2d4b954bafe2b43f7a0e502d5d9fb9048b` |
| `Qwen3.8-27B-DFlash2-Q6_K.gguf` | Q6_K | 1.48 GiB | 1,590,412,672 | 6.57 | `59c09bfdd28aa5dd3dfe18c8a899d008162aaafe58bc64a77ccf4f5f963ca0b7` |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | Q4_K_M | 1.06 GiB | 1,143,006,592 | 4.71 | `e44b99d7f4bce8ad5b573190c95e20678613fec9c9f455b073b6d9ef99c241e1` |
| `Qwen3.8-27B-DFlash2-Q4_K_S.gguf` | Q4_K_S | 1.02 GiB | 1,096,967,552 | 4.51 | `55bf2541d4d7af2c51153aafaab981d8e1caabeb29fff4923179951b65111bda` |

There is no Q6_K_M in llama.cpp; Q6_K is the 6-bit K-quant. If you want the smallest usable draft, take Q4_K_S. If you want the draft to cost as little accuracy as possible, take Q6_K or BF16.

## Model info

| Property | Value |
| --- | --- |
| Parameters | 1,924,404,480 (about 1.92B) |
| Layers | 5 (sliding attention, window 2048) |
| Hidden size | 5120 |
| Attention heads | 32 query, 8 key/value (head dim 128) |
| Feed forward | 17408 |
| Context length | 262,144 |
| Vocabulary | 248,320 |
| Speculation block size | 8 (7 draft tokens per verification step) |
| Convolution | kernel 2, group 16 |
| Selector | rank 256, top-k 16 |
| Target layers | 5, 19, 33, 47, 61 |
| RoPE theta | 10,000,000 |

Source model: `z-lab/Qwen3.8-27B-DFlash2`, commit `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, Apache-2.0.

## Requirements

DFlash 2 support in llama.cpp is not merged into master yet (as of 19 August 2026). Use a build from [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) or a later master commit once it lands. The files here were produced and validated with a build of that PR merged at commit `5c094f254`. Official release binaries will reject the `dflash` architecture keys until the PR ships.

## Usage with llama.cpp

Serve the target model with this draft (DFlash 2 is detected from the checkpoint, no extra flag needed):

```bash
./llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
    -md Qwen3.8-27B-DFlash2-Q6_K.gguf \
    --spec-type draft-dflash --spec-draft-n-max 7 -fa on --jinja
```

The `--spec-draft-n-max 7` matches the block size of 8, which gives 7 draft tokens per verification step. Generation without a server:

```bash
./llama-cli -m Qwen3.8-27B-Q4_K_M.gguf \
    -md Qwen3.8-27B-DFlash2-Q6_K.gguf \
    --spec-type draft-dflash --spec-draft-n-max 7 -fa on --jinja \
    -p "Hello" -n 128
```

The draft model needs roughly 1.0 to 3.6 GiB of memory depending on the quant, on top of the target model.

## How these files were made

```bash
# tokenizer comes from the target model, so --target-model-dir is required
python convert_hf_to_gguf.py z-lab/Qwen3.8-27B-DFlash2 \
    --target-model-dir Qwen/Qwen3.8-27B \
    --outtype bf16 --outfile Qwen3.8-27B-DFlash2-BF16.gguf

./llama-quantize Qwen3.8-27B-DFlash2-BF16.gguf Qwen3.8-27B-DFlash2-Q6_K.gguf Q6_K
./llama-quantize Qwen3.8-27B-DFlash2-BF16.gguf Qwen3.8-27B-DFlash2-Q4_K_M.gguf Q4_K_M
./llama-quantize Qwen3.8-27B-DFlash2-BF16.gguf Qwen3.8-27B-DFlash2-Q4_K_S.gguf Q4_K_S
```

All 81 tensors load in every file. The quantizer reports these figures:

| Quant | Model size | Quantized size |
| --- | ---: | ---: |
| BF16 | 3671.02 MiB | 3671.02 MiB |
| Q6_K | 3671.02 MiB | 1506.29 MiB |
| Q4_K_M | 3671.02 MiB | 1079.61 MiB |
| Q4_K_S | 3671.02 MiB | 1035.71 MiB |

## What DFlash 2 buys you

Numbers below come from the z-lab model card. They measured SGLang on one NVIDIA H200, FlashAttention 3, block size 8, temperature 1.0, top-p 0.95, top-k 20. Acceptance length is the per-request mean of completion tokens divided by verification steps; higher is better. All speculative methods propose 7 tokens per step.

| Task | Qwen3.8 built-in MTP | DFlash 2 |
| --- | ---: | ---: |
| GSM8K | 5.02 | 5.46 |
| MATH-500 | 4.72 | 5.28 |
| HumanEval | 3.91 | 4.39 |
| MBPP | 3.99 | 4.79 |
| MT-Bench | 3.74 | 4.10 |

At concurrency 1, throughput speedup over plain autoregressive decoding ranged from 2.67x to 3.43x across those five tasks. The llama.cpp PR reports 5.46 acceptance on GSM8K at Q4_K_M on an Apple M5 Pro with 64 GB, using the first 8 GSM8K problems.

## Sources

- Draft model: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (mirror of `incoai/Qwen3.8-27B-DFlash2`)
- Target model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- llama.cpp DFlash 2 support: [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342)
- DFlash 2 blog: [inco.ai/blog/dflash2](https://inco.ai/blog/dflash2/), code: [github.com/z-lab/dflash](https://github.com/z-lab/dflash)

License: Apache-2.0, same as the source model. The tokenizer in these GGUFs comes from the target model, Qwen/Qwen3.8-27B, also Apache-2.0.
