---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- exl3
- exllamav3
- cmp-170hx
- mixed-precision
---

# Qwen3.8-Flash-Next EXL3 — fitted to a single unlocked CMP 170HX 8GB

A single **unlocked NVIDIA CMP 170HX 8GB** (64 GB of VRAM) runs the **full 262,144-token context** with MTP speculative
decoding, filling the card's VRAM to the last few hundred MiB.

Mixed-bitrate EXL3 built from [turboderp's EXL3 quants](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): 4-bit routed experts on 20 of the 48 layers,
3-bit on the rest, 6-bit attention, shared experts and lm_head (details below). Requires **ExLlamaV3 v1.5.1**
or newer.

## Requirements

- GPU: an unlocked CMP 170HX 8GB (64 GB of VRAM)
- SSD: 39 GB free for the n-gram table, which is read from it during inference. With enough free RAM, the whole
  39 GB can be kept in RAM instead.
- RAM: free RAM is used as a cache for the n-gram table, which makes it slightly faster.

## Numbers

All measured on the machine described [below](#test-machine), with the [TabbyAPI config](#tabbyapi) in this repo
(one stream, MTP draft 3, 8-bit K/V cache, 262,144-token cache).

**Full context** — prompt = the first N tokens of wikitext-2 (test) + a summarization request, fresh prompt each
time (no prefix cache), greedy, thinking off, up to 500 new tokens:

| Context | Prompt processing | Generation | MTP acceptance |
|---:|---:|---:|---:|
| 16,437 | 1,481 tok/s | 87.8 tok/s | 67% |
| 65,589 | 1,601 tok/s | 85.5 tok/s | 67% |
| 131,125 | 1,680 tok/s | 85.7 tok/s | 70% |
| 250,053 | 1,653 tok/s | 79.2 tok/s | 68% |

**Short context** — 500 new tokens, greedy, thinking off (median of 3 after 1 warm-up):

| | tok/s |
|---|---:|
| Generation, code (a Python module) | 99.9 |
| Generation, prose (an essay) | 81.4 |
| Prompt processing, 16k tokens (synthetic word-list prompt) | 2,013 |

## Accuracy

Greedy, thinking off. HumanEval code is run against its tests; GSM8K is scored on the final number.

| Benchmark | Score |
|---|---:|
| HumanEval (base tests) | 157 / 164 (95.7%) |
| HumanEval+ (base + extra tests) | 152 / 164 (92.7%) |
| GSM8K (first 300 test problems) | 293 / 300 (97.7%) |

## How it was made

No new quantization. ExLlamaV3 stores every linear layer independently and reads its bitrate from the tensor
shape, so tensors from two conversions of the same source model can be combined module by module. This repo takes:

| Part | Bits | Source branch |
|---|---|---|
| Attention, linear attention, shared experts, lm_head | 6 | `4.05bpw_h6_ng6` |
| Sparse-attention indexer | 4 | `4.05bpw_h6_ng6` |
| Routed experts, layers 0–9 and 38–47 | 4 | `4.05bpw_h6_ng6` |
| Routed experts, layers 10–37 | 3 | `3.05bpw_h5_ng5` |
| MTP head | 4–6 | `4.05bpw_h6_ng6` |
| Vision tower | 6 | `4.05bpw_h6_ng6` |
| n-gram (PLE) table, 39 GB | K=6 | `4.05bpw_h6_ng6` |

The 4-bit layers follow ExLlamaV3's own promotion order (closest to either end of the stack first). Experts
average 3.43 bpw. Only the tensors needed were fetched from turboderp's repo with HTTP range requests and
written straight into per-layer shards (`experts-LNN.safetensors`, `dense.safetensors`, …), so a different layer
choice is a one-file swap. The script and the exact recipe are in [`recipe/`](recipe/) — pick your own number of
4-bit layers for a different VRAM budget with `build_mix.py`.

## TabbyAPI

[`recipe/tabbyapi-config.yml`](recipe/tabbyapi-config.yml) is the config every number above was measured with.
It uses the sampling defaults in [`recipe/sampler-override-flash-next.yml`](recipe/sampler-override-flash-next.yml)
(temperature 1.0, top_p 0.95, top_k 20, min_p 0, for thinking mode); copy that file to TabbyAPI's
`sampler_overrides/flash-next.yml`. The parts that matter:

```yaml
model:
  max_seq_len: 262144
  cache_size: 262144
  cache_mode: "8,8"
  chunk_size: 2048
  max_batch_size: 1
  ngram_ram: false
  vision: true
  vision_offload: true
draft_model:
  draft_mode: mtp
  draft_cache_mode: "8,8"
  draft_num_tokens: 3
```

MTP draft length, measured in a separate run (20 × 4-bit, one stream, 500 tokens, greedy):

| `draft_num_tokens` | code (tok/s) | prose (tok/s) |
|---:|---:|---:|
| 2 | 93.8 | 81.2 |
| **3** | **102.7** | **83.5** |
| 4 (model default) | 99.4 | 72.8 |

In ExLlamaV3's own example scripts the same settings are `-cs 262144 -cq 8 -mtp -ndt 3`.

### The n-gram table

By default the 39 GB n-gram table stays on disk, and only the rows each token needs are read, in parallel. The OS
keeps what has been read in free RAM (page cache), so frequently used rows come from RAM and you do not need enough
RAM to hold the whole table. To keep all 39 GB in RAM, use `ngram_ram: true` (TabbyAPI) or `-ngr` / `--ngram_ram`
(ExLlamaV3 scripts), which is slightly faster.

## Test machine

- GPU: NVIDIA CMP 170HX 8GB, unlocked to 64 GB; reports compute capability 8.0; PCIe 2.0 x4 link; power limit
  180 W, stock clocks; driver 615.71.09
- Software: Linux 7.2, CUDA 13.4, PyTorch 2.9.1+cu130, ExLlamaV3 1.5.1 (master `6b84a21`), TabbyAPI main `7208273`

## Building ExLlamaV3 for this card

The card reports compute capability 8.0, so build for `sm_80`. With GCC 16, PyTorch 2.9's headers fail to compile
through nvcc (`need 'typename' before ...` in `List_inl.h`); GCC 15 plus `-fpermissive` works:

```sh
CC=gcc-15 CXX=g++-15 CUDAHOSTCXX=g++-15 \
NVCC_APPEND_FLAGS="-Xcompiler=-fpermissive" \
TORCH_CUDA_ARCH_LIST="8.0" \
python setup.py build_ext --inplace
```

## Notes

- Sized for this card's exact budget. On other GPUs, rebuild with a different layer count from [`recipe/`](recipe/).
- The vision tower lives in pinned system RAM (`vision_offload: true`) to leave its VRAM to the experts.

## License and credits

- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen, under the Qwen Community
  License 1.0 (see [LICENSE](LICENSE)).
- Quantization: every tensor here comes from [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3).
- Inference: [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [TabbyAPI](https://github.com/theroyallab/tabbyAPI).
