---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: 6block
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
language:
- en
- zh
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- deepseek
- moe
- quantized
---

# DeepSeek-V4-Flash-0731 GGUF

GGUF quantizations of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), made with llama.cpp.

Chinese version: [README_zh.md](https://huggingface.co/6block/DeepSeek-V4-Flash-0731-GGUF/blob/main/README_zh.md)

## Why there is no Q8_0 / Q6_K / Q5_K_M / Q4_K_M

**The expert layers ship as FP8** (`quantization_config.fmt = e4m3`, block 128x128).
After conversion to GGUF they land in MXFP4, taking 137.06 GiB (94.1%) of the
156.4 GB master; everything else adds up to roughly 8.6 GiB.

Re-quantizing 4.25 bpw expert weights up to 8.5 bpw only inflates the file — it
cannot recover precision that was already discarded upstream. Measured with
`--dry-run` against the same master:

| Global type | Size | vs master |
|---|---|---|
| Q8_0 | 281.4 GiB | +93% |
| Q6_K | 217.3 GiB | +49% |
| Q5_K_M | 187.8 GiB | +29% |
| Q4_K_M | 160.0 GiB | +10% |
| IQ4_XS | 142.5 GiB | -2% |

So this repo starts at Q3_K_M. If you want near-lossless, use the original
weights directly — an intermediate tier buys you nothing here.

## Why there is no IQ2_M

It was built, and it failed testing. Quantized with the same settings, IQ2_M came
out at PPL = **1.08e9**, while the smaller IQ2_XXS sits at 7.47. Every chunk from
the very first one is a 9-digit number, reproducible on both CPU and GPU. Tensor
count and the per-layer type map match the working tiers line for line, so this
is not a misconfiguration — the `IQ2_S` code path fails on this model (likely
related to imatrix columns left empty by 256-expert top-6 sparse routing).
`IQ2_XS` covers the same size range instead.

## Quantizations

| Tier | Size | Shards | PPL (wikitext-2, n_ctx=512, 12 chunks) |
|---|---|---|---|
| master (expert layers = factory MXFP4) | 145.6 GiB | 4 | **2.7562 ± 0.12273** |
| `Q3_K_M` | 129.9 GiB | 4 | 3.1992 ± 0.14989 |
| `IQ3_XXS` | 105.9 GiB | 4 | 3.7911 ± 0.18074 |
| `Q2_K` | 100.5 GiB | 4 | 4.4583 ± 0.22758 |
| `IQ2_XS` | 82.1 GiB | 4 | 5.6371 ± 0.29464 |
| `IQ2_XXS` | 74.3 GiB | 4 | 7.4718 ± 0.41588 |
| `IQ1_M` | 64.6 GiB | 4 | 10.2099 ± 0.58384 |
| `IQ1_S` | 58.8 GiB | 4 | 11.7962 ± 0.69607 |

The master row is not a file in this repo. It is listed so the numbers above have
a reference point: its expert layers are the factory MXFP4 weights with no further
quantization, which is the accuracy ceiling for this model.

Every tier is imatrix-calibrated (220 chunks, mixed code / English / Chinese), and
the non-expert layers are protected:

| Tensors | Type | Reason |
|---|---|---|
| `attn_*` | Q8_0 | source weights are already FP8; compressing further gains nothing |
| `ffn_*_shexp` | Q8_0 | the shared expert runs on every token |
| `indexer*` / `*compressor*` | Q8_0 | sparse-attention index; compressing it picks the wrong KV |
| `ffn_gate_inp` | F32 | router; compressing it routes to the wrong experts |
| `hc_*` | F32 | per-layer control signals, negligible size |
| `token_embd` / `output` | Q6_K | a global type would otherwise squeeze these hard |

## Usage

```bash
# Plain decoding — fine across multiple GPUs
llama-cli -hf 6block/DeepSeek-V4-Flash-0731-GGUF:Q3_K_M -ngl 99 -c 8192

# Or point at the files yourself
llama-cli -m DeepSeek-V4-Flash-0731-Q3_K_M-00001-of-00004.gguf -ngl 99 -c 8192
```

For sharded tiers, download all shards and point at `-00001-of-*`; llama.cpp finds
the rest on its own.

### DSpark speculative decoding

The base model ships a DSpark speculative-decoding module. This repo exports it
separately as `dspark-DeepSeek-V4-Flash-0731-BF16.gguf` (10.1 GiB, 81 tensors,
architecture `dflash`). One draft file serves every tier.

```bash
# --spec-type is required, and everything has to fit on ONE GPU (see below)
CUDA_VISIBLE_DEVICES=0 llama-cli \
  -m DeepSeek-V4-Flash-0731-IQ1_S-00001-of-00004.gguf \
  -md dspark-DeepSeek-V4-Flash-0731-BF16.gguf \
  --spec-type draft-dspark \
  -ngl 99 -ngld 99 -c 2048
```

Two things worth knowing, both measured here on 8xH100 with llama.cpp 4ed2b13:

- **`-md` on its own does not turn it on.** Without `--spec-type draft-dspark` the
  draft file loads, the log prints `no implementations specified for speculative
  decoding`, and the slot reports `"speculative":false` — you get plain decoding at
  full speed and no error. `--spec-default` does not help either; it only enables
  `ngram-mod` and never touches the draft model. With the flag set, the log shows
  `"speculative":true` and repeated `called impl draft-dspark` lines.
- **One GPU at a time, for now.** Split over 2 or more GPUs it aborts on
  `GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)` (the cap is 30): the
  graph wants more cross-backend copies than the scheduler allows. Pinning the draft
  to one device with `-devd` does not avoid it, because the target model is what gets
  split. This is a llama.cpp scheduler limit, not a problem with these files. So the
  tier plus the 10.1 GiB draft has to fit in a single card — on an 80 GB card that
  means `IQ1_S`; the larger tiers need plain decoding until the limit is lifted.

## About the PPL numbers

Measured in-house with one fixed setup: wikitext-2 test, `n_ctx=512`, 12 chunks,
`-ngl 99`, and the master baseline run through the exact same command.
**These numbers are only comparable within this table.** Do not compare them
against PPL figures published by other repos — different corpora and chunk counts
make the absolute values meaningless across setups.

## License

MIT, inherited from [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). Quantized by 6block.
