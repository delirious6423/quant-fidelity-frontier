---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - fp4
  - mxfp4
  - deepseek
  - moe
  - quantized
  - multimodal
---

![DeepSeek-V4.1-Flash-1017-NVFP4: 475 GiB to 400 GiB, bit-exact routed experts, NVIDIA's NVFP4 tensor layout](assets/card.png)

# DeepSeek-V4.1-Flash-1017-NVFP4

This is the first NVFP4 checkpoint of
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
in the **same tensor layout and config schema as
[`nvidia/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)**.

- **Routed experts:** a lossless MXFP4 → NVFP4 transcode of the 1017 weights. The 4-bit codes
  are untouched; only the block scales are re-expressed. The per-projection activation scales
  are taken from NVIDIA's calibration of V4.1-Flash.
- **Engram tables:** stored as MXFP4, which is lossy (cosine **0.9934** vs the FP8 source).
  This is the same method and format as NVIDIA's.
- **Everything else:** byte-identical to upstream, including the MTP/DSpark experts.

The size goes from **475.24 GiB to 399.51 GiB**.

| | |
|---|---|
| base model | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision `1bae0e1088` |
| weights | 1017, routed experts transcoded bit-exactly to NVFP4 |
| activation scales (`input_scale`) | from `nvidia/DeepSeek-V4.1-Flash-NVFP4` (revision `c48dca01`), i.e. NVIDIA's ModelOpt calibration of V4.1-Flash; 12 fallback entries replaced (see [Activation scales](#activation-scales)) |
| layout | same tensor names, dtypes and shapes as `nvidia/DeepSeek-V4.1-Flash-NVFP4` (188,245 tensors); identical `config.json` and `hf_quant_config.json` |
| size | 399.51 GiB (428,966,398,040 bytes of tensors), 48 shards |
| FP4 tensor cores | NVIDIA Blackwell |

## What's in the checkpoint

| tensor group | upstream | this repo | block | size (upstream → here) | vs source |
|---|---|---|---|---:|---|
| routed experts (40 layers × 384 × w1/w2/w3) | MXFP4: E2M1 + E8M0 scale | **NVFP4 W4A4: E2M1 + E4M3 scale + FP32 global scale + FP32 `input_scale`** | 1×32 → **1×16** | 268.95 → 284.77 GiB | **weights bit-exact**; `input_scale` from NVIDIA's V4.1 calibration |
| Engram tables (layers 1 and 14) | FP8 E4M3 + E8M0 scale | **MXFP4: E2M1 + E8M0 scale** | 1×32 | 188.83 → **97.28 GiB** | lossy, cosine 0.9934 |
| MTP / DSpark experts (3 layers × 128 × 3) | MXFP4 | unchanged (in `ignore`, as in NVIDIA's) | 1×32 | 6.72 GiB | byte-identical |
| attention | FP8 E4M3 + E8M0 scale | unchanged | 32×32 | 4.80 GiB | byte-identical |
| shared experts | FP8 E4M3 + E8M0 scale | unchanged | 32×32 | 1.32 GiB | byte-identical |
| MTP / DSpark, non-expert | FP8 E4M3 32×32, BF16 | unchanged | 32×32 | 0.66 GiB | byte-identical |
| Engram projections | FP8 E4M3 + E8M0 scale | unchanged | 32×32 | 0.29 GiB | byte-identical |
| token embedding | BF16 | unchanged | — | 1.23 GiB | byte-identical |
| output head | BF16 | unchanged | — | 1.23 GiB | byte-identical |
| vision tower + aligner | BF16 | unchanged | — | 0.90 GiB | byte-identical |
| router | BF16 | unchanged | — | 0.15 GiB | byte-identical |
| norms, mHC parameters | F32 / BF16 | unchanged | — | 0.15 GiB | byte-identical |
| **total** | | | | **475.24 → 399.51 GiB** | |

**Expert tensors.** `<name>.weight` is `uint8 [out, in/2]`: E2M1 codes, two per byte, with
the even column in the low nibble. `<name>.weight_scale` is E4M3 `[out, in/16]` and
`<name>.weight_scale_2` is an FP32 scalar, so that:

```
w[r, c] = E2M1[code[r, c]] * weight_scale[r, c // 16] * weight_scale_2
```

`weight_scale_2` is 2^(largest source block exponent − 8). It is shared by w1 and w3 of an
expert, so the largest block scale lands on E4M3 256. This is the rule NVIDIA's checkpoint
uses; we confirmed it on sampled V4.1 experts, where their conversion also reproduces the
source exactly. `<name>.input_scale` is the FP32 static activation scale for W4A4.

**Engram tensors.** `layers.{1,14}.engram.embed.weight` is `uint8 [rows, 128]`, packed the
same way, and `.scale` is E8M0 `[rows, 8]`, one scale per 32 values.

## Why a transcode, not a re-quantization

DeepSeek ships the routed experts in MXFP4. NVFP4's 16-element blocks are a strict refinement
of MXFP4's 32-element blocks, and every E8M0 power-of-two scale is exactly representable as
E4M3 × a power-of-two global scale. So the same 4-bit codes map to NVFP4 with **zero error**.
Calibrated NVFP4 recipes choose weight scales to minimise error against a higher-precision
reference. Examples are [Four Over Six](https://arxiv.org/abs/2512.02010)'s per-block 4-or-6
scale selection in NVIDIA Model Optimizer, or MSE scale sweeps. Here the reference *is* the
shipped 4-bit weights, which the transcode already reproduces exactly. A search can at best
match it; any other choice of codes or scales adds error.

## Activation scales

The W4A4 experts need one static `input_scale` per projection. These 46,080 values come
from [`nvidia/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4)
(revision `c48dca01`). That is NVIDIA's Model Optimizer calibration of **V4.1-Flash**, of
which 1017 is DeepSeek's update. The weights are 1017's.

| | |
|---|---|
| coverage | 46,080 / 46,080 routed expert projections (46,068 unchanged, 12 replaced, see below) |
| range (NVIDIA's values) | min 4.36 × 10⁻⁵, median 1.31 × 10⁻³, max 5.58 × 10⁻² |
| placeholder values (1.0, ≤ 0, non-finite) | none |
| w1 vs w3 (shared input) | equal for all 15,360 experts |

Six experts received no calibration tokens in NVIDIA's run. For all three of their
projections, NVIDIA's checkpoint carries the model-wide maximum instead (their card: "18 of
46,080 projection entries used fallback scales"). We handle these 18 entries as follows:

- **w1 / w3** take the largest calibrated value among the other 383 experts in the same layer
  and projection. All experts in a layer see the same hidden states, so this is still a
  conservative estimate.
- **w2** keeps NVIDIA's value. Its input is the expert's own SwiGLU output, so the other
  experts say nothing about it, and an overestimate cannot clip.

| layer | expert | w1, w3: NVIDIA → shipped | w2: NVIDIA → shipped |
|---:|---:|---|---|
| 18 | 197 | 0.0271577 → **0.00115676** | 0.0558036 (kept) |
| 19 | 97 | 0.0271577 → **0.00166248** | 0.0558036 (kept) |
| 21 | 207 | 0.0271577 → **0.00148810** | 0.0558036 (kept) |
| 24 | 350 | 0.0271577 → **0.00173224** | 0.0558036 (kept) |
| 25 | 85 | 0.0271577 → **0.00179037** | 0.0558036 (kept) |
| 27 | 260 | 0.0271577 → **0.00325521** | 0.0558036 (kept) |

All other 46,068 values are NVIDIA's, unchanged. A calibration on 1017 itself will replace
all of the lifted scales.

## Verification

Every number below was measured on this checkpoint.

| check | result |
|---|---|
| reconstruct **every** expert weight in shards 4, 20, 21, 22, 24, 27, 28, 30, 33 and 42 (11,520 weights) from source and output | **0 mismatches over 135.90 × 10⁹ elements, `max\|Δ\| = 0`** |
| expert block scales inside E4M3's exact power-of-two window, all routed experts | **33,973,862,400 / 33,973,862,400** |
| w1 and w3 share one `weight_scale_2` | 15,360 / 15,360 experts |
| `input_scale` present and equal to the intended value | 46,080 / 46,080 |
| pass-through tensors byte-identical to upstream (incl. the 2,304 MTP/DSpark expert tensors) | **3,921 / 3,921** (17.46 GiB) |
| Engram cosine vs the FP8 source, ~1.5M rows per table sampled during the build | `layers.1` **0.993415**, `layers.14` **0.993446** |
| Engram cosine, 1,048,576 rows per table re-read from the written shards (64 windows spanning each table) | 0.993415 (min 0.989087) / 0.993444 (min 0.988480); mean relative error 0.121; 0 clipped values |
| index and shard headers | 188,245 tensors, each in exactly one shard; offsets contiguous to EOF; `total_size` = 428,966,398,040 B |
| tensor names, dtypes and shapes vs `nvidia/DeepSeek-V4.1-Flash-NVFP4` | **identical** (188,245 tensors, 0 differences) |
| `config.json` and `hf_quant_config.json` vs NVIDIA's | **byte-identical** |
| Engram method vs NVIDIA's MXFP4 Engram tables (run on the *original* V4.1-Flash, 65,536 sampled rows across both tables) | **byte-identical** weights and scales |

How it was made:

- The re-packer works one shard at a time. Shards are read with `pread()`, not memory-mapped,
  so the 94 GiB Engram tables are streamed and never held in memory.
- Experts keep their nibbles unchanged. Their E8M0 scales are split into E4M3 block scales and
  a shared power-of-two FP32 global scale.
- Engram scales are recomputed as the smallest power of two ≥ amax/6 per 32 values, so no
  value clips. Values are rounded to the nearest E2M1 value, with ties toward zero.
- The E2M1 table and nibble order come from DeepSeek's own `inference/convert.py`.
- The scripts are included: [`dsv41_fp4_stream.py`](./dsv41_fp4_stream.py) (re-packer; this
  build used `--global-rule nvidia --keep-mtp --input-scales`),
  [`finalize_repo.py`](./finalize_repo.py) (index and config),
  [`verify_transcode.py`](./verify_transcode.py) (bit-exactness and Engram cosine),
  [`check_repo.py`](./check_repo.py) (index, header and byte-identity checks), and
  [`test_engram_path.py`](./test_engram_path.py) / [`test_engram_shard.py`](./test_engram_shard.py)
  (cpu/cuda agreement and container round-trip).

## Serving

This repository has the **same tensor layout and config schema as
`nvidia/DeepSeek-V4.1-Flash-NVFP4`**. We checked that every tensor name, dtype and shape
matches, and `config.json` and `hf_quant_config.json` are identical. Load it the way NVIDIA
documents for theirs. Their card states:

- The MXFP4 Engram tables need a vLLM build that includes
  [vllm-project/vllm#61530](https://github.com/vllm-project/vllm/pull/61530). The
  `vllm/vllm-openai:deepseekv41-flash-0909` image predates it.
- The MXFP4 Engram tables are not supported in SGLang yet.
- Their tested hardware is NVIDIA GB300 (Blackwell).

NVIDIA's documented vLLM command, with this repository's ID:

```bash
vllm serve LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4 \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --language-model-only \
    --max-model-len 1048576 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}'
```

Sizes for planning: the whole checkpoint is 399.51 GiB. The two Engram tables are
97.28 GiB of that; NVIDIA's card notes that vLLM keeps them in pinned host memory by default.
Everything else is 302.23 GiB.

DeepSeek's bundled `inference/` code, included unmodified, does not load this layout. Its
`convert.py` converts only `int8` experts paired with `<name>.scale`, and its
`ParallelEngramEmbedding` allocates the tables as `float8_e4m3fn [rows, 256]`.

Upstream ships no Jinja chat template. DeepSeek's prompt-encoding reference is in
[`encoding/`](./encoding), copied unchanged.

## Engram

Engram is a hashed n-gram **gather**, not a GEMM. With `engram_layer_ids [1, 14]`,
`engram_max_ngram_size 4` and `engram_n_heads 8`, each token reads 3 n-gram sizes × 8 heads ×
2 layers = **48 rows**.

| | rows (layer 1 / layer 14) | bytes per row | per token | tables |
|---|---|---:|---:|---:|
| upstream FP8 | 384,006,168 / 384,016,682 | 264 | 12.4 KB | 188.83 GiB |
| this repo MXFP4 | same | **136** | **6.4 KB** | **97.28 GiB** |

To load a row: unpack 256 E2M1 values from 128 bytes, then multiply each group of 32 by its
E8M0 scale (`2^(e − 127)`). `config.json` declares the format with the keys vLLM PR #61530
reads: `engram_embed_quant: "mxfp4"`, `engram_embed_block_size: 32`,
`engram_embed_scale_dtype: "ue8m0"` and `engram_embed_layers: [1, 14]`.

The method is NVIDIA's. On the original V4.1-Flash, our Engram quantizer reproduced the MXFP4
tables in `nvidia/DeepSeek-V4.1-Flash-NVFP4` **byte for byte**: 65,536 of 65,536 sampled rows
across both tables, weights and scales.

## About the base model

DeepSeek describes V4.1-Flash-1017 as a minor update to
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), with gains in
reasoning, coding, agentic and visual tasks, and shorter outputs. Its benchmark table is on the
[upstream model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017); DeepSeek
measured it on the upstream weights.

The architecture is the same as V4.1-Flash: upstream `config.json` is identical, and all the
weights are new. It is a multimodal MoE with 40 layers, 384 routed experts + 1 shared expert,
6 routed experts per token, Engram tables at layers 1 and 14, 3 MTP layers, and a 1M-token
context.

## Credits

- **DeepSeek** made the model and released it under the MIT license. Their notice,
  Copyright (c) 2023 DeepSeek, is kept in [`LICENSE`](./LICENSE). All model credit is theirs.
- **NVIDIA** designed the layout this repository follows. The ModelOpt config schema,
  `config.json` and `hf_quant_config.json` are taken verbatim from
  [`nvidia/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4),
  as are the activation scales (their V4.1-Flash calibration, with the 12 replacements
  above) and the MXFP4 Engram format.
  The weights in this repository come from the 1017 release, not from NVIDIA.

## License

MIT, inherited from the base model. See [`LICENSE`](./LICENSE).

---

Published by [LibertAI](https://libertai.io). Other DeepSeek-V4.1 checkpoints from
LibertAIDAI:

- [DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4)
- [DeepSeek-V4.1-Flash-REAP-272E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E)
- [DeepSeek-V4.1-Flash-REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E)
