---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
library_name: mlx
tags:
- mlx
- omlx
- oqel
- quantized
- experimental
---

# Qwen3.8-Flash-Next-oQ2eL

> [!CAUTION]
> **2-bit is a large quality step down.** KL to BF16 is about 7 times the 4-bit build's. Use it only where memory rules out 3 or 4 bits.


> [!WARNING]
> **Release 2 in progress.** This repository is new in release 2. Its files are uploading. Until this note is gone, the files here may be incomplete: do not download yet. Install and speed sections follow with the release.

> [!WARNING]
> **Experimental.** This checkpoint needs a fork of oMLX that is not upstream yet ([yohann-bearzi/omlx](https://github.com/yohann-bearzi/omlx/tree/quant-plugins-poc), branch `quant-plugins-poc`, release-2 tag to follow) and the [omlx-oqel](https://github.com/yohann-bearzi/omlx-oqel) plugin 0.1.0. Stock oMLX, mlx-lm and mlx-vlm cannot load it. The oQeL layout (revision 1) and the plugin's kernels may change, and a later plugin may refuse this checkpoint, which would then be quantized again.

## What changes in release 2

- **The per-layer embedding (PLE) table is FP8**, grafted from Qwen's official FP8 table, instead of BF16. Size: **about 104 GB**.
- **Built on the fork after its merge of upstream oMLX** (e1a402cf), including oMLX's oQe rounding refinement (#5292).
- **Compared against a refined stock build.** The stock oQ2e it is measured against uses the same refinement and the same FP8 table.

## What it is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 2 in its oQe mode, with omlx-oqel 0.1.0:

- **The routed experts are 2-bit oQeL**: 147 matrices (gate, up and down of the 48 MoE layers and of the MTP layer), each on a 4-level table fitted to this model's experts, in groups of 16 weights with an E2M5 scale byte each and one bfloat16 scale per row.
- **Everything else matches the refined stock oQ2e build**: MLX affine 8-bit, and the FP8 PLE table.

## Measured so far

Against the refined stock **oQ2e** written from the same source, calibration, bit plan and FP8 PLE table (only the routed experts' format differs). M3 Ultra, release-2 runtime:

| | oQ2eL | oQ2e (refined stock) |
|---|---|---|
| KL to BF16, code / prose | **0.1067 / 0.2008** | 0.1369 / 0.2436 |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval) | 1.497 [1.354, 1.643] | 1.419 [1.292, 1.559] |

KL is full-vocabulary KL(BF16 ‖ quantized) per token, averaged over two 16,384-token documents (code; English prose), against the same BF16 reference as release 1. MTP acceptance is the mean over 32 fixed prompts of accepted draft tokens per verify cycle (greedy, 128 tokens each), MTP adaptive, the same statistic as release 1.

Decode and prefill speed on both Macs: in progress.

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). omlx-oqel is Apache-2.0.
