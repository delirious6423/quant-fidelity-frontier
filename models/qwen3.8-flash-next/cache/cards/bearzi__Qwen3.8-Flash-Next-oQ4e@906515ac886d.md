---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
library_name: mlx
tags:
- mlx
- omlx
- quantized
---

# Qwen3.8-Flash-Next-oQ4e

This is the stock baseline for [bearzi/Qwen3.8-Flash-Next-oQ4eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL): the same source, calibration and FP8 PLE table, with every quantized matrix in stock MLX affine. It is **release 3** of the baseline; the quant recipe and refinement are unchanged from release 2; the runtime changed.

## What it is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 4 in its oQe mode, with oMLX's oQe rounding refinement (#5292):

- **Every quantized matrix is stock MLX affine**: the routed experts are 4-bit
  and everything else is 8-bit, the same bit plan as the oQ4eL build.
- **The PLE table is FP8**, taken from Qwen's official FP8 table.
- **No plugin is needed.** It was verified to load and run on the oMLX fork's
  `oqel-poc-r3` tag with no plugin installed. Upstream oMLX should load it
  too (standard affine with an FP8 PLE table), but that was not tested.
- **Size: 134.1 GB.**

## What changed in release 3

- **Unified dense skeleton.** Stock affine and plugin formats now run through one
  dense skeleton chosen at load; the per-format dense kernels and the stock-only
  dense path were deleted, not kept as fallbacks.
- **Per-chip tuning tables.** The skeleton's tile geometry and tensor-unit route
  (steel on the M3 Ultra, NAX on the M5 Max) are compile-time parameters picked
  from versioned per-chip tables produced by the offline autotuner; upstream
  MLX speed is the floor, not the target.
- **MTP under format loads.** The MTP head's dense, attention, hyper-connection
  and routed layers load as format layers and run draft and verify through the
  same faces; the contract's verify-versus-decode gates are compared.
- **Metal Hessian refine (writer).** oQe's rounding refinement (#5292) now
  runs on the GPU, so plugin formats refine their codes on the shrunk
  calibration Hessian instead of being skipped; it is a writer feature and
  does not change these published checkpoints.
- **Numerics contract.** The contract replaces bit-exactness with FP32-reference and KL
  bars; same-chip run-to-run determinism, fixed reduction order and the FP32
  nonlinearities are unit-tested, while accumulation order, tiling and split-K
  are free within the bars.

## Install and run

```sh
python3.13 -m venv omlx-env
. omlx-env/bin/activate
git clone --branch oqel-poc-r3 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
hf download bearzi/Qwen3.8-Flash-Next-oQ4e --local-dir ./models/Qwen3.8-Flash-Next-oQ4e
omlx serve --model-dir ./models
```

On a Mac with 128 GB of memory, keep the PLE table on SSD (`"qwen4_ple_ssd_offload": true` in `model_settings.json`, as on the [oQ4eL card](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL)).

## Measured

Against the other build of the same source, calibration, bit plan and FP8 PLE
table (oQ4eL):

| | oQ4e (this) | oQ4eL |
|---|---|---|
| KL to BF16, code / prose | 0.01832 / 0.03358 | 0.01576 / 0.02905 |
| decode KL, vs stock MLX, code / prose | 0.00000 / 0.00000 | 0.00000 / 0.00000 |
| decode top-1 agreement, vs stock MLX, code / prose | 1.00000 / 1.00000 | 1.00000 / 1.00000 |
| change in KL to BF16, vs stock MLX, code / prose | 0 / 0 | 0 / 0 |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval), M3 Ultra | 1.604 [1.459, 1.747] | 1.586 [1.432, 1.746] |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval), M5 Max | 1.547 [1.389, 1.704] | 1.581 [1.418, 1.743] |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M3 Ultra | 76.26 / 60.46 / 43.85 | 75.90 / 60.18 / 43.27 |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M5 Max | 72.62 / 57.87 / 41.10 | 71.32 / 56.80 / 40.64 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M3 Ultra | 93.22 / 62.59 / 39.62 | 94.73 / 63.46 / 39.18 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M5 Max | 90.37 / 60.06 / 37.44 | 88.36 / 60.70 / 37.24 |
| Prefill tok/s 2k / 8k, M3 Ultra | 1,319 / 1,441 | 1,307 / 1,427 |
| Prefill tok/s 2k / 8k, M5 Max | 2,185 / 2,795 | 2,169 / 2,768 |
| Time to first token, cold / warm, M3 Ultra | 1.80 / 1.56 s | 1.80 / 1.57 s |
| Time to first token, cold / warm, M5 Max | 1.12 / 0.94 s | 1.13 / 0.95 s |

*The oQeL column's decode and change rows compare release 3 with release 2
(tag `oqel-poc-r2`, plugin 0.1.0), same checkpoint and Mac; the stock
column's compare with stock MLX (0.0 / 1.0, bitwise).*

How each row was measured:

- **KL to BF16:** full-vocabulary KL(BF16 || quantized) per token, averaged
  over the code and English prose documents. The value is release 2's, measured
  on the M5 Max; release 3 is bitwise identical to stock MLX here, so it is
  unchanged. The change row below is zero.
- **Decode KL / top-1, vs stock MLX:** the format path against stock MLX on the
  same checkpoint and Mac; 0.0 / 1.0 is bitwise identical.
- **Change in KL to BF16, vs stock MLX:** the format path's KL to BF16 minus
  stock MLX's, per document; 0.0 is bitwise. On this stock card the oQeL
  column's decode and change values use the release-2 reference (0.0, bitwise);
  the stock column's use stock MLX (0.0, bitwise).

- **MTP acceptance:** accepted draft tokens per verify cycle, mean over the 32
  fixed prompts (code, prose, chat), greedy, 128 tokens each, MTP adaptive, on
  each Mac.
- **Speed rows:** decode at 1, 2 and 4 concurrent streams and prefill at
  2,048 and 8,192 tokens, on each Mac, 16 timed rounds per MTP mode in fresh
  server processes alternating the two builds; the table gives medians.
  The release-3 harness balances the arm and mode order and uses a four-pair
  decode2 panel. Time to
  first token is cold and warm. The Macs are a Mac Studio M3 Ultra (512 GB) and
  a Mac Studio M5 Max (128 GB, with the PLE table on SSD).

## Current limits

- **MTP adaptive vs off.** Adaptive speeds decode at 1 and 2 streams but is
  slower at 4: 93.22 / 62.59 / 39.62 adaptive vs
  76.26 / 60.46 / 43.85 off on the M3 Ultra, and
  90.37 / 60.06 / 37.44 vs
  72.62 / 57.87 / 41.10 on the M5 Max.
- **Memory.** On the 128 GB M5 Max, keep the PLE table on SSD (the install block
  shows the setting).
- This is the stock baseline; the plugin comparison is on its oQeL card.

## Not measured

Public benchmark sets and task accuracy (MMLU, code benchmarks and the like),
contexts longer than 8,192 tokens, more than 4 concurrent streams, and Macs
other than the two above.

## Other levels

[oQ4eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL) ·
[oQ4e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4e) ·
[oQ3eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3eL) ·
[oQ3e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3e) ·
[oQ2eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL) ·
[oQ2e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2e)

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).

