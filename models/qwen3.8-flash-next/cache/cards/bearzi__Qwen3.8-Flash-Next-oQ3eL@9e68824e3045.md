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

# Qwen3.8-Flash-Next-oQ3eL

> [!WARNING]
> **Experimental.** This checkpoint needs a fork of oMLX that is not upstream yet
> ([yohann-bearzi/omlx](https://github.com/yohann-bearzi/omlx), tag `oqel-poc-r3`)
> and the [omlx-oqel](https://github.com/yohann-bearzi/omlx-oqel) plugin
> `0.2.0`. Stock oMLX, mlx-lm and mlx-vlm cannot load it. It is a proof
> of concept of a quant-format plugin interface for oMLX; the oQeL layout and the
> plugin's kernels may change, and a later plugin may refuse this checkpoint.

This is **release 3** of the oQeL proof of concept. The source, calibration, bit plan and FP8 PLE table are the same as release 2, and the checkpoints are unchanged; the runtime, the plugin's kernels and the served paths changed.

## What it is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 3 in its oQe mode:

- **The routed experts are 3-bit oQeL**: the gate, up and down matrices of the MoE
  layers and of the MTP layer, each on an 8-level table fitted to this model's
  experts, in groups of 16 weights with an E2M5 scale byte each and one bfloat16
  scale per row. The `group_size: 64` in `config.json` is oQ's nominal recipe;
  the oQeL layout does not read it.
- **Everything else is the same as the refined stock [oQ3e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3e) build**: MLX affine 8-bit.
- **The PLE table is FP8**, taken from Qwen's official FP8 table.
- **Size: 119 GB.**

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

## Install

You need an Apple Silicon Mac with Python 3.11 to 3.13, git, enough free disk for the checkpoint and a network connection:

```sh
python3.13 -m venv oqel-env
. oqel-env/bin/activate
git clone --branch oqel-poc-r3 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
git clone --branch v0.2.0 https://github.com/yohann-bearzi/omlx-oqel omlx-oqel
python -m pip install ./omlx-oqel
hf download bearzi/Qwen3.8-Flash-Next-oQ3eL --local-dir ./models/Qwen3.8-Flash-Next-oQ3eL
```

**On a Mac with 128 GB of memory, keep the PLE table on SSD.** Put this
`model_settings.json` in the server's base path, or set the same options in the
dashboard:

```json
{"version": 2, "models": {"Qwen3.8-Flash-Next-oQ3eL": {"mtp_enabled": true, "qwen4_ple_ssd_offload": true, "a8_prefill_enabled": false, "force_sampling": false}}}
```

Then serve it as usual (OpenAI-compatible, port 8000):

```sh
omlx serve --model-dir ./models
```

## Measured

Against the other build of the same source, calibration, bit plan and FP8 PLE
table (oQ3e):

| | oQ3eL | oQ3e |
|---|---|---|
| KL to BF16, code / prose | 0.03689 / 0.06506 | 0.04383 / 0.07933 |
| decode KL, code / prose | 0.00000 / 0.00000 | 0.00000 / 0.00000 |
| decode top-1 agreement, code / prose | 1.00000 / 1.00000 | 1.00000 / 1.00000 |
| change in KL to BF16, code / prose | 0 / 0 | 0 / 0 |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval), M3 Ultra | 1.600 [1.447, 1.757] | 1.572 [1.425, 1.730] |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval), M5 Max | not measured (acceptance measured on the M3 Ultra only) | not measured (acceptance measured on the M3 Ultra only) |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M3 Ultra | 77.81 / 62.53 / 45.16 | 77.84 / 62.53 / 45.09 |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M5 Max | not measured (speed measured on the M3 Ultra only) | not measured (speed measured on the M3 Ultra only) |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M3 Ultra | 114.14 / 65.10 / 41.42 | 112.10 / 62.87 / 41.37 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M5 Max | not measured (speed measured on the M3 Ultra only) | not measured (speed measured on the M3 Ultra only) |
| Prefill tok/s 2k / 8k, M3 Ultra | 1,285 / 1,409 | 1,300 / 1,424 |
| Prefill tok/s 2k / 8k, M5 Max | not measured (speed measured on the M3 Ultra only) | not measured (speed measured on the M3 Ultra only) |
| Time to first token, cold / warm, M3 Ultra | 1.83 / 1.60 s | 1.82 / 1.58 s |
| Time to first token, cold / warm, M5 Max | not measured (speed measured on the M3 Ultra only) | not measured (speed measured on the M3 Ultra only) |

Speed for this level was measured on the M3 Ultra only; the M5 Max cells are marked not measured.

*The oQeL column's decode and change rows compare release 3 with release 2
(tag `oqel-poc-r2`, plugin 0.1.0), same checkpoint and Mac; the stock
column's compare with stock MLX (0.0 / 1.0, bitwise).*

How each row was measured:

- **KL to BF16:** full-vocabulary KL(BF16 || quantized) per token, averaged
  over the code and English prose documents. The value is release 2's, measured
  on the M5 Max; release 3 is bitwise identical on the same Mac, so it is
  unchanged. The change row below is zero.
- **Decode KL / top-1, release 3 vs release 2:** the same checkpoint and Mac
  through the release-3 format path and through the release-2 runtime (tag
  `oqel-poc-r2`, plugin 0.1.0); 0.0 / 1.0 means identical.
- **Change in KL to BF16 vs release 2:** the release-3 build's KL to BF16 minus
  the release-2 build's, per document; 0.0 means identical.

- **MTP acceptance:** accepted draft tokens per verify cycle, mean over the 32
  fixed prompts (code, prose, chat), greedy, 128 tokens each, MTP adaptive, on
  each Mac.
- **Speed rows:** decode at 1, 2 and 4 concurrent streams and prefill at
  2,048 and 8,192 tokens, on each Mac, 16 timed rounds per MTP mode in fresh
  server processes alternating the two builds; the table gives medians.
  The release-3 harness balances the arm and mode order and uses a four-pair
  decode2 panel. The 3-bit M3 Ultra decode1/adaptive cell uses a 9-prompt
  c2-tools05-r3 panel; the other rows use the 16-round c2-tools04-r3 site. Time to
  first token is cold and warm. The Macs are a Mac Studio M3 Ultra (512 GB) and
  a Mac Studio M5 Max (128 GB, with the PLE table on SSD).

## Current limits

- **MTP adaptive vs off.** On the M3 Ultra, adaptive speeds decode at 1 and 2
  streams but is slower at 4: 114.14 / 65.10 / 41.42 adaptive vs
  77.81 / 62.53 / 45.16 off.
- **Against stock.** Prefill is about 1% slower than stock
  (1,285 / 1,409 vs 1,300 / 1,424) and
  cold TTFT is slightly slower (1.83 / 1.60 s vs
  1.82 / 1.58 s); decode with MTP off is at parity.
- **Memory.** On the 128 GB M5 Max, keep the PLE table on SSD (the install block
  shows the setting).

## Not measured

Public benchmark sets and task accuracy (MMLU, code benchmarks and the like),
contexts longer than 8,192 tokens, more than 4 concurrent streams,
M5 Max speed and acceptance for this level (speed measured on the M3 Ultra
only), and Macs other than the two above.

## Other levels

[oQ4eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL) ·
[oQ4e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4e) ·
[oQ3eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3eL) ·
[oQ3e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3e) ·
[oQ2eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL) ·
[oQ2e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2e)

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
omlx-oqel is Apache-2.0.
