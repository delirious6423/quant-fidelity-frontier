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

# Qwen3.8-Flash-Next-oQ2e

This is the stock baseline for [bearzi/Qwen3.8-Flash-Next-oQ2eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL): the same source, calibration and FP8 PLE table, with every quantized matrix in stock MLX affine. It is **release 4** of the baseline; the checkpoint is unchanged from release 2/3; the runtime changed.

## What it is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 2 in its oQe mode, with oMLX's oQe rounding refinement (#5292):

- **Every quantized matrix is stock MLX affine**: the routed experts are 2-bit and everything else is 8-bit, the same bit plan as the oQ2eL build.
- **The PLE table is FP8**, taken from Qwen's official FP8 table.
- **No plugin is needed.** Verified to load and run on the oMLX fork's `oqel-poc-r4` tag with no plugin installed.
- **Size: 104 GB.**

## What changed in release 4

- **VFAST batched verify & unarmed row-exact verify.** Multi-row verification on `qwen4_exp` now batches through construction-time `FormatLinear` verify-row schedules and per-chip DATA tables instead of forcing single-row replay, lifting 2-stream adaptive decode by ~12–17% across levels.
- **GDN partial-commit recurrence & 3-row short causal attention.** Partial MTP commits compute only the accepted key/value recurrence, and 3-row verify runs a single causal attention launch up to 256 keys.
- **Built-in `oQh` Hadamard transform & producer fusion.** Added `OqhFormat`, `OqhAffineFormat`, versioned `InputTransform` resolved at construction time, shared tensors, concrete-class dense MLP fusion allowlist, and the unified `encode_layer` writer hook (adopted by `omlx-oqel` `0.3.0`).
- **Ragged `left_padding` GPU sync removal & `FormatLinear` `hc_fused` immutable per-layer plan.**
- **Decode metric update.** Decode rows are now judged on server-side **tokens/s** (usage-counted completions), replacing release 3's SSE frames/s.

## Install and run

```sh
python3.13 -m venv omlx-env
. omlx-env/bin/activate
git clone --branch oqel-poc-r4 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
hf download bearzi/Qwen3.8-Flash-Next-oQ2e --local-dir ./models/Qwen3.8-Flash-Next-oQ2e
omlx serve --model-dir ./models
```

On a Mac with 128 GB of memory, keep the PLE table on SSD (`"qwen4_ple_ssd_offload": true` in `model_settings.json`, as on the [oQ2eL card](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL)).

## Measured

Against the other build of the same source, calibration, bit plan and FP8 PLE table (oQ2eL):

| | oQ2e (this) | oQ2eL |
|---|---|---|
| KL to BF16, code / prose (M5 Max baseline) | 0.13686 / 0.24361 | 0.10672 / 0.20079 |
| Verify vs decode KL (N1/N2, floor 4.505e-3 / 1.336e-2) | within cross-chip floor | within cross-chip floor |
| MTP accepted drafts per verify cycle (32 prompts), M3 Ultra | 1.402 [1.285, 1.533] | 1.517 [1.399, 1.651] |
| MTP accepted drafts per verify cycle (32 prompts), M5 Max | not measured (acceptance measured on the M3 Ultra only) | not measured (acceptance measured on the M3 Ultra only) |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M3 Ultra | 79.59 / 65.19 / 47.98 | 79.83 / 65.70 / 48.86 |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M5 Max | not measured (speed measured on the M3 Ultra only) | not measured (speed measured on the M3 Ultra only) |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M3 Ultra | 98.76 / 71.95 / 44.23 | 97.12 / 72.20 / 45.14 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M5 Max | not measured (speed measured on the M3 Ultra only) | not measured (speed measured on the M3 Ultra only) |
| Prefill tok/s 2k / 8k, M3 Ultra | 1,319 / 1,442 | 1,320 / 1,446 |
| Prefill tok/s 2k / 8k, M5 Max | not measured (speed measured on the M3 Ultra only) | not measured (speed measured on the M3 Ultra only) |
| Time to first token, cold / warm, M3 Ultra | 1.77 / 1.56 s | 1.77 / 1.56 s |
| Time to first token, cold / warm, M5 Max | not measured (speed measured on the M3 Ultra only) | not measured (speed measured on the M3 Ultra only) |

*Decode rows are measured in server-side **tokens/s** (usage-counted completions), not release 3's SSE frames/s. Release-4 site numbers were measured on the release-4 runtime and qualified by the freeze kernel fix (tag `oqel-poc-r4`).*

Speed for this level was measured on the M3 Ultra only; the M5 Max cells are marked not measured.

How each row was measured:

- **KL to BF16:** full-vocabulary KL(BF16 || quantized) per token over the 16,384-token code and English prose documents. Release-4 whole-model KL deltas vs pre-VFAST are 0.0 on the M5 Max and within the cross-chip C4b bars on the M3 Ultra.
- **Verify vs decode KL (N1/N2):** 128 verify windows (584 positions) comparing unarmed batched verify against 1-row decode; all rows pass well below the cross-chip MLX floor (`4.505e-3` code / `1.336e-2` prose).
- **MTP acceptance:** accepted draft tokens per verify cycle, mean over the 32 fixed prompts (code, prose, chat), greedy, 128 tokens each, MTP adaptive.
- **Speed rows:** decode at 1, 2 and 4 concurrent streams (tokens/s) and prefill at 2,048 and 8,192 tokens, 16 timed rounds per MTP mode in fresh server processes alternating the two builds; the table gives medians. Time to first token is cold and warm. The Macs are a Mac Studio M3 Ultra (512 GB) and a Mac Studio M5 Max (128 GB, with the PLE table on SSD).

## Current limits

- **MTP adaptive vs off.** Adaptive speeds decode strongly at 1 and 2 streams but is slower at 4: 98.76 / 71.95 / 44.23 adaptive vs 79.59 / 65.19 / 47.98 off on the M3 Ultra.
- **Memory.** On the 128 GB M5 Max, keep the PLE table on SSD (the install block shows the setting).
- This is the stock baseline; the plugin comparison is on its oQeL card.

## Not measured

Public benchmark sets and task accuracy (MMLU, code benchmarks and the like), contexts longer than 8,192 tokens, more than 4 concurrent streams, M5 Max speed for this level (measured on the M3 Ultra only), and Macs other than the two above.

## Other levels

[oQ4eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL) ·
[oQ4e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4e) ·
[oQ3eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3eL) ·
[oQ3e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3e) ·
[oQ2eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL) ·
[oQ2e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2e)

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
