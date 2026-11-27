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

# Qwen3.8-Flash-Next-oQ4eL

> [!WARNING]
> **Experimental.** This checkpoint needs a fork of oMLX that is not upstream yet
> ([yohann-bearzi/omlx](https://github.com/yohann-bearzi/omlx), tag `oqel-poc-r4`)
> and the [omlx-oqel](https://github.com/yohann-bearzi/omlx-oqel) plugin
> `0.3.0`. Stock oMLX, mlx-lm and mlx-vlm cannot load it. It is a proof
> of concept of a quant-format plugin interface for oMLX; the oQeL layout and the
> plugin's kernels may change, and a later plugin may refuse this checkpoint.

This is **release 4** of the oQeL proof of concept. The source, calibration, bit plan and FP8 PLE table are unchanged from release 2/3; the runtime, verify schedule, transform framework, and plugin writer hook changed.

## What it is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 4 in its oQe mode:

- **The routed experts are 4-bit oQeL**: the gate, up and down matrices of the MoE
  layers and of the MTP layer, each on a 16-level table fitted to this model's
  experts, in groups of 16 weights with an E2M5 scale byte each and one
  bfloat16 scale per row. The `group_size: 64` in `config.json` is oQ's nominal
  recipe; the oQeL layout does not read it.
- **Everything else is the same as the refined stock [oQ4e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4e) build**: MLX affine 8-bit.
- **The PLE table is FP8**, taken from Qwen's official FP8 table.
- **Size: 134.3 GB.**

## What changed in release 4

- **VFAST batched verify & unarmed row-exact verify.** Multi-row verification on `qwen4_exp` now batches through construction-time `FormatLinear` verify-row schedules and per-chip DATA tables instead of forcing single-row replay, lifting 2-stream adaptive decode by ~12–17% across levels.
- **GDN partial-commit recurrence & 3-row short causal attention.** Partial MTP commits compute only the accepted key/value recurrence, and 3-row verify runs a single causal attention launch up to 256 keys.
- **Built-in `oQh` Hadamard transform & producer fusion.** Added `OqhFormat`, `OqhAffineFormat`, versioned `InputTransform` resolved at construction time, shared tensors, concrete-class dense MLP fusion allowlist, and the unified `encode_layer` writer hook (adopted by `omlx-oqel` `0.3.0`).
- **Ragged `left_padding` GPU sync removal & `FormatLinear` `hc_fused` immutable per-layer plan.**
- **Decode metric update.** Decode rows are now judged on server-side **tokens/s** (usage-counted completions), replacing release 3's SSE frames/s.

## Install

You need an Apple Silicon Mac with Python 3.11 to 3.13, git, enough free disk for the checkpoint and a network connection:

```sh
python3.13 -m venv oqel-env
. oqel-env/bin/activate
git clone --branch oqel-poc-r4 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
git clone --branch v0.3.0 https://github.com/yohann-bearzi/omlx-oqel omlx-oqel
python -m pip install ./omlx-oqel
hf download bearzi/Qwen3.8-Flash-Next-oQ4eL --local-dir ./models/Qwen3.8-Flash-Next-oQ4eL
```

**On a Mac with 128 GB of memory, keep the PLE table on SSD.** Put this
`model_settings.json` in the server's base path, or set the same options in the
dashboard:

```json
{"version": 2, "models": {"Qwen3.8-Flash-Next-oQ4eL": {"mtp_enabled": true, "qwen4_ple_ssd_offload": true, "a8_prefill_enabled": false, "force_sampling": false}}}
```

Then serve it as usual (OpenAI-compatible, port 8000):

```sh
omlx serve --model-dir ./models
```

## Measured

Against the other build of the same source, calibration, bit plan and FP8 PLE table (oQ4e):

| | oQ4eL | oQ4e |
|---|---|---|
| KL to BF16, code / prose (M5 Max baseline) | 0.01576 / 0.02905 | 0.01832 / 0.03358 |
| Verify vs decode KL (N1/N2, floor 4.505e-3 / 1.336e-2) | within cross-chip floor (N5 M3 delta +1.60e-4 / +9.08e-4) | 0.00123 / 0.00243 (M5), 0.00119 / 0.00307 (M3) |
| MTP accepted drafts per verify cycle (32 prompts), M3 Ultra | 1.599 (N4 R4; R3 CI 1.586 [1.432, 1.746]) | 1.627 (N4 R4; R3 CI 1.604 [1.459, 1.747]) |
| MTP accepted drafts per verify cycle (32 prompts), M5 Max | 1.581 [1.418, 1.743] (R3, unchanged checkpoint) | 1.638 (N4 R4; R3 CI 1.547 [1.389, 1.704]) |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M3 Ultra | 75.76 / 60.31 / 43.32 | 76.14 / 60.68 / 43.77 |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M5 Max | 71.51 / 57.11 / 40.92 | 72.62 / 58.23 / 41.28 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M3 Ultra | 93.92 / 71.57 / 41.06 | 91.49 / 71.59 / 41.60 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M5 Max | 90.17 / 67.98 / 37.61 | 91.27 / 68.03 / 37.58 |
| Prefill tok/s 2k / 8k, M3 Ultra | 1,301 / 1,428 | 1,313 / 1,440 |
| Prefill tok/s 2k / 8k, M5 Max | 2,147 / 2,739 | 2,165 / 2,769 |
| Time to first token, cold / warm, M3 Ultra | 1.79 / 1.58 s | 1.78 / 1.57 s |
| Time to first token, cold / warm, M5 Max | 1.13 / 0.96 s | 1.12 / 0.95 s |

*Decode rows are measured in server-side **tokens/s** (usage-counted completions), not release 3's SSE frames/s. Release-4 site numbers were measured on the release-4 runtime and qualified by the freeze kernel fix (tag `oqel-poc-r4`, plugin `0.3.0`).*

*M5 adaptive 1-stream note: on the 4-prompt C7 panel, release 4 accepts fewer drafts on the stock checkpoint because greedy text diverges under the numerics contract (291/482 vs 359/553); on the 32-prompt paired panel release 4 is level with release 3 (98.75 vs 98.53 tok/s, acceptance 78.54% vs 78.05%).*

How each row was measured:

- **KL to BF16:** full-vocabulary KL(BF16 || quantized) per token over the 16,384-token code and English prose documents. Release-4 whole-model KL deltas vs pre-VFAST are 0.0 on the M5 Max and within the cross-chip C4b bars on the M3 Ultra.
- **Verify vs decode KL (N1/N2):** 128 verify windows (584 positions) comparing unarmed batched verify against 1-row decode; all rows pass well below the cross-chip MLX floor (`4.505e-3` code / `1.336e-2` prose).
- **MTP acceptance:** accepted draft tokens per verify cycle, mean over the 32 fixed prompts (code, prose, chat), greedy, 128 tokens each, MTP adaptive.
- **Speed rows:** decode at 1, 2 and 4 concurrent streams (tokens/s) and prefill at 2,048 and 8,192 tokens, 16 timed rounds per MTP mode in fresh server processes alternating the two builds; the table gives medians. Time to first token is cold and warm. The Macs are a Mac Studio M3 Ultra (512 GB) and a Mac Studio M5 Max (128 GB, with the PLE table on SSD).

## Current limits

- **MTP adaptive vs off.** Adaptive speeds decode strongly at 1 and 2 streams but is slower at 4: 93.92 / 71.57 / 41.06 adaptive vs 75.76 / 60.31 / 43.32 off on the M3 Ultra, and 90.17 / 67.98 / 37.61 vs 71.51 / 57.11 / 40.92 on the M5 Max.
- **Against stock.** Decode with MTP off and prefill are close to stock (75.76 / 60.31 / 43.32 vs 76.14 / 60.68 / 43.77 on M3 Ultra; prefill 1,301 / 1,428 vs 1,313 / 1,440).
- **Memory.** On the 128 GB M5 Max, keep the PLE table on SSD (the install block shows the setting).

## Not measured

Public benchmark sets and task accuracy (MMLU, code benchmarks and the like), contexts longer than 8,192 tokens, more than 4 concurrent streams, and Macs other than the two above.

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
