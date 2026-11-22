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
> **Experimental.** This checkpoint needs a fork of oMLX that is not upstream yet ([yohann-bearzi/omlx](https://github.com/yohann-bearzi/omlx/tree/oqel-poc-r2), tag `oqel-poc-r2`) and the [omlx-oqel](https://github.com/yohann-bearzi/omlx-oqel) plugin 0.1.0. Stock oMLX, mlx-lm and mlx-vlm cannot load it. It is a proof of concept of a quant-format plugin interface for oMLX. The oQeL layout (revision 1) and the plugin's kernels may change, and a later plugin may refuse this checkpoint, which would then be quantized again. Do not rely on it for anything you cannot redo.

This is **release 2**; the 3-bit level is new in it.

## What it is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 3 in its oQe mode, with omlx-oqel 0.1.0:

- **The routed experts are 3-bit oQeL**: 147 matrices (gate, up and down of the 48 MoE layers and of the MTP layer). Each is kept on an 8-level table fitted to this model's experts, in groups of 16 weights with an E2M5 scale byte each and one bfloat16 scale per row. The `group_size: 64` that `config.json` lists for these matrices is oQ's nominal recipe; the oQeL layout does not read it.
- **Everything else is the same as the refined stock [oQ3e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3e) build**: MLX affine 8-bit, with oMLX's oQe rounding refinement (#5292).
- **The PLE table is FP8**, taken from Qwen's official FP8 table.
- **Size: about 119 GB.**

## Install

You need an Apple Silicon Mac with Python 3.11 to 3.13, git, about 125 GB of free disk and a network connection:

```sh
python3.13 -m venv oqel-env
. oqel-env/bin/activate
git clone --branch oqel-poc-r2 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
git clone --branch v0.1.0 https://github.com/yohann-bearzi/omlx-oqel omlx-oqel
python -m pip install ./omlx-oqel
hf download bearzi/Qwen3.8-Flash-Next-oQ3eL --local-dir ./models/Qwen3.8-Flash-Next-oQ3eL
```

**On a Mac with 128 GB of memory, keep the PLE table on SSD.** Put this `model_settings.json` in the server's base path, or set the same options in the dashboard:

```json
{"version": 2, "models": {"Qwen3.8-Flash-Next-oQ3eL": {"mtp_enabled": true, "qwen4_ple_ssd_offload": true, "a8_prefill_enabled": false, "force_sampling": false}}}
```

The 3-bit build's memory on a 128 GB Mac was not measured. The 4-bit build peaked at 75 to 77 GB with this setting, and this one is smaller.

## Run

As a server (OpenAI-compatible, port 8000):

```sh
omlx serve --model-dir ./models
```

From Python (the model is a vision-language model, so it loads through mlx-vlm):

```python
from pathlib import Path

from mlx_vlm.utils import load
from omlx.model_settings import ModelSettings
from omlx.utils.model_loading import maybe_apply_pre_load_patches

path = Path("./models/Qwen3.8-Flash-Next-oQ3eL")
maybe_apply_pre_load_patches(
    str(path), ModelSettings(qwen4_ple_ssd_offload=True), for_vlm=True
)
model, processor = load(str(path), lazy=True)
```

Without the plugin, the load stops with `no quant format is registered for mode 'oqel'`.

## Measured

Against the refined stock **oQ3e** written from the same source, calibration, bit plan and FP8 PLE table (only the routed experts' format differs). Speed was measured on the M3 Ultra only:

| | oQ3eL | oQ3e |
|---|---|---|
| KL to BF16, code / prose | **0.0369 / 0.0651** | 0.0438 / 0.0793 |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval) | 1.629 [1.480, 1.778] | 1.536 [1.395, 1.689] |
| Decode tok/s, 1 / 2 / 4 streams, MTP off | 77.50 / 63.04 / 45.79 | 77.60 / 63.06 / 45.85 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive | 93.52 / 56.39 / 41.35 | 95.98 / 57.08 / 41.29 |
| Prefill tok/s 2k / 8k | 1,248 / 1,375 | 1,262 / 1,387 |
| Time to first token, cold / warm | 1.94 / 1.65 s | 1.92 / 1.63 s |

The methods are the same as on the [oQ4eL card](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL): two 16,384-token documents for KL, 32 fixed prompts for MTP acceptance, and 16 timed rounds per mode in fresh server processes on a Mac Studio M3 Ultra, reporting medians.

## Current limits

- **Decode with MTP off is at parity with stock oQ3e** (within 0.14%).
- **Adaptive MTP at one stream reads 2.6% slower.** On the speed benchmark's single prompt, oQ3eL's drafts happened to be accepted less often (1.37 against 1.55 per verify cycle). Over the 32 prompts above, it accepts more drafts than stock.
- **Prefill is about 1% slower.**

## Not measured

Public benchmark sets and task accuracy (MMLU, code benchmarks and the like), contexts longer than 8,192 tokens, more than 4 concurrent streams, the M5 Max, and other Macs.

## Other levels

[oQ4eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL) (measured on both Macs) and [oQ2eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL), with their refined stock baselines [oQ4e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4e) and [oQ2e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2e).

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). omlx-oqel is Apache-2.0.
