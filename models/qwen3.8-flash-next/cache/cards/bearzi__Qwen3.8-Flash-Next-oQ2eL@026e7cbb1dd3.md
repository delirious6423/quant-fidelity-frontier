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
> **2-bit is a large quality step down.** KL to BF16 is about 7 times the 4-bit build's (0.1067 against 0.01577 on code). Use it only where memory rules out 3 or 4 bits.

> [!WARNING]
> **Experimental.** This checkpoint needs a fork of oMLX that is not upstream yet ([yohann-bearzi/omlx](https://github.com/yohann-bearzi/omlx/tree/oqel-poc-r2), tag `oqel-poc-r2`) and the [omlx-oqel](https://github.com/yohann-bearzi/omlx-oqel) plugin 0.1.0. Stock oMLX, mlx-lm and mlx-vlm cannot load it. It is a proof of concept of a quant-format plugin interface for oMLX. The oQeL layout (revision 1) and the plugin's kernels may change, and a later plugin may refuse this checkpoint, which would then be quantized again. Do not rely on it for anything you cannot redo.

This is **release 2**; the 2-bit level is new in it.

## What it is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 2 in its oQe mode, with omlx-oqel 0.1.0:

- **The routed experts are 2-bit oQeL**: 147 matrices (gate, up and down of the 48 MoE layers and of the MTP layer). Each is kept on a 4-level table fitted to this model's experts, in groups of 16 weights with an E2M5 scale byte each and one bfloat16 scale per row. The `group_size: 64` that `config.json` lists for these matrices is oQ's nominal recipe; the oQeL layout does not read it.
- **Everything else is the same as the refined stock [oQ2e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2e) build**: MLX affine 8-bit, with oMLX's oQe rounding refinement (#5292).
- **The PLE table is FP8**, taken from Qwen's official FP8 table.
- **Size: about 104 GB.**

## Install

You need an Apple Silicon Mac with Python 3.11 to 3.13, git, about 110 GB of free disk and a network connection:

```sh
python3.13 -m venv oqel-env
. oqel-env/bin/activate
git clone --branch oqel-poc-r2 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
git clone --branch v0.1.0 https://github.com/yohann-bearzi/omlx-oqel omlx-oqel
python -m pip install ./omlx-oqel
hf download bearzi/Qwen3.8-Flash-Next-oQ2eL --local-dir ./models/Qwen3.8-Flash-Next-oQ2eL
```

**On a Mac with 128 GB of memory, keep the PLE table on SSD.** Put this `model_settings.json` in the server's base path, or set the same options in the dashboard:

```json
{"version": 2, "models": {"Qwen3.8-Flash-Next-oQ2eL": {"mtp_enabled": true, "qwen4_ple_ssd_offload": true, "a8_prefill_enabled": false, "force_sampling": false}}}
```

The 2-bit build's memory on a 128 GB Mac was not measured. The 4-bit build peaked at 75 to 77 GB with this setting, and this one is smaller.

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

path = Path("./models/Qwen3.8-Flash-Next-oQ2eL")
maybe_apply_pre_load_patches(
    str(path), ModelSettings(qwen4_ple_ssd_offload=True), for_vlm=True
)
model, processor = load(str(path), lazy=True)
```

Without the plugin, the load stops with `no quant format is registered for mode 'oqel'`.

## Measured

Against the refined stock **oQ2e** written from the same source, calibration, bit plan and FP8 PLE table (only the routed experts' format differs). Speed was measured on the M3 Ultra only:

| | oQ2eL | oQ2e |
|---|---|---|
| KL to BF16, code / prose | **0.1067 / 0.2008** | 0.1369 / 0.2436 |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval) | 1.497 [1.354, 1.643] | 1.419 [1.292, 1.559] |
| Decode tok/s, 1 / 2 / 4 streams, MTP off | 79.70 / 65.94 / 48.86 | 79.39 / 65.53 / 47.74 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive | 107.54 / 57.67 / 43.38 | 96.93 / 61.03 / 42.44 |
| Prefill tok/s 2k / 8k | 1,289 / 1,415 | 1,268 / 1,394 |
| Time to first token, cold / warm | 1.88 / 1.60 s | 1.91 / 1.62 s |

The methods are the same as on the [oQ4eL card](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL): two 16,384-token documents for KL, 32 fixed prompts for MTP acceptance, and 16 timed rounds per mode in fresh server processes on a Mac Studio M3 Ultra, reporting medians.

## Current limits

- **oQ2eL is faster than stock oQ2e on almost every speed row**: decode with MTP off by 0.4 to 2.3%, prefill by about 1.6%, and adaptive MTP at one stream by 11%.
- **Adaptive MTP at 2 streams reads 5.5% slower** (57.67 against 61.03 tok/s). This is the only row where oQ2eL loses. The swing from 11% faster at one stream to 5.5% slower at two follows acceptance on the speed benchmark's single prompt, where oQ2eL's drafts were accepted less often (1.19 against 1.28 per verify cycle). Over the 32 prompts above, it accepts more drafts than stock. A full repeat run reproduced the row: 57.71 against 61.00 tok/s. The benchmark's prompt is decoded greedily, so its acceptance, and with it this gap, is the same on every run.
- **The 2-bit quality cost is large** (see the caution above).

## Not measured

Public benchmark sets and task accuracy (MMLU, code benchmarks and the like), contexts longer than 8,192 tokens, more than 4 concurrent streams, the M5 Max, and other Macs.

## Other levels

[oQ4eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL) (measured on both Macs) and [oQ3eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3eL), with their refined stock baselines [oQ4e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4e) and [oQ3e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3e).

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). omlx-oqel is Apache-2.0.
