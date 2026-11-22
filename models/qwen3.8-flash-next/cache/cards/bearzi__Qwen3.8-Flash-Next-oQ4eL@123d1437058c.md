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
> **Experimental.** This checkpoint needs a fork of oMLX that is not upstream yet ([yohann-bearzi/omlx](https://github.com/yohann-bearzi/omlx/tree/oqel-poc-r2), tag `oqel-poc-r2`) and the [omlx-oqel](https://github.com/yohann-bearzi/omlx-oqel) plugin 0.1.0. Stock oMLX, mlx-lm and mlx-vlm cannot load it. It is a proof of concept of a quant-format plugin interface for oMLX. The oQeL layout (revision 1) and the plugin's kernels may change, and a later plugin may refuse this checkpoint, which would then be quantized again. Do not rely on it for anything you cannot redo.

This is **release 2**. It replaces release 1, whose per-layer embedding (PLE) table was BF16 (178 GB). Release 1's files are no longer here.

## What it is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 4 in its oQe mode, with omlx-oqel 0.1.0:

- **The routed experts are 4-bit oQeL**: 147 matrices (gate, up and down of the 48 MoE layers and of the MTP layer). Each is kept on a 16-level table fitted to this model's experts, in groups of 16 weights with an E2M5 scale byte each and one bfloat16 scale per row. The `group_size: 64` that `config.json` lists for these matrices is oQ's nominal recipe; the oQeL layout does not read it.
- **Everything else is the same as the refined stock [oQ4e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4e) build**: MLX affine 8-bit, with oMLX's oQe rounding refinement (#5292).
- **The PLE table is FP8**, taken from Qwen's official FP8 table.
- **Size: 134.3 GB.**

## Install

You need an Apple Silicon Mac with Python 3.11 to 3.13, git, about 140 GB of free disk and a network connection:

```sh
python3.13 -m venv oqel-env
. oqel-env/bin/activate
git clone --branch oqel-poc-r2 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
git clone --branch v0.1.0 https://github.com/yohann-bearzi/omlx-oqel omlx-oqel
python -m pip install ./omlx-oqel
hf download bearzi/Qwen3.8-Flash-Next-oQ4eL --local-dir ./models/Qwen3.8-Flash-Next-oQ4eL
```

**On a Mac with 128 GB of memory, keep the PLE table on SSD.** Put this `model_settings.json` in the server's base path, or set the same options in the dashboard:

```json
{"version": 2, "models": {"Qwen3.8-Flash-Next-oQ4eL": {"mtp_enabled": true, "qwen4_ple_ssd_offload": true, "a8_prefill_enabled": false, "force_sampling": false}}}
```

With it, the server process peaked at 75 to 77 GB on a 128 GB M5 Max.

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

path = Path("./models/Qwen3.8-Flash-Next-oQ4eL")
maybe_apply_pre_load_patches(
    str(path), ModelSettings(qwen4_ple_ssd_offload=True), for_vlm=True
)
model, processor = load(str(path), lazy=True)
```

Without the plugin, the load stops with `no quant format is registered for mode 'oqel'`.

## Measured

Against the refined stock **oQ4e** written from the same source, calibration, bit plan and FP8 PLE table (only the routed experts' format differs):

| | oQ4eL | oQ4e |
|---|---|---|
| KL to BF16, code / prose | **0.01576 / 0.02905** | 0.01832 / 0.03358 |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval), M3 Ultra | 1.542 [1.394, 1.692] | 1.536 [1.408, 1.669] |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M3 Ultra | 75.66 / 60.17 / 42.79 | 76.14 / 60.72 / 43.67 |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M5 Max | 71.39 / 56.26 / 39.74 | 72.61 / 57.69 / 40.93 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M3 Ultra | 98.67 / 54.35 / 38.98 | 98.14 / 56.21 / 39.91 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M5 Max | 88.83 / 51.74 / 34.70 | 87.71 / 52.00 / 35.05 |
| Prefill tok/s 2k / 8k, M3 Ultra | 1,265 / 1,392 | 1,270 / 1,397 |
| Prefill tok/s 2k / 8k, M5 Max | 2,121 / 2,704 | 2,135 / 2,729 |
| Time to first token, cold / warm, M3 Ultra | 1.92 / 1.62 s | 1.91 / 1.62 s |
| Time to first token, cold / warm, M5 Max | 1.21 / 0.97 s | 1.20 / 0.96 s |

How each row was measured:

- **KL:** full-vocabulary (248,320 tokens) KL(BF16 ‖ quantized) per token, averaged over two 16,384-token documents (code, and English prose). The BF16 reference is release 1's.
- **MTP acceptance:** the mean over 32 fixed prompts (code, prose, chat) of accepted draft tokens per verify cycle, greedy, 128 tokens each, MTP adaptive. This is the same statistic as release 1's 1.618, which was measured on the runtime from before the upstream merge; the intervals overlap. A BF16-PLE oQ4eL measured 1.504 on this runtime, so the FP8 table costs nothing measurable here.
- **Decode, prefill and time to first token:** the oMLX server on the release code (fork tag `oqel-poc-r2`, omlx-oqel 0.1.0). Each Mac ran 16 timed rounds per MTP mode, each round alternating the two builds in fresh server processes; the table gives medians. Decode used 1, 2 and 4 concurrent streams; prefill used 2,048- and 8,192-token prompts. The Macs are a Mac Studio M3 Ultra (512 GB) and a Mac Studio M5 Max (128 GB, with the PLE table on SSD).

## Current limits

- **Decode with MTP off is 0.6 to 2.9% slower than stock oQ4e**, more at 4 streams. oQeL's level-table decode costs a little more per row than the affine grid.
- **Adaptive MTP at 2 and 4 streams on the M3 Ultra reads 2.3 to 3.3% slower.** It is not an acceptance gap: in the same run oQ4eL's drafts were accepted more often (1.447 against 1.325 per verify cycle). Each verify cycle simply costs more.
- **Release 2's runtime decodes more slowly with adaptive MTP than release 1's did**, for both builds (for example about 35 tok/s at 4 streams on the M5 Max, against 37.4). This came with the merge of upstream oMLX; stock is affected the same way.

## Not measured

Public benchmark sets and task accuracy (MMLU, code benchmarks and the like), contexts longer than 8,192 tokens, more than 4 concurrent streams, and Macs other than the two above.

## Other levels

[oQ3eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3eL) and [oQ2eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL), with their refined stock baselines [oQ3e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3e) and [oQ2e](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2e).

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). omlx-oqel is Apache-2.0.
