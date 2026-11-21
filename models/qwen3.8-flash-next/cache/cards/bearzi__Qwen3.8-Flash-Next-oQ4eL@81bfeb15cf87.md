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
> **Experimental.** This checkpoint needs a fork of oMLX that is not upstream yet ([yohann-bearzi/omlx](https://github.com/yohann-bearzi/omlx/tree/quant-plugins-poc), branch `quant-plugins-poc`) and the [omlx-oqel](https://github.com/yohann-bearzi/omlx-oqel) plugin. Stock oMLX, mlx-lm and mlx-vlm cannot load it. It is a proof of concept of a quant-format plugin interface for oMLX: the oQeL layout (revision 1) and the plugin's kernels may change, and a later plugin may refuse this checkpoint, which would then be quantized again. Do not rely on it for anything you cannot redo.

> [!NOTE]
> **Release in progress (2026-11-20).** The weights are final. The oMLX fork branch and the omlx-oqel plugin they need are being published in the next hours, together with a decode-speed fix; the speed figures below will be updated then.

## What it is

[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 4 in its oQe mode (each weight's error weighed by how much its input matters, measured on calibration text), with omlx-oqel 0.1.0:

- **The routed experts are 4-bit oQeL**: 147 matrices (gate, up and down of the 48 MoE layers and of the MTP layer). Each is kept on a 16-level table fitted to this model's experts, in groups of 16 weights with an E2M5 scale byte each and one bfloat16 scale per row. (The `group_size: 64` that `config.json` lists for these matrices is oQ's nominal recipe; the oQeL layout does not read it.)
- **Everything else is the same as the stock oQ4e build**: MLX affine 8-bit, in groups of 64 (598 tensor classes) or 128 (the 147 shared-expert projections).
- **Size: 178.3 GB (166.0 GiB).** The stock oQ4e written by the same fork is the same size: both keep the per-layer embedding (PLE) table in BF16, about 95 GB. The published [Jundot/Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) is 107.7 GB because it stores that table smaller; a smaller-PLE build of this checkpoint is planned.

## Install

An Apple Silicon Mac with Python 3.11 to 3.13, git, about 180 GB of free disk and a network connection:

```sh
python3.13 -m venv oqel-env
. oqel-env/bin/activate
git clone --branch quant-plugins-poc https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
git clone https://github.com/yohann-bearzi/omlx-oqel omlx-oqel
python -m pip install ./omlx-oqel
hf download bearzi/Qwen3.8-Flash-Next-oQ4eL --local-dir ./models/Qwen3.8-Flash-Next-oQ4eL
```

**On a Mac with 128 GB of memory, keep the PLE table on SSD** (otherwise its ~95 GB must be resident). Put this `model_settings.json` in the server's base path, or set the same options in the dashboard:

```json
{"version": 2, "models": {"Qwen3.8-Flash-Next-oQ4eL": {"mtp_enabled": true, "qwen4_ple_ssd_offload": true, "a8_prefill_enabled": false, "force_sampling": false}}}
```

With it, the model used 71 to 74 GB of memory on a 128 GB M5 Max. Without it (and, from Python, without `for_vlm=True` and `lazy=True` below), the PLE table is loaded into memory and a 128 GB Mac runs out of memory.

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

Without the plugin the load stops with `no quant format is registered for mode 'oqel'`.

## Measured

Against the stock **oQ4e** written by the same fork, from the same source, calibration and bit plan (only the routed experts' format differs):

| | oQ4eL | oQ4e |
|---|---|---|
| KL to BF16, code / prose | **0.01580 / 0.02866** | 0.02131 / 0.03737 |
| KL p99, code / prose | **0.185 / 0.226** | 0.274 / 0.292 |
| Top-1 agreement with BF16, code / prose | **96.06% / 92.28%** | 95.43% / 91.41% |
| Perplexity, code / prose (BF16: 2.1480 / 7.9083) | 2.1482 / 7.9028 | 2.1485 / 7.8900 |
| MTP accepted drafts per verify cycle, M3 Ultra / M5 Max | 1.45 / 1.30 | 1.50 / 1.50 |
| Decode tok/s, 1 stream, MTP off, M3 Ultra / M5 Max | 65.35 / 67.18 | 75.86 / 72.75 |
| Decode tok/s, 1 stream, MTP adaptive, M3 Ultra / M5 Max | 85.41 / 88.05 | 99.36 / 90.52 |
| Decode tok/s, 2 / 4 streams, MTP off, M3 Ultra | 53.01 / 39.64 | 60.53 / 43.45 |
| Decode tok/s, 2 / 4 streams, MTP off, M5 Max | 53.40 / 39.10 | 57.45 / 40.57 |
| Prefill tok/s 2k / 8k, M3 Ultra | 1,253 / 1,384 | 1,261 / 1,390 |
| Prefill tok/s 2k / 8k, M5 Max | 2,096 / 2,685 | 2,120 / 2,711 |

Methods:

- KL: full-vocabulary (248,320 tokens) KL(BF16 ‖ quantized) per token, averaged over two 16,384-token documents (code; English prose). The BF16 log-probabilities were computed once on an M3 Ultra; both quantized builds ran on an M5 Max. Top-1 agreement is the share of tokens where both argmaxes match; perplexity is on the same two documents.
- MTP acceptance: accepted draft tokens per verify cycle, from separate diagnostic server processes on four fixed prompts per build and mode.
- Decode and prefill: the oMLX server, 16 timed rounds per MTP mode on each Mac, each round alternating the two builds in fresh server processes; medians. 1, 2 and 4 concurrent streams; 2,048- and 8,192-token prompts. Mac Studio M3 Ultra (512 GB) and Mac Studio M5 Max (128 GB). Runtime: the fork at the commit whose serving code `quant-plugins-poc` ships unchanged, omlx-oqel 0.1.0's code.
- Greedy output with MTP on equals output with MTP off, byte for byte (three chats and a completion, on the published branch).

## Current limits

- **Decode is slower than stock oQ4e**: 7-14% at one stream with MTP off, up to 14% with adaptive MTP on the M3 Ultra, less on the M5 Max. A kernel investigation traced it to the fused routed-decode path: it takes a format's experts only when their face is exact for exactly one activation dtype, and oQeL's face is exact for every float type, so each MoE block keeps the composed path and runs extra kernel launches every token. A fix that admits plugin formats to the fused path brings one-token decode to 1.01 of stock on the M3 Ultra; multi-row verify windows are next. An update will follow.
- **MTP acceptance is lower** (1.30-1.45 against 1.50). Per operation, the oQeL experts are no less accurate than stock against a float32 reference; the drafts agree less often with the target's rounding. Disclosed, not fixed.
- **The stock comparison predates oMLX's oQe rounding refinement** (#5292, which improves stock oQe builds at no size or speed cost); the refinement does not yet apply to plugin formats. An update will compare against a refined stock build.
- **The PLE table is in BF16** (about 95 of the 178 GB); a smaller-PLE build will follow.

## Not measured

Public benchmark sets and task accuracy (MMLU, code benchmarks and the like); contexts longer than 8,192 tokens; more than 4 concurrent streams; Macs other than the two above; the 3- and 2-bit levels (coming).

## Reproduce

```sh
omlx-oqel /path/to/Qwen3.8-Flash-Next Qwen3.8-Flash-Next-oQ4eL --bits 4 --fit --preserve-mtp --samples 128 --seq-length 512
```

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). omlx-oqel is Apache-2.0.
