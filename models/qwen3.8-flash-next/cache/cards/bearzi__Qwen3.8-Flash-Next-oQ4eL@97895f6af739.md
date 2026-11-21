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
| MTP accepted drafts per verify cycle (32 prompts), M3 Ultra / M5 Max | 1.618 / 1.575 | 1.555 / 1.572 |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M3 Ultra | 75.69 / 59.81 / 42.79 | 76.26 / 60.65 / 43.41 |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M5 Max | 71.24 / 56.08 / 39.73 | 72.66 / 57.39 / 40.60 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M3 Ultra | 94.27 / 54.46 / 39.67 | 99.39 / 56.10 / 40.23 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M5 Max | 91.09 / 52.05 / 37.42 | 90.73 / 53.50 / 37.43 |
| Prefill tok/s 2k / 8k, M3 Ultra | 1,261 / 1,387 | 1,265 / 1,392 |
| Prefill tok/s 2k / 8k, M5 Max | 2,104 / 2,688 | 2,116 / 2,713 |

Methods:

- KL: full-vocabulary (248,320 tokens) KL(BF16 ‖ quantized) per token, averaged over two 16,384-token documents (code; English prose). The BF16 log-probabilities were computed once on an M3 Ultra; both quantized builds ran on an M5 Max. Top-1 agreement is the share of tokens where both argmaxes match; perplexity is on the same two documents.
- MTP acceptance: accepted draft tokens per verify cycle, greedy, 128 tokens on each of 32 fixed prompts (code, prose, chat), MTP adaptive; 95% intervals overlap on both Macs (M3 Ultra 1.618 [1.463, 1.774] vs 1.555 [1.420, 1.700]).
- Decode and prefill: the oMLX server on the release code (fork tag `oqel-poc-r1`, omlx-oqel 0.1.0), 16 timed rounds per MTP mode on each Mac, each round alternating the two builds in fresh server processes; medians. 1, 2 and 4 concurrent streams; 2,048- and 8,192-token prompts. Mac Studio M3 Ultra (512 GB) and Mac Studio M5 Max (128 GB). 
- Greedy output with MTP on equals output with MTP off, byte for byte (three chats and a completion, on the published branch).

## Current limits

- **Decode is within 1-2.3% of stock oQ4e with MTP off** on both Macs, and at parity with adaptive MTP on the M5 Max. Getting there took admitting plugin formats to oMLX's fused routed-decode and verify-window paths (they had refused faces exact for every float type); oQeL's level-table decode still costs a little more per row than the affine grid.
- **One row reads 5.15% slower: adaptive MTP at one stream on the M3 Ultra.** On that benchmark's single prompt oQ4eL's drafts happen to be accepted less often, so it needs 63 verify cycles for the same 128 tokens where stock needs 60 (5.0% more); per verify cycle the two are at parity. Over 32 prompts acceptance is equal (above), and on four other prompts the same build measured 0.36% behind stock.
- **The stock comparison predates oMLX's oQe rounding refinement** (#5292, which improves stock oQe builds at no size or speed cost); the refinement does not yet apply to plugin formats. A later release compares against a refined stock build.
- **The PLE table is in BF16** (about 95 of the 178 GB); a smaller-PLE build will follow.

## Not measured

Public benchmark sets and task accuracy (MMLU, code benchmarks and the like); contexts longer than 8,192 tokens; more than 4 concurrent streams; Macs other than the two above; the 3- and 2-bit levels (coming).

## Reproduce

```sh
omlx-oqel /path/to/Qwen3.8-Flash-Next Qwen3.8-Flash-Next-oQ4eL --bits 4 --fit --preserve-mtp --samples 128 --seq-length 512
```

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). omlx-oqel is Apache-2.0.
