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

# Qwen3.8-Flash-Next-oQ4e (refined, FP8 PLE)

This is the stock baseline for [bearzi/Qwen3.8-Flash-Next-oQ4eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL): [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 4 in its oQe mode, with oMLX's oQe rounding refinement (#5292). It was published with release 2 of the oQeL proof of concept.

- **Every quantized matrix is stock MLX affine**: the routed experts are 4-bit and everything else is 8-bit, the same bit plan as the oQ4eL build.
- **The per-layer embedding (PLE) table is FP8**, taken from Qwen's official FP8 table.
- **No plugin is needed.** It was verified to load and run on the oMLX fork's release-2 tag with no plugin installed. Upstream oMLX should load it too, since it is standard affine with an FP8 PLE table, but that was not tested.
- **Size: 134.1 GB.**

## Install and run

```sh
python3.13 -m venv omlx-env
. omlx-env/bin/activate
git clone --branch oqel-poc-r2 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
hf download bearzi/Qwen3.8-Flash-Next-oQ4e --local-dir ./models/Qwen3.8-Flash-Next-oQ4e
omlx serve --model-dir ./models
```

On a Mac with 128 GB of memory, keep the PLE table on SSD (`"qwen4_ple_ssd_offload": true` in `model_settings.json`, as on the [oQ4eL card](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL)).

## Measured

Against the oQ4eL build of the same source, calibration, bit plan and FP8 PLE table:

| | oQ4e (this) | oQ4eL |
|---|---|---|
| KL to BF16, code / prose | 0.01832 / 0.03358 | **0.01577 / 0.02905** |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval), M3 Ultra | 1.536 [1.408, 1.669] | 1.542 [1.394, 1.692] |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M3 Ultra | 76.14 / 60.72 / 43.67 | 75.66 / 60.17 / 42.79 |
| Decode tok/s, 1 / 2 / 4 streams, MTP off, M5 Max | 72.61 / 57.69 / 40.93 | 71.39 / 56.26 / 39.74 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M3 Ultra | 98.14 / 56.21 / 39.91 | 98.67 / 54.35 / 38.98 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive, M5 Max | 87.71 / 52.00 / 35.05 | 88.83 / 51.74 / 34.70 |
| Prefill tok/s 2k / 8k, M3 Ultra | 1,270 / 1,397 | 1,265 / 1,392 |
| Prefill tok/s 2k / 8k, M5 Max | 2,135 / 2,729 | 2,121 / 2,704 |

The methods are the same as on the oQ4eL card: two 16,384-token documents for KL, 32 fixed prompts for MTP acceptance, and 16 timed rounds per mode on each Mac in fresh server processes, reporting medians.

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
