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

# Qwen3.8-Flash-Next-oQ2e (refined, FP8 PLE)

> [!CAUTION]
> **2-bit is a large quality step down.** KL to BF16 is large: 0.1369 on code, against 0.01832 for the 4-bit stock build. Use it only where memory rules out 3 or 4 bits.

This is the stock baseline for [bearzi/Qwen3.8-Flash-Next-oQ2eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL): [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 2 in its oQe mode, with oMLX's oQe rounding refinement (#5292). It was published with release 2 of the oQeL proof of concept.

- **Every quantized matrix is stock MLX affine**: the routed experts are 2-bit and everything else is 8-bit, the same bit plan as the oQ2eL build.
- **The per-layer embedding (PLE) table is FP8**, taken from Qwen's official FP8 table.
- **No plugin is needed.** It was verified to load and run on the oMLX fork's release-2 tag with no plugin installed. Upstream oMLX should load it too, since it is standard affine with an FP8 PLE table, but that was not tested.
- **Size: about 104 GB.**

## Install and run

```sh
python3.13 -m venv omlx-env
. omlx-env/bin/activate
git clone --branch oqel-poc-r2 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
hf download bearzi/Qwen3.8-Flash-Next-oQ2e --local-dir ./models/Qwen3.8-Flash-Next-oQ2e
omlx serve --model-dir ./models
```

On a Mac with 128 GB of memory, keep the PLE table on SSD (`"qwen4_ple_ssd_offload": true` in `model_settings.json`, as on the [oQ2eL card](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL)).

## Measured

Against the oQ2eL build of the same source, calibration, bit plan and FP8 PLE table. Speed was measured on the M3 Ultra only:

| | oQ2e (this) | oQ2eL |
|---|---|---|
| KL to BF16, code / prose | 0.1369 / 0.2436 | **0.1067 / 0.2008** |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval) | 1.419 [1.292, 1.559] | 1.497 [1.354, 1.643] |
| Decode tok/s, 1 / 2 / 4 streams, MTP off | 79.39 / 65.53 / 47.74 | 79.70 / 65.94 / 48.86 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive | 96.93 / 61.03 / 42.44 | 107.54 / 57.67 / 43.38 |
| Prefill tok/s 2k / 8k | 1,268 / 1,394 | 1,289 / 1,415 |

The methods are the same as on the [oQ4eL card](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL).

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
