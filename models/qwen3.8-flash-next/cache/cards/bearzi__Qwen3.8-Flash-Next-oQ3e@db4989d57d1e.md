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

# Qwen3.8-Flash-Next-oQ3e (refined, FP8 PLE)

This is the stock baseline for [bearzi/Qwen3.8-Flash-Next-oQ3eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3eL): [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 3 in its oQe mode, with oMLX's oQe rounding refinement (#5292). It was published with release 2 of the oQeL proof of concept.

- **Every quantized matrix is stock MLX affine**: the routed experts are 3-bit and everything else is 8-bit, the same bit plan as the oQ3eL build.
- **The per-layer embedding (PLE) table is FP8**, taken from Qwen's official FP8 table.
- **No plugin is needed.** It was verified to load and run on the oMLX fork's release-2 tag with no plugin installed. Upstream oMLX should load it too, since it is standard affine with an FP8 PLE table, but that was not tested.
- **Size: about 119 GB.**

## Install and run

```sh
python3.13 -m venv omlx-env
. omlx-env/bin/activate
git clone --branch oqel-poc-r2 https://github.com/yohann-bearzi/omlx omlx
python -m pip install -e ./omlx
hf download bearzi/Qwen3.8-Flash-Next-oQ3e --local-dir ./models/Qwen3.8-Flash-Next-oQ3e
omlx serve --model-dir ./models
```

On a Mac with 128 GB of memory, keep the PLE table on SSD (`"qwen4_ple_ssd_offload": true` in `model_settings.json`, as on the [oQ3eL card](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ3eL)).

## Measured

Against the oQ3eL build of the same source, calibration, bit plan and FP8 PLE table. Speed was measured on the M3 Ultra only:

| | oQ3e (this) | oQ3eL |
|---|---|---|
| KL to BF16, code / prose | 0.0438 / 0.0793 | **0.0369 / 0.0651** |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval) | 1.536 [1.395, 1.689] | 1.629 [1.480, 1.778] |
| Decode tok/s, 1 / 2 / 4 streams, MTP off | 77.60 / 63.06 / 45.85 | 77.50 / 63.04 / 45.79 |
| Decode tok/s, 1 / 2 / 4 streams, MTP adaptive | 95.98 / 57.08 / 41.29 | 93.52 / 56.39 / 41.35 |
| Prefill tok/s 2k / 8k | 1,262 / 1,387 | 1,248 / 1,375 |

The methods are the same as on the [oQ4eL card](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ4eL).

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
