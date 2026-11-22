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
> **2-bit is a large quality step down.** KL to BF16 is about 7 times the 4-bit build's. Use it only where memory rules out 3 or 4 bits.


> [!NOTE]
> **Release 2 in progress.** The files are complete and hash-verified. Install and speed sections follow with the release.

The stock baseline for [bearzi/Qwen3.8-Flash-Next-oQ2eL](https://huggingface.co/bearzi/Qwen3.8-Flash-Next-oQ2eL): [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) quantized with oQ, oMLX's quantizer, at level 2 in its oQe mode, with oMLX's oQe rounding refinement (#5292).

- **Every quantized matrix is stock MLX affine**: 2-bit routed experts and 8-bit elsewhere, the same bit plan as the oQ2eL build.
- **The per-layer embedding (PLE) table is FP8**, grafted from Qwen's official FP8 table.
- **No plugin is needed.** It loaded and ran a forward on the fork's release-2 runtime with no plugin installed. Upstream oMLX should load it, since it is standard affine with an FP8 PLE table, but that was not tested.
- **Size: about 104 GB.**

## Measured so far

M3 Ultra, release-2 runtime, against the same BF16 reference as the oQ2eL card:

| | oQ2e (this) | oQ2eL |
|---|---|---|
| KL to BF16, code / prose | 0.1369 / 0.2436 | **0.1067 / 0.2008** |
| MTP accepted drafts per verify cycle (32 prompts; 95% interval) | 1.419 [1.292, 1.559] | 1.497 [1.354, 1.643] |

The methods are the same as on the oQ2eL card. Decode and prefill speed on both Macs: in progress.

## Licence

The weights are Qwen/Qwen3.8-Flash-Next's, under its licence: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
