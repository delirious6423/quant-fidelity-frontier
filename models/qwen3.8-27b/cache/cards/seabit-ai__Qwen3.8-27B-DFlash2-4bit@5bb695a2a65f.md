---
license: apache-2.0
base_model:
  - incoai/Qwen3.8-27B-DFlash2
  - Qwen/Qwen3.8-27B
tags:
  - mlx
  - speculative-decoding
  - dflash2
  - draft-model
library_name: mlx-vlm
---

# Qwen3.8-27B DFlash 2 draft, 4-bit MLX

The DFlash 2 drafter for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), from
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (z-lab / Inco AI),
quantized to 4 bits for MLX: **1.1 GB instead of 4.0 GB** (bf16). It is not a language model on its own:
it drafts blocks of tokens that the target model verifies.

## Provenance

- Source: `incoai/Qwen3.8-27B-DFlash2`, revision `015e795645c74b1a0eeef3b570031fb62e769bc5` (bf16, 2B parameters).
- Tool: `python -m mlx_vlm.convert --hf-path incoai/Qwen3.8-27B-DFlash2 -q --q-bits 4` (mlx-vlm 0.6.16):
  affine 4-bit, group size 64, 4.5 bits per weight. `config.json` is the source's plus the quantization entry.
- Checksums: `SHA256SUMS`.

## Measured (Apple M3 Ultra, 96 GB; target `lmstudio-community/Qwen3.8-27B-MLX-4bit`)

Against the bf16 original on the same prompts, greedy, 400 tokens: the same tokens accepted per round
(4.30 against 4.26 on code) and the same output, token for token; decoding speed unchanged on this Mac
(the draft step's cost there is compute, not reading weights). The gain is size: 3 GB less to download
and to keep in memory. Details: github.com/seabit-ai/lmk, `research/2026-09-23-speculative-decoding/`
(exp07, exp12).

## Use

With [lmk](https://github.com/seabit-ai/lmk): `model.name: qwen3.8-27b-4bit`, `speculative_decoding: true`,
`draft: dflash2` in `~/.lmk/config.yaml`; `lmk up` fetches it.

With mlx-vlm directly (0.6.16 or later):

```python
from mlx_vlm.speculative.drafters import load_drafter
draft, kind = load_drafter("seabit-ai/Qwen3.8-27B-DFlash2-4bit")   # kind == "dflash"
```

## License

Apache-2.0, as the source. The weights are z-lab's / Inco AI's work; this repository only quantizes them.
