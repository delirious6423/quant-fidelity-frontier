---
library_name: gguf
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - speculative-decoding
  - dflash
---

# Qwen3.8-27B DFlash2 drafter (GGUF)

The DFlash2 draft model used by the qualified **Qwen 3.8 27B** profile on
[Lucebox](https://www.lucebox.com) appliances. Lucebox Manage downloads this file
and verifies it byte for byte before installing the profile.

| | |
|---|---|
| file | `Qwen3.8-27B-DFlash2-Q8_0.gguf` |
| size | 2,045,471,776 bytes |
| sha256 | `bb727abc583498aa4deea8b3cd0c34c2d96553954cbff25b5f7bdd469f0f1306` |

Pair it with the target model
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
at `Qwen3.8-27B-UD-IQ4_XS.gguf`.

The DFlash2 block-diffusion drafter design comes from
[z-lab/dflash](https://github.com/z-lab/dflash). This is a separate build from the
public [`z-lab/Qwen3.8-27B-DFlash2-GGUF`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)
weights and the two files are not interchangeable for the byte-exact check above.
