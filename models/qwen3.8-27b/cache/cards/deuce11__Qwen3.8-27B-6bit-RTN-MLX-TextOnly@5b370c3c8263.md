---
language: en
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- text-only
- rtn
- 6-bit
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B 6-bit RTN, text only

MLX checkpoint. Affine 6-bit, group size 64, 21.86 GB, 21 shards. The vision tower is not in these weights. `config.json` still has `language_model_only: false`. Serve it with Splash `--language-only`.

The packed integers are a round-to-nearest snap. `mlx-community/Qwen3.8-27B-8bit` was dequantized, then quantized with mlx-lm 0.32.2 at 6 bits, group 64, affine. No scales were trained. A DWQ pass on that student, the same tulu-3 recipe as the 3-bit run, raised the validation loss, so those trained weights are not this repo.

Validation loss against the 8-bit teacher on the 32 held-out rows (`allenai/tulu-3-sft-mixture`, max sequence 512):

| weights | loss |
|---|---:|
| this file, round-to-nearest | 0.005 |
| DWQ, learning rate 1e-6, step 511 | 0.071 |
| DWQ, learning rate 1e-5, step 511 | 1.736 |

mlx-lm warned that 0.071 was worse than the initial 0.005.

Splash 1.3.1 on an M5 Max with 32 GPU cores and 36 GB, int8 KV, thinking off, DFlash2 on, 128K window, cold cache, 128 new tokens:

| | 4k prose | 4k code | 64k prose | 64k code |
|---|---:|---:|---:|---:|
| Prefill tok/s | 911 | 989 | 740 | 739 |
| Decode tok/s | 34.2 | 99.2 | 34.7 | 83.4 |
| TTFT | 4.5 s | 4.1 s | 88.6 s | 88.7 s |

Draft acceptance was 22% and 87% at 4k, and 27% and 82% at 64k. Splash memory was 25.3 GB and 25.6 GB at 4k, and 27.5 GB at 64k. The peak was 28.1 GB. Swap did not grow.

The base model is Apache 2.0.
