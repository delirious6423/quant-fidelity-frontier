---
license: mit
library_name: mlx
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash
- rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX
tags:
- mlx
- deepseek
- speculative-decoding
- multi-token-prediction
- apple-silicon
- rapid-mlx
pipeline_tag: text-generation
---

## Run the paired model on your Mac

This repository is a sidecar, not a standalone model. The command serves its paired target and uses this sidecar.

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
rapid-mlx serve deepseek-v41-flash-reap-2bit
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

Use it from OpenCode / Claude Code / Codex: `rapid-mlx agents <name> --setup` · [Agent setup](https://rapidmlx.com/docs/agents)

[Download the Desktop app](https://rapidmlx.com/download) · 256 GB unified memory required for the paired 212.9 GB target plus this 4.6 GB sidecar. This repo alone cannot answer prompts.

# DeepSeek V4.1 Flash DSpark 4d2e MLX


Experimental three-stage DSpark speculative-decoding sidecar for the Rapid-MLX
DeepSeek V4.1 Flash REAP 2-bit target.

## This is a sidecar, not a standalone model

This repository contains only the 4.62 GB DSpark head. It cannot generate text
by itself and does not contain the approximately 213 GB target model.

Required target:

- Repository: `rapid-mlx/DeepSeek-V4.1-Flash-REAP-2bit-MLX`
- Qualified target revision: `a25fec277b9e7cedc0e9f3f15da874a5cf9d491b`
- Runtime: Rapid-MLX's owned DeepSeek V4.1 DSpark implementation
- Recommended verification window: **K=4**
- Minimum supported machine class: **256 GiB unified memory**

Rapid-MLX downloads and verifies this sidecar separately when its experimental
DSpark path is enabled. Users who do not enable DSpark do not need these files.

## Precision layout

`4d2e` means:

- Attention, shared-expert, main-projection, Markov, and confidence paths:
  affine 4-bit, group size 64
- Bandwidth-dominant routed experts: affine 2-bit, group size 64
- Target-shared embedding and LM head: provided by the 2-bit target at runtime

The mixed layout is 4,617,792,648 bytes, only 3.5% larger than the fully 2-bit
DSpark head. A uniform 4-bit head was not viable inside the qualified 256 GiB
memory boundary.

## Measured performance

Environment:

- Apple M3 Ultra, 256 GiB unified memory
- Target: REAP12.5 native affine 2-bit checkpoint, approximately 212.93 GB
- Rapid-MLX packed DSpark verification, K=4
- Four prompt domains: code, arithmetic reasoning, JSON-only structured output,
  and Chinese
- Up to 128 output tokens, two consecutive repeats per prompt

| Path | Weighted decode throughput | Relative to AR |
| --- | ---: | ---: |
| Target autoregressive | 9.58 tok/s | 1.00x |
| Mixed DSpark, K=4 | **19.39 tok/s** | **2.02x** |
| Mixed DSpark, K=5 | 18.49 tok/s | 1.93x |

The K=4 result is approximately 20 tok/s for this measured configuration; it
is not a guarantee for every prompt, context length, thermal state, or machine.
Per-domain K=4 results ranged from 11.87 to 25.93 tok/s. Mean accepted draft
length was 1.85 extra tokens per verification block. Peak MLX memory was
218.232 GB.

K=5 accepted more draft tokens but was slower end to end, so K=4 is the
qualified fixed-window setting.

## Correctness and experimental status

All four K=4 outputs were token-for-token stable across two repetitions. The
verifier is target-authoritative: draft tokens are emitted only after target
verification, and corrections come from target logits.

Fixed-batch speculative output is not promised to be bitwise identical to a
sequential autoregressive run. Low-margin target decisions can differ with the
verification batch shape. The current 2-bit target also showed weak or
repetitive instruction-following on several English probes. Treat this sidecar
as experimental while broader comparative task-quality qualification is in
progress.

## Artifact integrity

The manifest records the immutable source revisions and shard hashes. Rapid-MLX
pins the repository revision and independently validates every required file's
size and SHA-256 before loading it.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `dspark-mixed-stage-0.safetensors` | 1,556,346,856 | `381a7fbc8758cd86baab55e8f1ae3020e49e2977f067d4269416889c1aee8118` |
| `dspark-mixed-stage-1.safetensors` | 1,512,099,424 | `2ca7dc77528ebe3220e6e1633733925ca3420e346b4475163e1639541333876e` |
| `dspark-mixed-stage-2.safetensors` | 1,549,346,368 | `1ed4663f0487e13372e753b2a5e8b760ae8fd43ba661fae891dfdff62384e64b` |
| `config.json` | 138,476 | `3518b608fd8b90bc518c7d4533c751367fe3c2416aa3a4f7fde40bc6e498f012` |
| `model.safetensors.index.json` | 265,016 | `70cbf70324c1b8d7d7c3d5f7f522f38d0be6cba673e90430a0f9a1d8b1ab8cff` |

## Sources and license

The dense DSpark paths were converted from the official DeepSeek V4.1 Flash
release at revision `dba1be0a40aa45a94ad051997016db3960a90277`. Routed
expert values come from the pinned affine 2-bit source at revision
`802f1a00982705d81b79ad1c83aa0ccc0b863ebc`.

This repository is distributed under the MIT license. See `LICENSE` and
`NOTICE`.
