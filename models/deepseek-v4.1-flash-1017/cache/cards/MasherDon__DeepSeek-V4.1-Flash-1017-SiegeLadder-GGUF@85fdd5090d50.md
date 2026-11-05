---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- gguf
- quantized
- gptq
- mixture-of-experts
- deepseek
- llama.cpp
---

# DeepSeek V4.1 Flash 1017 — Siege-Ladder `og` (GGUF, ~2-bit experts)

These are the weights we run DeepSeek V4.1 Flash 1017 with on one home PC. The model has 552B parameters in its MoE plus Engram. It runs on the [Siege-Ladder](https://github.com/MasherDon/Siege-Ladder) fork of llama.cpp, which streams experts through VRAM → RAM → NVMe. The code, patches and measurements are in that repository. This project is not affiliated with DeepSeek.

**This is not the final version.** Work on quantization quality continues, and these files will be replaced by better builds.

## What is inside

| Part | Format | Size |
|---|---|---|
| Routed experts, layers 4–35 | our GPTQ: gate/up IQ2_XXS, down IQ3_XXS | 74.74 + 55.49 GB in total |
| Routed experts, layers 0–3 and 36–39 | our GPTQ: IQ2_S / IQ3_S (more bits where errors cost more) | 23.22 + 15.57 GB |
| Dense and shared weights | Q5_K, plus small BF16 / F32 / Q8_0 tensors | 5.33 + ~0.8 GB |
| Engram | official FP8 (E4M3) tables with E8M0 scales, unchanged | 196.61 + 6.14 GB |
| `g44/1017-lens-d2.gguf` | learned correction of the expert-prediction lens | 149 MB |

- The total is 11 GGUF parts, 377.9 GB. The word `MXFP4` in the file names comes from the first assembly; it does not describe the tensors.
- Parts 1–4 (metadata, Engram, dense) are shared with our lighter `qd` build. Parts 5–11 hold the `og` experts.

## What was done for quality

- The experts are quantized with **GPTQ** on our own calibration texts, not with round-to-nearest. Every routed-expert layer is covered.
- More bits go to the most sensitive parts: the down projections, and the first and last four layers.
- Engram is kept at the official FP8 precision.
- We measure quality as the **distance to the full model**: the extra negative log-likelihood, in nat/token, of the full model's own (API) continuations, on three prompt sets (general, Russian, router). Smaller is better.
  - `og` exact: 0.090 / 0.051 / 0.069.
  - With the engine's default speed rules: 0.091 / 0.054 / 0.072.
  - HumanEval: 160 / 164.
  - All of these were measured on 27–28 October 2026.

## How to run

1. Build the fork from the patches in the [GitHub repository](https://github.com/MasherDon/Siege-Ladder). It is tested on Windows with CUDA 13 on an RTX 50 card (sm_120).
2. Pass `-m .../DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf`. The other parts load by themselves.
3. Set up the lens (G44 / G40, on by default). Either put `1017-lens-d2.gguf` into a folder named `g44` next to the model's folder, or point `LLAMA_MOE_STREAM_G44` at the file. Without the lens the engine still runs, but decodes slower.

Our PC: RTX 5070 Ti 16 GB on PCIe 4.0 x16, Ryzen 5 7600, 64 GB DDR5, one PCIe 5.0 NVMe (plus an optional expert mirror on a second NVMe).

| | tok/s |
|---|---|
| decode, short context | ~20 (21.30 teacher-forced with G44 + G40, 4 November) |
| reading a 246K-token document (256K context) | 930 (5 November) |
| reading a 479K-token document (512K context) | 867–900 (5 November) |
| reading a 947K-token document (1M context) | 778 (5 November) |

These are single runs, not averages.

## Hardware notes

- **PCIe 5.0.** With a PCIe 5.0 x16 card *and* slot, both prompt reading and decode get faster with no code change. Copying experts over the bus is a large part of every decode token.
  - Estimate: in an older profile, copies alone took 27.1 ms of a 54 ms token. Halving them gives about 24.7 instead of 18.5 tok/s.
- **A second NVMe** (expert mirror) helps prompt reading a little: +14–22% on cold prompts in an older test. It is optional.
- **Training.** Speed will also improve separately through training (expert prediction), independently of these weights.

## Which build

`og` (this repository) is the default and the closest to the full model. The lighter `qd` build is faster with the engine's `fast` preset, but loses more quality; ask if you need it.

## License

MIT, like the base model; see `LICENSE` (Copyright (c) 2023 DeepSeek).
