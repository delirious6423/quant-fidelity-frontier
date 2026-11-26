---
license: mit
library_name: mlx
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- mlx
- oq
- 4-bit
- abliterated
- uncensored
- mtp
---

# DeepSeek-V4.1-Flash-1017-oQ4e-mtp-uncensored

An uncensored DeepSeek-V4.1-Flash-1017 for Apple Silicon at 4-bit quality: MLX format, oQ4e mixed precision (4.529 bits/weight), with the multi-token prediction (MTP) head. It is sized for a Mac Studio with 512 GB of memory.

## How it was made

This build puts two published works together. Our part is the assembly and the checks.

- **Base:** [mlx-community/DeepSeek-V4.1-Flash-1017-oQ4e-mtp](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ4e-mtp), the 4-bit oQ4e build made with oQ in [oMLX](https://github.com/jundot/omlx) 0.7.1. Every file was checked against its published hash.
- **Uncensoring:** the directional ablation by pqhaz, [pqhaz/DeepSeek-V4.1-Flash-1017-heretic](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic), which follows the [Heretic](https://github.com/p-e-w/heretic) method. It edits 38 tensors: `attn.wo_b.weight` in layers 2 to 39.
- **Assembly:** in the oQ4e build these 38 tensors hold the same bytes as the original FP8 weights. So the edited tensors from that repository's patch file (`patch/heretic_patch.safetensors`) are written into the 4-bit shards as they are, with no requantization. We checked that the patch kept the block scales of all 38 tensors and that every other byte equals the base build.

If you already have the mlx-community build, the only difference is these 38 tensors.

## Size

| | |
|---|---|
| Parameters | about 763B in the files, the same as the base model repository (the DeepSeek-V4.1-Flash model card gives a 552B backbone plus 196B of Engram memory) |
| Size on disk | 432 GB (402 GiB) |
| Memory right after loading | about 403 GiB |
| Memory while serving | about 454 GiB observed, with the server context limit set to 262,144 |

The parameter counter on this page shows about 129B. It does not read the packing format of this build, so it counts packed storage words instead of the 4-bit weights inside them. No weights are missing.

GB is decimal, the way Hugging Face shows file sizes. GiB is binary, the way the 512 GB of a Mac's memory is counted.

## Measured (Mac Studio M3 Ultra 512 GB, oMLX 0.8.3)

| | |
|---|---|
| Generation speed | 36-40 tokens/s |
| Long context | a cold 243,819-token prompt was read in 537 s and answered correctly (server limit set to 262,144) |
| JailbreakBench JBB-Behaviors (100 prompts), with a 4,000-token thinking budget set in the runtime | 99 answered, 1 refused, 0 without an answer |
| The same run without a thinking budget | 83 answered, 0 refused, 17 gave no answer within 12,000 tokens (the model kept reasoning) |
| Tool calling, vision | not re-tested on this build |

Refusals are judged by a phrase heuristic on the first 300 characters of each answer (temperature 0). Prompts that ran out of tokens while the model was still reasoning were asked once more with a 12,000-token budget. This is not a judge-model evaluation. The author of the ablation reports 1 refusal of 100 on the same prompt set, judged by a model.

Tip: on prompts where the model reasons for a very long time, set a thinking budget in your runtime (in oMLX: the per-request `thinking_budget` field or the per-model setting).

## Requirements

A Mac with 512 GB of unified memory and an MLX runtime that supports this model family. Tested in oMLX 0.8.3.

## Credits

- DeepSeek for the base model.
- mlx-community and the oMLX project (oQ quantization) for the 4-bit build.
- pqhaz for the ablation, and the Heretic project for the method.

## License

MIT, the same as the base model. See `LICENSE`.

## Responsibility

This model answers requests that the base model refuses. You are responsible for how you use it.
