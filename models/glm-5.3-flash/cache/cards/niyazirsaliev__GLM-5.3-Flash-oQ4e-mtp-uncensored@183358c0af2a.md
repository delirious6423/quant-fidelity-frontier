---
license: mit
library_name: mlx
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- 4-bit
- abliterated
- uncensored
- mtp
---

# GLM-5.3-Flash-oQ4e-mtp-uncensored

An uncensored GLM-5.3-Flash for Apple Silicon at 4-bit quality: MLX format, oQ4e mixed precision (4.641 bits/weight), with the vision encoder and the multi-token prediction (MTP) head. It fits a Mac with 256 GB of memory.

## How it was made

This build puts two published works together. Our part is the assembly and the checks.

- **Base:** [Jundot/GLM-5.3-Flash-oQ4e-mtp](https://huggingface.co/Jundot/GLM-5.3-Flash-oQ4e-mtp), the 4-bit oQ4e build made with oQ in [oMLX](https://github.com/jundot/omlx).
- **Uncensoring:** the directional ablation by mrfakename, [mrfakename/GLM-5.3-Flash-abliterated](https://huggingface.co/mrfakename/GLM-5.3-Flash-abliterated): a rank-2 edit of 90 tensors, `W' = W - s·Q·(QᵀW)`, with the basis `Q` and the per-layer strengths `s` published in `abliteration-edit.npz`. The tensors are `self_attn.o_proj` in layers 0 to 44, `mlp.shared_experts.down_proj` in layers 3 to 44 and `mlp.down_proj` in the dense layers 0 to 2.
- **Assembly:** in the oQ4e build these 90 tensors are 8-bit, so the edited FP8 tensors cannot be copied in directly. We applied the published edit to the original [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) FP8 weights, checked the result against mrfakename's published tensors, quantized the 90 tensors to the 8-bit affine group-64 layout of the base build and wrote them into the same byte positions. The codes are rounded against the base build's stored scales and biases and then refit by least squares. Every other byte equals the base build: routed experts, vision encoder and MTP head are untouched.

## Size

| | |
|---|---|
| Parameters | 321B, the same as the base model |
| Size on disk | 186 GB (174 GiB) |
| Memory in use | about 196 GiB |

GB is decimal, the way Hugging Face shows file sizes. GiB is binary, the way the 256 GB of a Mac's memory is counted.

## Measured (Mac Studio M5 Ultra 256 GB, oMLX 0.7.3rc1)

| | |
|---|---|
| Generation speed | 70-95 tokens/s |
| JailbreakBench JBB-Behaviors (100 prompts) | 99 answered, 1 refused, 0 without an answer |
| Same check on the standard 4-bit build (10 prompts of our own) | standard build refused 8 of 10; this build refused 0 |
| Tool calling, vision, long context | not re-tested on this build |

Refusals are judged by a phrase heuristic on the first 300 characters of each answer (temperature 0). Prompts that ran out of tokens while the model was still reasoning were asked once more with a 12,000-token budget. This is not a judge-model evaluation. The author of the ablation reports 3 refusals of 100 on their own prompt set (8-bit GGUF).

## Requirements

A Mac with 256 GB of unified memory and an MLX runtime that supports this model family. Tested in oMLX 0.7.3rc1.

## Credits

- Z.ai (zai-org) for the base model.
- Jundot and the oMLX project (oQ quantization) for the 4-bit build.
- mrfakename for the ablation.

## License

MIT, the same as the base model. See `LICENSE`.

## Responsibility

This model answers requests that the base model refuses. You are responsible for how you use it.
