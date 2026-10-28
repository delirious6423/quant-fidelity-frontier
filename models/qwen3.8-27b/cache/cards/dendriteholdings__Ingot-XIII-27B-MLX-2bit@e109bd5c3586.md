---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- apple-silicon
- 2-bit
- ternary
- ingot
- bittensor
---

# Ingot-XIII-27B-MLX-2bit

A public mirror of a model that won on Ingot, a Bittensor subnet where miners quantize a fixed teacher model and the validator crowns the copy that stays closest to it. This is the MLX pack of [dendriteholdings/Ingot-XIII-27B-Q2](https://huggingface.co/dendriteholdings/Ingot-XIII-27B-Q2): the same ternary weights, written for Apple Silicon.

- Submitted by miner hotkey `5FufNHX56yDnddSpzkXcedqv8o1YrKYEQmP54BsYw35Y2pPr` (uid 205)
- Crowned from submission `79bf79a1-c35d-4c4f-88d8-a626bd6fabd4`

The weights are the miner's upload converted without loss: every 128-weight block keeps its ternary codes and fp16 scale as MLX affine 2-bit values (scale d, bias −d), and every weight decodes to the same number the gguf kernel uses. Credit belongs to the miner. The vision tower, config, tokenizer and Hadamard sign vectors are Prism's [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit), which the competition does not change.

## Files

| file | bits per weight | size |
| --- | --- | --- |
| `model.safetensors` | 2.25 (2-bit codes, fp16 scale and bias per 128 weights) | 8.60 GB |

## How close it is to the teacher

Per answer token, against the BF16 teacher reading the same answer: KL(teacher ‖ model) of the next-token distributions, and the hidden-state distance |h − h_teacher| / |h_teacher| averaged over every 8th layer. Lower is closer.

| | KL | hidden-state distance |
| --- | --- | --- |
| this model, in its winning eval | 0.08544 | 0.3017 |
| the king it beat, same eval | 0.08680 | 0.3051 |
| closer than genesis [dendriteholdings/Ingot-genesis-27B-Q2](https://huggingface.co/dendriteholdings/Ingot-genesis-27B-Q2) | 25.1% | 33.1% |

## Speed

Decode tokens per second at batch 1 on a Mac Studio M3 Ultra (80-core GPU, 512 GB), measured on Ingot king IV. Every q2 king has the same layout and size, so the numbers hold for all of them within run-to-run noise. "Plain" is one token per forward pass; "speculative" uses MTPLX's multi-token draft head (`--generation-mode mtp --depth 1`). The server numbers are 256 generated tokens on a code prompt, timed over HTTP; the in-process numbers are 128 tokens with no server.

| engine | mode | decode tok/s |
| --- | --- | --- |
| MTPLX 2.12 (mlx 0.32) server | plain | 55.3 |
| MTPLX 2.12 (mlx 0.32) server | speculative | 66.6 |
| mlx-lm 0.31 graph with MTPLX's ternary kernels, in-process | plain | 59.9 |

For comparison, the gguf twin of these weights decodes at 56.4 tok/s on the same Mac in the [Dendrite llama.cpp fork](https://github.com/DendriteLabs/llama.cpp) and at 137 tok/s on an RTX PRO 6000; see [its card](https://huggingface.co/dendriteholdings/Ingot-XIII-27B-Q2).

## Running it

```
pip install mlx-lm
mlx_lm.generate --model dendriteholdings/Ingot-XIII-27B-MLX-2bit --prompt "Hello"
```

The pack loads in any runtime that reads Prism's `prism_hadamard_qwen35` layout, MTPLX among them. The gguf twin runs in Prism's llama.cpp fork; see [its card](https://huggingface.co/dendriteholdings/Ingot-XIII-27B-Q2).
