---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - oq
  - quantized
  - mtp
  - moe
  - ornith
  - 4-bit
---

# Ornith-1.5-35B-A3B — MLX oQ4e + MTP

A mixed-precision 4-bit MLX quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (revision `10fbf86fed7ecee4a061f8b499a618f46001cac1`): Ornith AI's reasoning MoE (35B total / ~3B active per token, `qwen3_5_moe`), post-trained from Qwen3.6 35B A3B for coding agents and tool use. The vision tower and the model's own MTP (multi-token prediction) draft head are kept, so runtimes that support it can use speculative decoding. Built for [Midium](https://midium.dev).

## What was quantized

- **Quantizer:** oMLX 0.7.2's oQ4e (oQ level 4, enhanced: importance-weighted quantization from an activation imatrix; calibration set `oqe_code_multilingual`, 128 samples × 512 tokens, taken on a uniform 4-bit proxy because the bf16 model is above oQ's in-memory calibration limit on a 128 GB Mac).
- **Bit layout:** taken tensor for tensor from [Jundot/Qwen3.6-35B-A3B-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.6-35B-A3B-oQ4e-mtp) instead of oQ 0.7.x's own allocation: routed experts 4-bit (group 64); attention and linear-attention projections 5–6-bit (group 64/128) on most layers, 8-bit where that pack keeps them; shared experts, embeddings and `lm_head` 8-bit; conv1d, router gates and norms unquantized. Same quantized-tensor set, bits and group size as that pack (483 quantized tensors). Newer oQ4e builds keep every attention projection at 8-bit, which reads ~24% more weight bytes per token.
- **Kept as in the source:** the vision tower (bf16); the MTP head (`mtp.*`, attention 8-bit, routed experts 4-bit, `fc` bf16); the chat template, tokenizer and generation config.

Weights: 21.6 GB (bf16 source: 67 GB).

## Evaluation

**Against the bf16 model** — teacher-forced next-token loss on raw text (no chat template; 6 × 2,048-token code windows and 4 prose windows, last 1,024 positions each):

| | NLL change vs bf16 | Top-1 agrees with bf16 |
|---|---:|---:|
| Code (6,144 tokens) | +0.0016 ± 0.0075 | 94.6% |
| Prose (4,096 tokens) | +0.0026 ± 0.0123 | 89.0% |

**Against [scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp](https://huggingface.co/scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp)** (oMLX 0.7.4.dev1 oQ4e, 8-bit attention), same Midium engine session, M5 Max 128 GB, one model loaded at a time.

Tool calling (BFCL v4 single-call and multi-turn, BFCL's own scorer, thinking on, greedy, the same 260 cases for both): the two packs score within noise of each other.

Decode speed (tok/s; temperature 1.0, top-p 0.95, top-k 20, thinking off; 6 runs with MTP drafting, 2 without, per job):

| Job | This pack, drafting | scottlowry, drafting | This pack, no drafting | scottlowry, no drafting |
|---|---:|---:|---:|---:|
| Short coding ask | 240 | 225 | 145 | 123 |
| New module after a ~33k-token codebase | 147 | 132 | 117 | 104 |
| Rewrite a 24,000-character file with one change | 586 | 585 | 138 | 119 |
| The same edit as a `write_file` tool call | 444 | 435 | 117 | 104 |

## Usage

Loads with `mlx-vlm` (`qwen3_5_moe`). Ornith is a reasoning model: the assistant turn opens with `<think>` unless `enable_thinking` is false in the chat template kwargs. Tool calls use Qwen's XML format (`<tool_call><function=…><parameter=…>`). Recommended sampling (model card): temperature 0.6, top-p 0.95, top-k 20 for general use; temperature 1.0 to reproduce the reported benchmarks.

In Midium the MTP head is split into a drafter on first load and used for speculative decoding automatically.

## License

MIT, as the source model.
