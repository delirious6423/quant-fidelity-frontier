---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
  - exl3
  - a4nx
  - moe
  - cmp-170hx
---

# DeepSeek-V4.1-Flash-1017 EXL3 2.2 bpw, A4NX codebook

This is an EXL3 quantization of
[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at 2.2 bits
per weight, made for serving on three NVIDIA CMP 170HX cards (64 + 40 + 64 GB). It keeps the
whole model: text and image input, the three MTP layers the DSpark drafter uses, and the full
context.

It is ordinary EXL3 apart from one change. The routed experts use the **A4NX** trellis codebook
in place of EXL3's default `mul1`, so they run on the cards' INT8 tensor cores.

Only the CMP 170HX vLLM build has A4NX kernels, so stock exllamav3, TabbyAPI and upstream vLLM
can't load this checkpoint. The recipe that serves it is
[AlexBrandes/cmp170hx-deepseek-v41-flash](https://github.com/AlexBrandes/cmp170hx-deepseek-v41-flash).

## The A4NX codebook

EXL3 stores weights as a bitshift trellis over 16×16 tiles, after Hadamard rotations. A codebook
turns each trellis state into weight values. `mul1` maps each 16-bit state to one FP16 value.
A4NX keeps the tiles, rotations, trellis and Viterbi encoder, and changes the decode:

- **Four weights per multiply.** One multiply-by-a-constant gives a 32-bit word whose eight
  nibbles, summed in pairs, are four weights.
- **Integer weights** from −15 to 15: exact INT8 tensor-core operands, decoded with no
  floating-point math. The scale is applied once, to the accumulated result.
- **A longer state**, 32 bits minus the step size: 24 bits at 2.0 bpw and 22 at 2.5. The extra
  freedom recovers part of the accuracy the coarser values lose.
- **Tensor-core order.** Values are stored in the order of the `m16n8k16` operand fragment, so a
  thread decodes its own registers.

The format is specified in
[doc/a4nx.md](https://github.com/AlexBrandes/exllamav3/blob/6cfd44b2fca93bf5420f416a0c05e3563af2f4a0/doc/a4nx.md) on the
`a4nx` branch of our exllamav3 fork, with a reference decoder. A4NX was first released for
GLM-5.3-Flash at 2.5 bpw; this model adds the 2.0-bpw rate.

## What's in it

| Tensors | Codebook | Bits per weight |
|---|---|---:|
| Routed experts, layers 0–6 and 36–39 | A4NX | 2.5 |
| Routed experts, layers 7–35 | A4NX | 2.0 |
| Routed experts of the three MTP layers | A4NX | 2.0 |
| Attention (with the compressor and indexer), shared experts, Engram projections, the MTP layers' other matrices, LM head | `mul1` | 6 (INT8 copy in `int8/`, see below) |
| Vision tower, token embeddings | — | BF16 |
| Engram tables | not included | |

The checkpoint is 166.8 GB, the INT8 file included. **2.2 bpw** is the converter's figure for
the model without the LM head (2.18); the routed experts alone average 2.14.

### Engram tables

DeepSeek-V4.1's Engram tables (shards 47–48 of the original repo, 203 GB) are not converted and
not copied here. `ngram_external.json` names the original shards that hold them; the engine reads
them from those files at run time. The recipe's `download.sh` fetches them from
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

### Dense weights: 6-bit EXL3 or INT8

- **6-bit EXL3**, in the checkpoint itself, so it stays a complete EXL3 model.
- **INT8**, in `int8/dense.safetensors` (8.1 GB), made from the original weights with one scale
  per output row. Each matrix is stored in EXL3's rotated input space with its own sign vector,
  so it can stand in for the matching EXL3 tensor; the LM head is stored unrotated.

The recipe serves the INT8 copy (`VLLM_EXL3_DENSE_W8`), which runs the dense matrices on INT8
tensor cores too. Every number on this page was measured with it.

## How it was made

Converted from the original release (`deepseek-ai/DeepSeek-V4.1-Flash-1017`, revision
`1bae0e10`) with exllamav3 1.6.6 and the
[`a4nx` branch](https://github.com/AlexBrandes/exllamav3/tree/6cfd44b2fca93bf5420f416a0c05e3563af2f4a0) of our fork (commit `6cfd44b2`);
doc/a4nx.md has the exact command and recipe.

**Calibration:** 512 rows of 2,048 tokens (725,542 tokens).

| Rows | Source |
|---:|---|
| 403 | The model's own replies (DeepSeek-V4.1-Flash-1017, thinking on), rendered with its chat encoding |
| 109 | exllamav3's bundled sources (code, Wikipedia, C4, technical, multilingual) |

## Accuracy

The reference is the original release (FP8 dense weights, MXFP4 experts) served by vLLM 0.34.0
on 4× RTX PRO 6000. This checkpoint was served by the recipe's engine on 3× CMP 170HX. Both were
measured by us with the same prompts and settings.

**Tasks**, with thinking off:

| | Original | This |
|---|---:|---:|
| GSM8K, chat, 5-shot | 97.3 % | 96.4 % |
| GSM8K, completion, 5-shot | 90.8 % | 89.3 % |
| HumanEval+, chat | 87.2 % | 88.4 % |
| IFEval, prompt-level strict | 88.2 % | 86.9 % |
| MMLU | 85.7 % | 83.8 % |
| RULER, 32k | 87.7 % | 84.3 % |

**Distance from the original**, teacher-forced on the original's own outputs, top-20 logprobs:

| | KL divergence | Same top token | Perplexity ratio |
|---|---:|---:|---:|
| Chat replies (280 prompts) | 0.119 | 93.0 % | 1.124 |
| WikiText (565 passages) | 0.207 | 85.4 % | 1.133 |

## Licence and credits

MIT, the same licence as the base model.

- **Base model:** DeepSeek's DeepSeek-V4.1-Flash-1017.
- **EXL3:** turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3). A4NX reuses its
  trellis, rotations and converter.
