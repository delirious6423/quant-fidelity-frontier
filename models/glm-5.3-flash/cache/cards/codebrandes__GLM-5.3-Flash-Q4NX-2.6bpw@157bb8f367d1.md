---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
  - exl3
  - q4nx
  - moe
  - cmp-170hx
---

# GLM-5.3-Flash EXL3 2.6 bpw, Q4NX codebook

This is an EXL3 quantization of [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) at
2.6 bits per weight, optimized for serving on two NVIDIA CMP 170HX cards. It keeps the whole model:
text, image and video input, the MTP layer and the full context.

It is ordinary EXL3 apart from one change. The routed experts, which are 97 % of the
parameters, use a new trellis codebook, **Q4NX**, in place of EXL3's default `mul1`. Q4NX is
designed so the experts run on the cards' INT8 tensor cores.

Only the CMP 170HX vLLM build has Q4NX kernels, so stock exllamav3, TabbyAPI and upstream vLLM
can't load this checkpoint. The recipe that serves it is
[AlexBrandes/cmp170hx-glm53-flash-exl3](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3).

## The Q4NX codebook

EXL3 stores weights as a bitshift trellis over 16×16 tiles, after Hadamard rotations. A
codebook turns each trellis state into weight values.

`mul1` uses a 16-bit state. Each state is multiplied by a constant, the four bytes of the result
are summed, and the sum is mapped to one FP16 value.

Running `mul1` on the CMP 170HX has two costs:

- Every weight needs FP16 arithmetic before it can be used.
- The decoded values only fit the FP16 tensor cores. Rounding them to INT8 loses precision. On
  this card INT8 tensor-core throughput is about 1.5× FP16 (189–200 TOPS against 115–137
  TFLOPS).

Q4NX keeps everything else in EXL3: the tiles, the rotations and sign vectors, the trellis, and
the Viterbi encoder. It changes how a state becomes weights:

- **Four weights per multiply.** The same multiply-by-a-constant gives a 32-bit word. Q4NX
  reads its eight nibbles in pairs, and each pair's sum is one weight. One multiply gives four
  weights where `mul1` gives one. Each trellis step therefore covers four weights, 10 bits at
  2.5 bpw.
- **Integer weights.** Each weight is an integer from −15 to 15, an exact INT8 tensor-core
  operand. Decoding needs no floating-point math. The scale is applied once, to the
  accumulated result.
- **A longer state.** The state grows from 16 bits to 32 minus the step size: 22 bits at
  2.5 bpw. Two lanes' states still fit one 32-bit read. The longer state gives the encoder more
  freedom, which recovers part of the accuracy lost to the coarser integer values.
- **Tensor-core order.** Tile values are stored in the order of the tensor core's operand
  fragment (`m16n8k16`), so a thread decodes its registers directly with no shuffling.

These are the measured speeds of the routed-expert kernels on one card, using real layers of
this model at the same 2.5 bpw:

| | `mul1` | Q4NX |
|---|---:|---:|
| Prefill, one MoE layer call at 2,048 tokens | 23.7 ms | **9.9 ms** |
| Decode, at the 1–8 rows per call used in serving | 1× | **1.45–1.59×** |

Q4NX was designed and tuned on the CMP 170HX. The same reasoning applies to other sm_80 GPUs,
but none have been tried.

The format is specified in
[doc/q4nx.md](https://github.com/AlexBrandes/exllamav3/blob/q4nx/doc/q4nx.md) on the `q4nx`
branch of our exllamav3 fork, with a reference decoder.

## What's in it

| Tensors | Codebook | Bits per weight |
|---|---|---:|
| Routed experts, layers 4–44 | Q4NX | 2.5 |
| Routed experts, layer 3 | `mul1` | 3.5 |
| Attention, MLA projections, indexer, shared experts, dense MLPs (layers 0–2), LM head | `mul1` | 6 (INT8 copy in `int8/`, see below) |
| MTP layer | `mul1` | 4 |
| Vision tower | `mul1` | 6 |

The checkpoint is 109 GB. The size can be quoted three ways:

- **2.6 bpw** excludes the LM head, as EXL3 checkpoints are usually labelled.
- **2.71 bpw** counts all 321.3 billion parameters.
- **About 2.52 bpw** is the figure for the routed experts alone.

### Dense weights: 6-bit EXL3 or INT8

The repo has two versions of the dense weights: every matrix outside the routed experts, plus
the LM head.

- **6-bit EXL3**, in the checkpoint itself. This keeps the checkpoint a complete EXL3 model.
- **INT8**, in `int8/dense.safetensors` (7.9 GB). These were made from the original weights,
  with one scale per output row. Each matrix is stored in the rotated input space that EXL3
  uses, along with its own sign vector, so it can stand in for the matching EXL3 tensor. The
  LM head is stored unrotated.

The recipe serves the INT8 copy. The engine loads it in place of the 6-bit tensors (set by
`VLLM_EXL3_DENSE_W8`), which lets the dense matrices run on INT8 tensor cores too. Every number
on this page was measured with this INT8 file.

Without the INT8 file, the engine runs the 6-bit tensors on its EXL3 kernels, at lower speed.

The INT8 file is about 1 GB larger per card than the 6-bit tensors it replaces, so it leaves
less room for the KV cache.

## How it was made

The checkpoint was converted from the original FP8 release (`zai-org/GLM-5.3-Flash`, revision
`46a050be`). The converter is exllamav3 1.6.6 with the
[`q4nx` branch](https://github.com/AlexBrandes/exllamav3/tree/q4nx) of our fork; doc/q4nx.md
has the exact command and recipe.

**Calibration:** 719 rows of 2,048 tokens.

| Rows | Source |
|---:|---|
| 250 | turboderp's self-generated chat trace for GLM-5.3-Flash |
| 469 | exllamav3's bundled sources: 172 code, 150 Wikipedia, 88 C4, 33 technical, 26 multilingual |

## Compared with turboderp's EXL3

The baseline is turboderp's
[GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) `sc_2.60bpw`. It's the
same size with `mul1` throughout, and it's what our previous recipe served. Both were measured
on 2× CMP 170HX with the same INT8 dense weights.

**Accuracy against the original model.** Lower is better.

| | turboderp EXL3 | This |
|---|---:|---:|
| KL divergence vs BF16, malaiwah fidelity suite (128 × 2,048-token windows) | 0.1559 | **0.1525** |
| Top-1 agreement with BF16 | 86.81 % | **87.00 %** |
| KL divergence vs FP8, fidelity-v1, all 857 sequences | 0.1570 | **0.1511** |
| … WikiText (565 passages) | 0.2104 | **0.1985** |
| … chat replies (280 prompts) | **0.0586** | 0.0616 |
| Perplexity, fidelity-v1 (FP8 reference: 2.441) | 2.747 | **2.727** |

- **BF16 suite:** the [malaiwah GLM-5.3-Flash fidelity suite](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1).
  It measures full-vocabulary KL on code, encyclopedic, literary and multilingual text.
- **fidelity-v1:** the FP8 release served by stock vLLM is the reference. It uses top-20
  logprobs over WikiText and over the FP8 model's own replies to 280 chat prompts.
- **What the comparison shows:**
  - This checkpoint is closer to the original overall and on raw text.
  - On chat replies it is about 5 % further in KL, with a 95 % interval of 2–8 %.
  - The difference matches the calibration mix, which adds web, Wikipedia and code text to
    turboderp's chat trace.

**Tasks**, greedy with thinking off:

| | turboderp EXL3 | This |
|---|---:|---:|
| GSM8K, chat, 5-shot | 93.1 % | 93.2 % |
| GSM8K, completion, 5-shot | 90.4 % | 91.6 % |
| IFEval, prompt-level strict | — | 68.8 % |
| Image and video checks | 7/7 | 7/7 |

**Speed**, sparkDash protocol, each served by its recipe:

| | Previous recipe (turboderp EXL3) | This recipe |
|---|---:|---:|
| Decode, 1 user, tok/s (code / json / prose / structured) | 88 / 87 / 76 / 102 | **162 / 152 / 111 / 176** |
| Prefill, 32k-token prompt, tok/s | 2,501 | **4,605** |

The speed gain comes from the Q4NX kernels plus engine work in the new recipe. The recipe page
has the full tables and settings.

## Licence and credits

MIT, the same licence as the base model.

- **Base model:** Z.ai's GLM-5.3-Flash.
- **EXL3:** turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3). Q4NX reuses
  its trellis, rotations and converter. The calibration trace is from turboderp's GLM-5.3-Flash
  conversion.
- **BF16 fidelity suite:** malaiwah, built on brandonmusic's BF16 reference logits.
