---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
  - qwen3_5
  - compressed-tensors
  - gptq
  - int4
  - w4a16
  - sglang
---

# Qwen3.8-27B W4A16, Gated DeltaNet included

An int4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in which the
**Gated DeltaNet projections are quantized too**, which is the single thing that separates it from
the sibling [Qwen3.8-27B-W4A16](https://huggingface.co/Nicolassuez/Qwen3.8-27B-W4A16) where they stay
in bf16.

That one difference is worth **7.5 GB and 24% of decode**, because those projections are the largest
weight block in this architecture: 10.36 GiB across 48 of the 64 layers, larger than everything a
conventional W4A16 pass actually touches.

Weights only. No distillation, no fine-tuning, no change to the tokenizer or the chat template.

## Method

GPTQ, scheme W4A16: symmetric int4 weights, group size 128, activations left in bf16. Calibrated on
512 samples of `HuggingFaceH4/ultrachat_200k` at 2048 tokens, through `llmcompressor`. Output format
is `compressed-tensors`, `pack-quantized`.

## What is quantized, and what is not

Sizes are the bf16 footprint of each block in the source checkpoint.

| block | tensors | precision here | note |
| --- | --- | --- | --- |
| MLP gate / up / down, 64 layers | 576 | **int4** | |
| Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `out_proj`, 48 layers | 432 | **int4** | 10.36 GiB in bf16 |
| Full attention q / k / v / o, 16 layers | 224 | **int4** | |
| `lm_head` and `embed_tokens` | 2 | bf16 | 2.37 GiB each, see below |
| Vision tower | 333 | bf16 | 0.86 GiB, encoders are not int4-safe |
| MTP / NEXTN draft head | 15 | bf16 | shipped as `model-mtp-bf16.safetensors` |
| GDN `in_proj_a`, `in_proj_b`, 48 layers | 96 | bf16 | 47 MB total, see below |
| norms, `conv1d`, `A_log`, `dt_bias`, biases | 129 | bf16 | GPTQ targets `Linear` only |

**Why `in_proj_a` and `in_proj_b` are the exception.** Marlin requires the output dimension to be a
multiple of 64. Across the whole GDN block exactly two tensors fail that rule, both 48 x 5120, and
serving engines fuse them into a single `in_proj_ba` module with 96 outputs, so the repack kernel
aborts with `size_n = 96 is not divisible by tile_n_size = 64`. Excluding those two, about 1% of the
GDN's parameters, is what lets the other 99% compress.

**Why the vocabulary stays bf16.** `lm_head` and `embed_tokens` are not `Linear` modules, so a GPTQ
pass never sees them. They can be packed separately, but read the trap below before doing it.

## Measured

H100 PCIe, SGLang, context 65536, fp8 KV cache, card otherwise empty. Decode is single stream.

| checkpoint | weights | no speculation | NEXTN 3 | NEXTN 4 | concurrent requests |
| --- | --- | --- | --- | --- | --- |
| `Qwen3.8-27B-W4A16` | 25.21 GB | 62 tok/s | 125 tok/s | - | 12 without spec, 4 with |
| **this checkpoint** | **17.65 GB** | **83 tok/s** | **151 tok/s** | **155 tok/s** | **17 without spec, 6 with** |

30% fewer bytes bought 34% more decode, and the KV pool grew from 330326 to 461132 tokens. This
engine runs at 86% of the card's measured bandwidth, so decode is simply bandwidth divided by bytes
read: compressing more makes it faster rather than slower.

Against Qwen's own 8-bit release, served on the same engine and flags:

| checkpoint | weights | no speculation | NEXTN 4 | implied bandwidth |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B-FP8` | 28.47 GB | 55 tok/s | 117 tok/s | 1566 GB/s |
| `Qwen3.8-27B-W4A16` | 25.21 GB | 62 tok/s | 125 tok/s (depth 3) | 1563 GB/s |
| this checkpoint | 17.65 GB | 83 tok/s | 155 tok/s | 1465 GB/s |

## Quality

Compared against `Qwen/Qwen3.8-27B-FP8` rather than against another int4 build, 8 bits being close
to lossless and not our own work.

- 27-item verification set: **24/27 with thinking off, 27/27 with thinking on**, and the three items
  missed are the SAME three arithmetic items missed by the official FP8 and by the plain W4A16.
  Identical failures are not quantization damage.
- Multilingual set: 4/4, same as both references.
- At temperature 0, every verifiable conclusion is identical across the three checkpoints, including
  an exact fraction, a closed form with its own check, an induction proof, and translations into
  French, Arabic, Mandarin, Japanese and Russian.
- Long context: a needle placed in a 249887-token prompt is retrieved. This is the test that would
  expose accumulated error in the recurrent GDN state, and it passes.

No delta against the bf16 source has been measured, so this is a not-broken gate rather than a
precision figure.

## Serving

Runs on SGLang and on vLLM as a standard `compressed-tensors` checkpoint. Native context is 262144.

The checkpoint carries an MTP head, so speculative decoding works and is worth roughly 2x on decode.
On SGLang, `--speculative-algo NEXTN --speculative-num-steps 4 --speculative-num-draft-tokens 5`,
depth 4 being the measured optimum (3 gives 151, 4 gives 155, 5 gives 150).

**The draft head is the bottleneck, and a token map fixes it.** The NEXTN draft owns no `lm_head`, it
reads the target's, once per draft token, so five times per iteration at depth 4. At 248320 x 5120 in
bf16 that is 2.37 GiB read five times, more than a third of the bandwidth spent on one matrix.
`--speculative-token-map` restricts what the draft may PROPOSE while the target still verifies over
its full vocabulary, so speculation stays lossless. Measured: **151 to 191 tok/s, and the accept
length rises from 3.27 to 3.67**, because a head restricted to plausible rows has sharper logits than
one spreading mass over 183000 tokens it would never usefully propose.

Memory levers, all measured free on decode: `--mamba-ssm-dtype bfloat16` (smaller AND faster, the
state is re-read every step), `--max-mamba-cache-size` matched to `--max-running-requests`, and
`--disable-prefill-cuda-graph`. Note that on this hybrid architecture it is the **GDN state cache**
that caps concurrency, never the KV pool, and a refusal to start always comes from there.

## Two traps

**Do not pack the vocabulary and serve it on a stock SGLang.** A quantized `lm_head` loads without
error, the server reports healthy, its own warmup passes, and then every prompt is answered with an
unbroken run of `!`. Nothing in the logs flags it. The cause is a missing dispatch:
`CompressedTensorsConfig.get_quant_method` tests `LinearBase` then `FusedMoE` and returns `None` for
a vocabulary module, so the table falls back to the unquantized method, whose parameters the
checkpoint never fills. This checkpoint deliberately keeps both tables in bf16 and is therefore safe
as published; the warning is about variants.

**`reasoning_effort` is stricter than the OpenAI vocabulary.** This family's chat template accepts
`none`, `low`, `medium` and `xhigh` (the default) and raises on anything else, so `high` returns a
400. Thinking is on by default, and on hard prompts the model will spend a whole small token budget
reasoning and return an empty message, so give it a large `max_tokens` or lower the effort.

## Reproducing

Produced by `script_quantif_w4a16.sh` with `MAX_COMPRESSION=1`, which is the `gdn` profile of
`run_qwen38_27b_w4a16.sh`. The `recipe.yaml` in this repository records the exact modifier.
