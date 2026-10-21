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

# Qwen3.8-27B W4A16, Gated DeltaNet left in bf16

A conservative int4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): the
MLP and full-attention projections go to int4, the Gated DeltaNet linear-attention projections stay
in bf16.

**In most cases you want
[Qwen3.8-27B-W4A16-GDN](https://huggingface.co/Nicolassuez/Qwen3.8-27B-W4A16-GDN) instead**, which
quantizes the GDN block as well. It is smaller AND faster AND more concurrent, at quality measured
identical. There is no axis on which this checkpoint wins. Keep reading for the numbers and for the
two cases where this one is still the right file.

## The name oversells it: only 35% of this checkpoint is in 4 bits

Measured from the safetensors headers, every tensor accounted for. Total 25.77 GiB on disk.

| block | precision | size | tensors |
| --- | --- | --- | --- |
| Gated DeltaNet linear attention | **bf16** | **10.36 GiB** | 432 |
| MLP | int4 | 8.22 GiB | 576 |
| `embed_tokens` | bf16 | 2.37 GiB | 1 |
| `lm_head` | bf16 | 2.37 GiB | 1 |
| Vision tower | bf16 | 0.86 GiB | 333 |
| Full attention | int4 | 0.81 GiB | 224 |
| MTP / NEXTN draft head | bf16 | 0.79 GiB | 15 |
| norms and misc | bf16 | ~0.00 GiB | 129 |

Quantized to int4: **9.03 GiB**. Left in bf16: **15.96 GiB**. The single largest weight block in this
model is the one that was NOT quantized, and it is larger than everything the pass actually touched.

## Why the GDN was excluded here, and why that reason was wrong

This checkpoint was built under the belief that serving engines had no int4 path for Gated DeltaNet.
That was tested afterwards and is false. The engine accepts int4 GDN weights and fails only in the
Marlin repack kernel, on exactly two tensors out of the whole block: `in_proj_a` and `in_proj_b`,
both 48 x 5120, which engines fuse into one `in_proj_ba` with 96 outputs where Marlin requires a
multiple of 64. Excluding those two, about 1% of the GDN's parameters, lets the other 99% compress.
That is what the `-GDN` sibling does.

## Measured

H100 PCIe, SGLang, context 65536, fp8 KV cache, card otherwise empty. Decode is single stream.

| checkpoint | weights | no speculation | NEXTN 3 | NEXTN 4 | concurrent requests |
| --- | --- | --- | --- | --- | --- |
| **this checkpoint** | 25.21 GB | 62 tok/s | 125 tok/s | - | 12 without spec, 4 with |
| `Qwen3.8-27B-W4A16-GDN` | **17.65 GB** | **83 tok/s** | **151 tok/s** | **155 tok/s** | **17 without spec, 6 with** |

30% fewer bytes bought 34% more decode, and the KV pool grew from 330326 to 461132 tokens. This is
not a size-against-speed trade: the engine runs at 86% of the card's bandwidth, so decode is simply
bandwidth divided by bytes read, and compressing more makes it faster.

## Quality

Both checkpoints were compared against `Qwen/Qwen3.8-27B-FP8` on the same engine and flags, 8 bits
being close to lossless and not our own work. On a 27-item verification set both score **24/27 with
thinking off and 27/27 with thinking on**, missing the same three arithmetic items as the official
FP8 reference, and both answer 4/4 on the multilingual set. At temperature 0 every verifiable
conclusion is identical across the three. So the extra 7.5 GB this checkpoint carries buys nothing
measurable in quality.

No delta against the bf16 source has been measured, so this is a not-broken gate rather than a
precision figure.

## When this file is still the right one

1. **Rollback insurance.** It is the fallback if a deployment on the `-GDN` checkpoint ever
   misbehaves, and it is the artifact the `-GDN` results were compared against.
2. **An engine without an int4 Marlin path for the GDN block.** If your runtime cannot repack those
   projections, this checkpoint loads where the other would abort at load with
   `size_n = 96 is not divisible by tile_n_size = 64`.

Outside those two cases, prefer the `-GDN` sibling.

## Serving

A standard `compressed-tensors` checkpoint. Native context 262144, and it carries an MTP head, so
speculative decoding works and is worth roughly 2x on decode. Serving notes, memory levers and the
token-map trick that adds another 26% are documented on the
[`-GDN` card](https://huggingface.co/Nicolassuez/Qwen3.8-27B-W4A16-GDN); they apply here unchanged.

Two traps also documented there and worth repeating: do NOT pack the vocabulary and serve it on a
stock SGLang (it loads, reports healthy, and answers with an unbroken run of `!`, with nothing in the
logs), and `reasoning_effort` accepts only `none`, `low`, `medium` and `xhigh` on this family's chat
template, so `high` returns a 400.

## Reproducing

Produced by `script_quantif_w4a16.sh` at its default settings, which is the `standard` profile of
`run_qwen38_27b_w4a16.sh`. The `-GDN` sibling is the same script with `MAX_COMPRESSION=1`. The
`recipe.yaml` in this repository records the exact modifier.
