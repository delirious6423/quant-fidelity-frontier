---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - radiance
  - rocm
  - amd
  - rdna4
  - int8
  - gptq
  - trellis
---

# Qwen3.8-27B, KL-IQR-V1-4.00, Tiny class (radiance)

Qwen3.8-27B in one `.rad` file for the [radiance](https://codeberg.org/StillDeadcode/radiance) inference engine on AMD
RDNA4 cards (Radeon AI PRO R9700, gfx1201). It is the Tiny class entry (3 to 4 bits a weight) of the radiance quant challenge:
[PR #364](https://codeberg.org/StillDeadcode/radiance/pulls/364).

| | |
|---|---|
| Label | KL-IQR-V1-4.00 |
| File | `qwen3.8-27b-kl-tiny.rad` |
| Size | 3.9980 bits a weight over the text plane |
| Weights | IQR computed trellis, quarter-step widths, mostly 2^20 states, row slabs |
| Activations | int8, a scale per 128 columns, two passes |
| Head | vocabulary-weighted, 3-bit to 6-bit classes, 16-bit input |
| Embedding | a 3-bit table trellis |

## Running it

It needs radiance built from the entry's branch,
[Perire/radiance `qc/qwen38-kl-guided-iqr`](https://codeberg.org/Perire/radiance/src/branch/qc/qwen38-kl-guided-iqr),
for gfx1201. Stock radiance releases, 1.17.0 included, do not read this container: they lack its 2^20-state forms,
quarter steps, row slabs and activation forms.

    radiance --model qwen3.8-27b-kl-tiny.rad --tp 2 --kv-cache-dtype bf16 --tp-wire exact \
             --max-model-len 262144 --max-num-seqs 16

## How it was made

`docs/QC-QWEN38-KL.md` on the branch describes each step, and `data/recipes/q38-27b-kl-tiny.recipe` is the recipe.
