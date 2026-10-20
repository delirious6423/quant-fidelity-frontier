---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: radiance
tags:
  - radiance
  - exl3
  - moe
  - rocm
  - gfx1201
---

# Qwen3.8-Flash-Next: shared-input-vector EXL3 experts for radiance (research container)

A [radiance](https://huggingface.co/StillDeadcode) container of **Qwen3.8-Flash-Next** whose routed experts are EXL3
with **one shared input vector per layer** for the routed gate/up matrices. Built 2026-10-18 as a research artifact
for [radiance-exl3](https://github.com/Dyluhn/radiance-exl3).

- Routed experts of layers 1-47 and the MTP head: exllamav3 1.5.2 + a shared-`suh` overlay with a joint per-layer
  scale refit, turboderp's 4.05 settings, 4 bpw.
- Layer 0's routed experts: turboderp's 4.05 (`turboderp/Qwen3.8-Flash-Next-exl3@55a732e0`), because 1.5.2's allocator
  gave our layer 0 5 bpw.
- Every other weight is byte-identical to radiance's published container `StillDeadcode/qwen3.8-next-flash-fp8-iq4r-moe`.
- `p6-expert-subset/` holds the EXL3 routed-expert tensors the container was built from, with their fetch provenance;
  `reports/` the build, check and `rad-info` logs.

**Quality** (radiance KL mode vs the bf16 model on ref64, 52,132 scored positions): KL 0.048560 for this container vs
0.046732 for the turboderp-4.05 container [`Dyluhn/qwen3.8-flash-next-exl3`](https://huggingface.co/Dyluhn/qwen3.8-flash-next-exl3)
(+3.9 %, paired 95 % CI -0.7 % to +8.6 %, so not significant); top-1 agreement equal. No control conversion was run.

It is meant for radiance-exl3's shared-input-vector kernels, which are in development and not released; it was not
tested with the released plugins. Licence: Qwen Community License 1.0 (`LICENSE`).
