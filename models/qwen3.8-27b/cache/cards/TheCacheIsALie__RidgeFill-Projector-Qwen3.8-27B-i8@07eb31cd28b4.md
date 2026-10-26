---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - ridgefill
  - llama.cpp
  - gguf
  - prefill-acceleration
  - qwen3.8
---

# RidgeFill Projector for Qwen3.8-27B (INT8, split 16)

Per-layer linear maps for RidgeFill in the llama.cpp fork
[Approximate-Science/llama.cpp-ridgefill](https://github.com/Approximate-Science/llama.cpp-ridgefill).
RidgeFill is a training-free prefill accelerator. Layers 0–15 run for every prompt row. For the other prompt rows,
except an exact tail and a selected set of exact rows, the projector predicts the inputs of layers 16–63 from the
layer-15 hidden state. Generation after the prompt is exact.

By Dylan Johnston and tcclaviger. Cite with DOI [10.5281/zenodo.23179168](https://doi.org/10.5281/zenodo.23179168).
License: Apache-2.0. The authors, license and DOI are also stored in the GGUF metadata.

## What this file is
- One GGUF file of int8 ridge maps (split 16) and per-layer Gated-DeltaNet state corrections for Qwen3.8-27B.
- It is not a language model. It is loaded with `--ridgefill` next to an unmodified Qwen3.8-27B GGUF.
- Fitted for Qwen3.8-27B; measured with `unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-UD-Q4_K_XL.gguf` (revision 4ca7207).

## Files
| File | Bytes | SHA256 |
|---|---|---|
| ridgefill-projector-qwen3.8-27b-i8-S16.gguf | 1,452,188,032 | 8646d41a5a28d60ef53cc5e1fee4b2bfb20d0775db88d3f89d0731ad9f6b629d |

## Use
```sh
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 --local-dir models
hf download TheCacheIsALie/RidgeFill-Projector-Qwen3.8-27B-i8 ridgefill-projector-qwen3.8-27b-i8-S16.gguf --local-dir models
```
Build the fork and start `llama-server` with the recommended settings listed in
[RIDGEFILL.md](https://github.com/Approximate-Science/llama.cpp-ridgefill/blob/main/RIDGEFILL.md) (Quick start, Qwen3.8-27B).
Prompts shorter than 8192 tokens run exactly. Draft models and MTP are refused while RidgeFill is enabled.

## Measured (4× A100 PCIe 40 GB, CUDA; stock = upstream llama.cpp 1242be66f; same flags except RidgeFill)
| Context | TTFT speedup, projector in VRAM | TTFT speedup, projector streamed from host | Extra VRAM, in VRAM / streamed |
|---|---|---|---|
| 16,384 | 1.25× | 1.19× | 1592 / 268 MiB |
| 32,768 | 1.30× | 1.27× | 1802 / 478 MiB |
| 65,536 | 1.31× | 1.30× | 2270 / 944 MiB |
| 120,000 | 1.28× | 1.29× | 3014 / 1732 MiB |

| Benchmark | Stock | RidgeFill | Δ [95% CI] |
|---|---|---|---|
| RULER 16K, 13 tasks × 20 | 85.69 | 84.51 | −1.19 [−3.13, +0.79] |
| RULER 32K, 13 tasks × 20 | 87.85 | 86.42 | −1.42 [−3.21, +0.35] |
| LongBench v2, official CoT, full length, 503 items | 38.57 | 37.18 | −1.39 [−5.17, +2.19] |

Post-prompt next-token agreement with stock (100 LongBench documents per length): top-1 94.6% at 16K and 93.6% at 32K;
median KL 0.0040 / 0.0042 nats per token. Output is not identical to stock llama.cpp.
