---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - gguf
  - mixed-quant
  - qwen4exp
  - qwen3.8-flash-next
  - dgx-spark
  - ds4
---

# Qwen3.8-Flash-Next Mixed-Quant GGUF

## Support my work

I work on making large language models practical on hardware they were never really designed to fit on — through mixed quantization, inference optimization, custom kernels, and serving experiments.

While much of the development happens on local hardware, calibration, profiling, and large-scale validation often require expensive on-demand GPUs.

Contributions help pay for that compute, storage, and testing infrastructure, so I can keep experimenting and publishing the results openly.

<a href="https://www.buymeacoffee.com/baekpica" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="height: 60px !important;width: 217px !important;"></a> <a href="https://github.com/sponsors/Baekpica" target="_blank"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor Baekpica on GitHub" style="height: 60px !important;width: 217px !important;"></a>

> **Runtime work starts with the SSD-offload variant:** ds4 implementation and DGX Spark validation are being carried out first against [`Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF`](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF), the SSD-offload track for this model family.

> **Q2_K artifact available:** the four-shard no-imatrix `MQ-NG5-Q2K-Q4-SPLIT-E4` GGUF is published below. Its local structure/full-file hashes and remote LFS hashes are verified. Model quality, ds4 runtime behavior, and DGX Spark serving remain provisional until the later validation gates are complete. The calibrated `MQ-NG5-Q4-SPLIT-E5` recipe remains published for review, but its weight build is paused while the higher-priority [SSD-PLE Q5/Q6 track](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF) is produced.

This repository hosts DGX Spark/GB10-oriented mixed-precision GGUF conversions of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), pinned to source revision [`f5d08274bafd880402bd16f5e3e6c514136ec06c`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/f5d08274bafd880402bd16f5e3e6c514136ec06c).

The planned runtime is the [`dfm` branch of Baekpica/ds4](https://github.com/Baekpica/ds4/tree/dfm), initially grounded at ds4 tag [`v0.6.3-dfm`](https://github.com/Baekpica/ds4/releases/tag/v0.6.3-dfm). This is a custom `qwen4exp` GGUF schema; compatibility with upstream llama.cpp or other GGUF runtimes is **not** implied.

## Release tracks

Two tracks keep the calibration dependency and precision trade-off explicit:

| Candidate | Interior routed gate/up | Calibration dependency | Artifact size | Status | Conservative Spark headroom |
|---|---:|---:|---:|---:|---:|
| **`MQ-NG5-Q2K-Q4-SPLIT-E4`** | `Q2_K` | none; standard weight-only quantization | **98.50165 GB / 91.7368 GiB actual** | **published; integrity verified** | **4.10 GiB projected** |
| **`MQ-NG5-Q4-SPLIT-E5`** | `IQ2_XXS` on layers 3–45 | merged official-model activation importance matrix | **93.85 GB / 87.40 GiB projected** | **calibration passed; recipe retained, weight build paused** | **8.43 GiB projected** |

The Q2_K candidate was produced first. “No imatrix” means its Q2_K/Q4_K blocks are quantized directly from the pinned BF16 weights; it does not waive H200-quality, runtime, or DGX Spark validation.

### Published Q2_K files

The complete artifact is in [`MQ-NG5-Q2K-Q4-SPLIT-E4/`](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-GGUF/tree/main/MQ-NG5-Q2K-Q4-SPLIT-E4). It contains four GGUF shards, [`SHA256SUMS`](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-GGUF/blob/main/MQ-NG5-Q2K-Q4-SPLIT-E4/SHA256SUMS), the quantization recipe, artifact manifest, and strict structure-verification report.

- Total: **98,501,648,608 bytes / 91.7368 GiB**.
- Structure: **1,756 tensors, zero verifier errors**.
- Type counts: 88 Q2_K, 56 Q4_K, 48 Q4_0, 128 Q5_1, 486 Q8_0, 363 BF16, 584 F32, and 3 I64 tensors.
- Integrity: independent local full-file SHA-256 passed; all four remote LFS SHA-256 values and byte counts match the local manifest.

### Calibrated E5 decision

Three official-model H200 passes contributed 1,510,400 processed tokens and 724,992,000 routed selections across a primary instruction/reasoning mix and an independent expanded multilingual/chat/science/table/code/safety/web corpus. The initial calibrated E4 scope had no zero cells and p1 3,167.97, but nine cells in layer 2 remained below the locked 512-route minimum; its least-used expert was observed only twice.

The release recipe therefore promotes layer 2 gate/up to `Q4_K` instead of forcing an under-supported `IQ2_XXS` quantization. Over the final IQ2 scope, layers 3–45, the minimum is **545 routes**, p1 is **1,193.93 routes**, and zero-coverage cells are **0**. All pass inputs, corpus hashes, overlap accounting, accumulator/imatrix hashes, and thresholds are recorded in the [`MQ-NG5-Q4-SPLIT-E5` recipe](https://huggingface.co/Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-GGUF/blob/main/MQ-NG5-Q4-SPLIT-E5/quant-recipe.yaml). These route floors are operational calibration criteria, not a claim of downstream model quality.

| Model region | Q2_K no-imatrix track | Calibrated IQ2_XXS track | Reason |
|---|---:|---:|---|
| 51.2B-parameter PLE n-gram table (128 shards) | `Q5_1` | `Q5_1` | Dominant residency cost; higher precision than the most aggressive routed experts while preserving Spark headroom. |
| Routed expert gate/up, interior layers 2–45 | `Q2_K`, no imatrix | layer 2 `Q4_K`; layers 3–45 `IQ2_XXS` + merged activation importance matrix | Keep the aggressively quantized region broad without applying IQ2 to an under-covered layer. |
| Routed expert gate/up, protected layers | layers 0, 1, 46, 47 `Q4_K` | layers 0, 1, 2, 46, 47 `Q4_K` | Protect the first/last stages and the calibration-identified layer-2 tail. |
| Routed expert down, main 512 columns | `Q4_K` | `Q4_K` | Preserve the main expert output path. |
| Routed expert down, 128-column tail | `Q4_0` | `Q4_0` | Separate lower-cost tail enabled by a lossless semantic tensor split. |
| MTP routed experts and most always-active matrices | primarily `Q8_0` | primarily `Q8_0` | Keep speculative decoding and continuously active paths at high precision. |
| Hyper-connection, non-quantizable convolution/vision tensors | `BF16` | `BF16` | Preserve sensitive or shape-incompatible paths. |
| Norms, gates, recurrent/control state | `F32` where required | `F32` where required | Preserve numerical control behavior. |
| Integer PLE controls | `I64` | `I64` | Exact integer representation. |

Each metadata template contains 1,756 tensors. Relative to E5, the Q2_K track uses Q2_K instead of IQ2_XXS for 86 tensors in layers 3–45, while E5 additionally raises the two layer-2 gate/up tensors from Q2_K to Q4_K. The net Q2_K payload is **4,679,270,400 bytes / 4.36 GiB** larger.

## DGX Spark memory target

The provisional 256K-context projection uses:

- approximately 91.74 GiB for the published Q2_K model artifact, or 87.40 GiB for the projected calibrated E5 artifact;
- approximately 6.80 GiB for QSA KV/index state, GDN recurrent/conv state, and a conservative MTP transient allowance;
- 11 GiB reserved for runtime workspace, server/session state, and vision transients;
- an additional 8 GiB `MemAvailable` safety reserve.

Under an assumed 121.63 GiB usable unified-memory budget, that leaves about **4.10 GiB projected headroom for Q2_K** and **8.43 GiB for calibrated E5**. Raising a driver wired-memory limit does not create physical memory, so only measured residency and successful generation on one DGX Spark can pass the final memory gate.

## Required validation gates

Publishing a structurally valid artifact is not the same as declaring it runtime-ready. Current gate status is:

1. **Completed:** generate the complete no-imatrix Q2_K GGUF and verify metadata, tensor shapes, offsets, types, local SHA-256 checksums, and remote LFS objects;
2. **Completed:** independently collect per-layer/per-expert routed activation importance using the official Transformers BF16 implementation, expand under-covered data, record corpus/token hashes, and adapt the precision map to the measured coverage;
3. **Paused after verified template:** generate and verify the complete calibrated E5 GGUF; the incomplete local shard was never uploaded;
4. compare BF16, Q8_0, and mixed-quant outputs on H200, including PLE reset behavior, GDN, QSA, four-branch gated residuals, vision, and MTP;
5. run the ds4 C/CUDA serving path and API smoke tests;
6. measure one-DGX-Spark startup, physical memory, long-context cache growth, and generation quality before declaring the release ready.

## Runtime implementation references

The ds4 implementation will cross-check the official/reference serving paths where useful, especially hybrid GDN/QSA scheduling, four-branch gated residual mixing, PLE lookup/prefetch, fused MoE routing, and MTP:

- [SGLang Qwen3.8-Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- [vLLM Qwen3.8-Flash-Next recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
- [TokenSpeed Qwen3.8-Flash-Next recipe](https://lightseek.org/tokenspeed/recipes/models#qwen3-8-flash-next)
- [Qwen3.8-Flash-Next official repository](https://github.com/QwenLM/Qwen3.8-Flash-Next)

These references describe BF16/FP8 serving engines, not this GGUF format. Their architecture and scheduling implementations are evidence for ds4 design decisions; they do not establish binary compatibility with this release.

## Public mixed-quant references

The working method is informed by the publicly accessible releases in the [DS4-Mixed-Quant-for-Spark collection](https://huggingface.co/collections/Baekpica/ds4-mixed-quant-for-spark-6a79321cc8a55c35231d3ed3):

- [Baekpica/dots3-note-prev-Mixed-Quant-GGUF](https://huggingface.co/Baekpica/dots3-note-prev-Mixed-Quant-GGUF)
- [Baekpica/Motif-3-Mixed-Quant-GGUF](https://huggingface.co/Baekpica/Motif-3-Mixed-Quant-GGUF)
- [Baekpica/Solar-Open2-250B-Mixed-Quant-GGUF](https://huggingface.co/Baekpica/Solar-Open2-250B-Mixed-Quant-GGUF)
- [Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF](https://huggingface.co/Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF)

This Qwen release follows their general workflow—protect always-active, attention, shared-expert, control, and speculative paths; concentrate aggressive quantization in routed expert weights; and reserve unified-memory headroom for an actual server session. Its exact tensor types and validation gates remain model-specific.

## Architecture retained

The target keeps the complete multimodal/speculative topology: 48 text layers (36 GDN + 12 QSA), 512 routed experts with top-10 routing, shared experts, four gated-residual branches, the 51.2B PLE n-gram table, the vision tower, and the one-layer MTP head.

## License

Use of the converted weights is governed by the original **Qwen Community License 1.0**. The exact upstream license file is included in this repository. No Apache-2.0 license is claimed for these weights.
