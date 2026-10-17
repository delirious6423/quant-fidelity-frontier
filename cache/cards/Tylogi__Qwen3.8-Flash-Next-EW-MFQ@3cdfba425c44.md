---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
tags:
- mfq
- quantization
- expert-wise
- mixed-precision
- multimodal
- qwen3_8
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
---

# Qwen3.8-Flash-Next · Expert-wise MFQ

<p style="margin-top: 0; margin-bottom: 8px;">
  <em>Fine-grained mixed precision for <a href="https://huggingface.co/Qwen/Qwen3.8-Flash-Next">Qwen3.8-Flash-Next</a>, with efficient native inference in MFQ.</em>
</p>

<div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
  <a href="https://github.com/Tylogi/MFQ">
    <img src="https://img.shields.io/badge/GitHub-MFQ-172033?logo=github&logoColor=white" alt="MFQ GitHub">
  </a>
  <a href="https://github.com/Tylogi/MFQ#readme">
    <img src="https://img.shields.io/badge/Documentation-English-2477d4" alt="English documentation">
  </a>
  <a href="https://github.com/Tylogi/MFQ/blob/master/README.zh-CN.md">
    <img src="https://img.shields.io/badge/Documentation-中文-2477d4" alt="中文文档">
  </a>
  <a href="#benchmarks">
    <img src="https://img.shields.io/badge/Benchmark-KLD_%26_Top1-e07a2e" alt="KLD and Top1 benchmarks">
  </a>
</div>

<div align="center">
  <a href="https://github.com/Tylogi/MFQ">
    <img src="./assets/tylogi-ai-lab.svg" width="520" alt="Tylogi AI Lab">
  </a>
</div>

## Overview

These are calibrated, mixed-precision releases of Qwen3.8-Flash-Next for [MFQ](https://github.com/Tylogi/MFQ). The model combines a 512-expert MoE with hybrid QSA/GDN attention, vision support and multi-token prediction (MTP). The released checkpoints include the vision encoder, PLE table and MTP head.

The aim is **better quality at a given model size, without giving up efficient inference when precision becomes fine-grained**. Precision is assigned independently to each routed expert's Gate, Up and Down projections, and scalar and vector formats can coexist within the same layer. MFQ's native CUDA and Metal kernels execute these heterogeneous weights directly.

- **Finer than one precision per expert.** Across 48 MoE layers, there are 24,576 routed experts and 73,728 Gate/Up/Down projection matrices. Each projection can use its own format and precision, unlike mainstream quantized-inference libraries that require all experts in a layer to share the same format and precision for a given projection type.
- **A joint choice of format and precision.** The allocator can spend its budget on NINT scalar formats or NVQ vector formats, matching the quantizer to the weight instead of restricting the whole model to one format family.
- **Execution designed for that granularity.** Heterogeneous expert fusion, grouped projection dispatch and format-aware kernels make the fine-grained allocation practical at inference time. The quality and local speed measurements below show both sides of that design.

[Choosing a tier](#choosing-a-tier) · [Quality benchmarks](#benchmarks) · [Local inference speed](#local-inference-speed) · [How it works](#how-it-works) · [Memory planning](#memory-planning) · [Run with MFQ](#run-with-mfq)

## Choosing a tier

| Tier | Positioning | Recommended for |
| :--- | :--- | :--- |
| [V1-S](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-S) | Minimum-footprint compact | Constrained setups where fitting Flash-Next comes first. |
| [V1-M](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-M) | Compact daily-use pick | A lightweight main model with less aggressive compression than V1-S. |
| [S2-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S2-L) | Entry to the middle tiers | Moving beyond the compact releases without committing to a large setup. |
| [V3-XXS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V3-XXS) | Lean high-value all-rounder | An everyday main model with a memory-conscious deployment. |
| [S3-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S3-L) | Middleweight alternative | An intermediate fit before stepping up to the larger-budget recommendations. |
| [V4-XS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V4-XS) | Quality-leaning value pick | More emphasis on fidelity while staying with a balanced, high-value choice. |
| [S4-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S4-L) | Large-budget daily driver | Our balanced main-model recommendation when generous resources are available. |
| [S5-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S5-L) | Fidelity flagship | Resource-rich users who want our highest-fidelity release. |

## Benchmarks

<img src="./assets/Qwen3.8-Flash-Next-EW-MFQ-WT2Benchmark.png" alt="Qwen3.8-Flash-Next: MFQ · EW, Unsloth UD and GSQ-RCO model size versus KLD and Top1 agreement" width="100%">

WikiText-2 evaluation. Lower KLD is better; higher Top1 agreement with the reference model is better.

At approximately 74.5 GB, **V1-M reduces KLD by 37.4% and improves Top1 agreement by 5.03 percentage points** over Unsloth UD IQ1_M in this comparison. The larger tiers progressively trade more storage for closer agreement with the reference model.

| Method | Tier | Weight size (GB) | KLD ↓ | Top1 ↑ |
| :--- | :--- | ---: | ---: | ---: |
| **MFQ · EW** | [V1-S](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-S) | 72.536 | **0.22157** | **83.77%** |
| **MFQ · EW** | [V1-M](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-M) | 74.529 | **0.19689** | **84.72%** |
| **MFQ · EW** | [S2-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S2-L) | 78.859 | **0.14630** | **86.54%** |
| **MFQ · EW** | [V3-XXS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V3-XXS) | 81.952 | **0.12436** | **87.62%** |
| **MFQ · EW** | [S3-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S3-L) | 89.976 | **0.08698** | **89.55%** |
| **MFQ · EW** | [V4-XS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V4-XS) | 93.672 | **0.07524** | **90.36%** |
| [Unsloth UD](https://unsloth.ai/docs/models/qwen3.8-next#quantization-analysis) | IQ1_S | 72.5 | 0.39607 | 77.33% |
| Unsloth UD | IQ1_M | 74.5 | 0.31474 | 79.69% |
| Unsloth UD | Q2_K_XL | 78.9 | 0.22461 | 82.72% |
| Unsloth UD | IQ3_XXS | 82.0 | 0.16512 | 85.41% |
| Unsloth UD | Q3_K_XL | 90.0 | 0.10650 | 88.31% |
| Unsloth UD | IQ4_XS | 93.7 | 0.08363 | 89.55% |
| [GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | Q2_0 | 66.424 | 0.40660 | 77.27% |
| GSQ-RCO | IQ2_XS | 68.026 | 0.33374 | 79.06% |
| GSQ-RCO | IQ3_XXS | 75.840 | 0.22134 | 83.00% |
| GSQ-RCO | IQ3_S | 83.618 | 0.13717 | 86.90% |

<p><sub>For all models shown, weight sizes exclude the vision encoder and MTP head. GB = 10⁹ bytes.</sub></p>

## Local inference speed

### Apple

| Tier | RAM Usage | Device | Input Size | Prompt Processing | Generation (no MTP) | Generation (MTP) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| [V1-M](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-M) | 43.36 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,538.5 tok/s | 71.6 tok/s | 101.2 tok/s |
| [S2-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S2-L) | 47.41 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,538.9 tok/s | 71.0 tok/s | 97.9 tok/s |
| [V3-XXS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V3-XXS) | 50.33 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,527.6 tok/s | 70.8 tok/s | 105.4 tok/s |
| [S3-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S3-L) | 57.95 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,565.6 tok/s | 68.1 tok/s | 98.2 tok/s |
| [V4-XS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V4-XS) | 61.48 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,540.1 tok/s | 67.7 tok/s | 96.7 tok/s |
| [S4-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S4-L) | 78.25 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,481.7 tok/s | 63.6 tok/s | 93.6 tok/s |

### CUDA

| Tier | RAM Usage | Device | Input Size | Prompt Processing | Generation (no MTP) | Generation (MTP) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| [V1-S](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-S) | ≈41.04 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,779.7 tok/s | 62.9 tok/s | 87.0 tok/s |
| [V1-M](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-M) | ≈42.90 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,787.2 tok/s | 63.4 tok/s | 77.5 tok/s |
| [S2-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S2-L) | ≈46.93 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,707.8 tok/s | 58.8 tok/s | 77.8 tok/s |
| [V3-XXS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V3-XXS) | ≈49.81 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,626.2 tok/s | 51.7 tok/s | 61.0 tok/s |
| [S3-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S3-L) | ≈57.28 GiB | 3090Ti + 64GiB RAM | 4,096 tokens | 1,408.5 tok/s | 39.4 tok/s | 42.1 tok/s |
| [V4-XS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V4-XS) | ≈60.72 GiB | 3090Ti + 64GiB RAM | 4,096 tokens | 1,317.1 tok/s | 36.0 tok/s | 36.2 tok/s |

## How it works

### Allocate precision where it matters

Experts differ in sensitivity, and the Gate, Up and Down projections within one expert need not benefit equally from extra bits. The allocation therefore operates at the **expert × projection** level, with a model-wide storage budget.

Our calibration method is open source; see [Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE Quantization](https://arxiv.org/abs/2610.20353). NatScale calibrates the contrast of expert sensitivity weights rather than relying on their ranking alone, and evaluates Gate/Up/Down configurations jointly to estimate expert output error.

1. **Evaluate candidate configurations.** Compare scalar and vector quantizers, including different precisions for the three projections of an expert.
2. **Calibrate their importance.** Combine the estimated errors with calibrated expert sensitivity weights, so an error in a more sensitive expert carries more weight in the allocation.
3. **Allocate the global budget.** Select configurations across the model, spending additional bytes where they provide the most useful error reduction. An expert's three projections do not have to share a format or bit-width.

The heatmaps below use **V3-XXS** as an example.

<img src="./assets/Qwen3.8-Flash-Next-V3-XXS-expert-precision.png" alt="V3-XXS expert precision: Gate/Up/Down average, Gate, Up and Down" width="100%">

The four heatmaps show the Gate/Up/Down average and each projection separately, using the same color scale. Brighter cells use more bits per weight.

### Keep fine-grained inference efficient

Fine-grained allocation is only useful if the runtime can execute it efficiently. MFQ's fused expert execution groups compatible work while preserving each projection's own quantized representation. Gate, Up and Down can use different formats without turning every routed expert into a separate standalone execution path.

**Operator fusion for heterogeneous precision, precision virtualization and JIT specialization** reduce the overhead of these mixed configurations. Grouped dispatch also supports independently quantized attention projections, extending the same approach beyond the expert FFNs. This is the runtime counterpart to the allocator: more freedom to place bits, with kernels designed to use those choices efficiently.

[MFQ](https://github.com/Tylogi/MFQ) provides a broad set of hardware-optimized operators for efficient quantization and inference of fine-grained mixed-precision models across devices, alongside a complete, high-performance serving stack that works out of the box.

## Memory planning

**Download size is not the amount of memory the model needs to keep resident.** The PLE table is a sparse lookup table: MFQ can keep it on SSD and read the required rows on demand, with asynchronous prefetching.

For example, **V1-M downloads 72.44 GiB, including a 29.54 GiB PLE table, while its resident weights occupy 43.36 GiB**. Context caches and runtime buffers use additional memory. The local speed tables report the weight footprint for each tested tier.

If the remaining weights do not fit, MFQ also supports SSD-streamed experts with a configurable resident memory budget. MFQ Studio provides model loading, context and cache controls, resource monitoring, and OpenAI- and Anthropic-compatible serving for local applications and agents.

## Run with MFQ

### Desktop and prebuilt releases

<img src="./assets/mfq-studio-overview.png" alt="MFQ Studio: model serving, live performance and resource overview" width="100%">

Download the package for your platform from [MFQ Releases](https://github.com/Tylogi/MFQ/releases). For desktop use, install **MFQ Studio**; no source build is needed.

In Studio, download your preferred tier from the **Downloads** page and load it from **Models**. Alternatively, download the tier directly from the [files on this model page](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main), keep all of its shards in the same directory, and load the first numbered `.mfq` shard in Studio.

These are `.mfq` checkpoints for MFQ, not GGUF files. The vision encoder and MTP head are already included; no separate projector or draft-model download is needed.

### Build from source

#### 1. Build the runtime

Source builds require Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), CMake 3.26+, and a C++ toolchain. NVIDIA systems also need CUDA Toolkit 12+; Apple silicon uses Metal. Install Node.js/npm to build the Web UI. See the [build guide](https://github.com/Tylogi/MFQ/blob/master/docs/cli/build.md).

```bash
git clone https://github.com/Tylogi/MFQ.git MFQ
cd MFQ
```

NVIDIA / CUDA:

```bash
uv sync
uv run mfq build --backend cuda
```

Apple silicon / Metal:

```bash
uv sync --extra metal
uv run mfq build --backend metal
```

#### 2. Download a tier

For example, download all four `V1-S` shards with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
uvx --from huggingface-hub hf download Tylogi/Qwen3.8-Flash-Next-EW-MFQ \
  --include "V1-S/*.mfq" \
  --local-dir ./models/Qwen3.8-Flash-Next-EW-MFQ
```

Replace `V1-S` with another tier to download its shards.

#### 3. Start the server

Run from the `MFQ` checkout:

```bash
uv run mfq serve \
  --model ./models/Qwen3.8-Flash-Next-EW-MFQ/V1-S/Qwen3.8-Flash-Next-EW-V1-S-00001-of-00004.mfq
```

Open <http://127.0.0.1:8090/> to chat. MFQ automatically loads the remaining shards.

For applications and agents, the default request endpoints are:

- **OpenAI-compatible:** `http://127.0.0.1:8090/v1/chat/completions`
- **Anthropic-compatible:** `http://127.0.0.1:8091/v1/messages`

Configure MTP, memory budgets and context settings in MFQ Studio or the Web UI. For additional deployment options, see the [`mfq serve` guide](https://github.com/Tylogi/MFQ/blob/master/docs/cli/serve.md).

## License

Model weights follow the original [Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
