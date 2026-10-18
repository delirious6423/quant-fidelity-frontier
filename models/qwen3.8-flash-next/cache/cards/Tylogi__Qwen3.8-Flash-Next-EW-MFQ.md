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
    <img src="https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/resolve/main/assets/tylogi-ai-lab.svg" width="520" alt="Tylogi AI Lab">
  </a>
</div>

## Overview

**Every Bit. Maximum Fidelity.**

These are calibrated, mixed-precision releases of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for
[MFQ](https://github.com/Tylogi/MFQ). The model combines a 512-expert MoE with
hybrid QSA/GDN attention, vision support and multi-token prediction (MTP).
The released checkpoints include the vision encoder, PLE table and MTP head.

MFQ combines fine-grained mixed-precision calibration, compact weight formats
and native CUDA/Metal runtimes to preserve model quality within local memory,
storage and latency budgets. Use MFQ Studio for downloads, loading and
quantization inspection, or connect applications and agents through OpenAI-
and Anthropic-compatible APIs.

[Results](#results-at-a-glance) · [Choosing a tier](#choosing-a-tier) ·
[Why MFQ](#why-mfq) · [Quality tables](#benchmarks) ·
[Local inference speed](#local-inference-speed) · [Memory planning](#memory-planning) ·
[Run with MFQ](#run-with-mfq) · [Citation](#citation)

## Results at a glance

All results below use **Qwen3.8-Flash-Next**, with the tested tier and conditions
shown for each result.

| What matters | Measured result | Configuration |
| --- | --- | --- |
| Quantization quality | **37.4% lower KLD**; Top1 agreement **up 5.03 percentage points** | V1-M vs. Unsloth UD IQ1_M on WikiText-2, at approximately 74.5 GB weight size |
| Weight residency | **43.36 GiB resident weights** | V1-M on Metal, with the PLE table read from SSD |
| Apple / Metal generation | **101.2 tok/s** | V1-M; M5 Max, 40-core GPU, 128 GB; 4,096-token input; MTP enabled |
| NVIDIA / CUDA generation | **87.0 tok/s** | V1-S; RTX 3090 Ti + 32 GiB RAM; 4,096-token input; MTP enabled |

Weight residency excludes KV cache and runtime allocations. The quality plot
shows model size against agreement with the reference model: lower KLD and
higher Top1 agreement are better.

<p align="center">
  <img src="https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/resolve/main/assets/qwen38-flash-next-quality.svg" alt="Qwen3.8-Flash-Next: MFQ, Unsloth UD and GSQ-RCO model size versus KLD and Top1 agreement" width="1100">
</p>

Weight sizes in the quality comparison exclude the vision encoder and MTP
head; GB = 10⁹ bytes. The [quality table](#benchmarks) lists all plotted points.

## Choosing a tier

| Tier | Positioning | Recommended for |
| :--- | :--- | :--- |
| [V1-S](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-S) | Minimum-footprint compact | Constrained setups where fitting Flash-Next comes first. |
| [V1-M](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-M) | Compact daily-use pick | A lightweight main model with less aggressive compression than V1-S. |
| [S2-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S2-L) | Entry to the middle tiers | Moving up from the compact releases without committing to a large setup. |
| [V3-XXS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V3-XXS) | Lean high-value all-rounder | An everyday main model with a memory-conscious deployment. |
| [S3-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S3-L) | Middleweight alternative | An intermediate fit before stepping up to the larger-budget recommendations. |
| [V4-XS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V4-XS) | Quality-leaning value pick | More emphasis on fidelity while staying with a balanced, high-value choice. |
| [S4-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S4-L) | Large-budget daily driver | Our balanced main-model recommendation when generous resources are available. |
| [S5-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S5-L) | Fidelity flagship | Resource-rich users who want our highest-fidelity release. |

<a id="how-it-works"></a>

## Why MFQ

### Preserve quality with fine-grained precision

MFQ assigns precision to individual experts, projections, chunks, or neurons,
retaining higher precision where the model needs it most. NINT scalar and NVQ
vector formats can coexist within the same layer, with independent choices for
each expert's Gate, Up and Down projections. The native runtimes execute these
mixed formats directly.

MFQ's base formats also reduce reconstruction error. Across the seven listed
MFQ/GGUF format pairs, mean SSE is **4.04%–25.16% lower** on the tested layer-0
expert matrices. The figure shows the actual bpw for each pair.

For **Qwen3.8-Flash-Next**, the packed expert tensors in the Unsloth UD
`IQ3_XXS` recipe select one GGUF format for each layer's Gate, Up and Down
projections. MFQ `V3-XXS` selects a profile for **each expert projection**:
**1,536 allocation slots per layer**, compared with **3** in that recipe.

<p align="center">
  <img src="https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/resolve/main/assets/qwen38-flash-next-allocation-comparison.svg" alt="Qwen3.8-Flash-Next layer 0: GGUF versus MFQ expert-projection precision allocation, plus seven base-format comparisons with actual bpw and 4.04%–25.16% lower mean reconstruction SSE for MFQ" width="1100">
</p>

The allocation panels show **Qwen3.8-Flash-Next, layer 0, experts 0–11**. Colors identify
the selected profiles. Expert 9 uses **NVQ3J-L for Gate and Up** and **NINT4 for
Down**. The **512×** ratio compares allocation slots per layer for MFQ and this
GGUF recipe; GGUF block-level quantization parameters can still vary. See the
[allocation data and sources](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/raw/main/assets/data/qwen38-flash-next-layer0-allocation.json).

The lower panel's SSE is the equally weighted mean across **1,536 matrices in
layer 0**: all 512 experts' Gate, Up and Down projections. Actual bpw differs
within each format pair. See the [format benchmark notes](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/blob/main/assets/benchmarks/qwen38-flash-next-base-format-sse.md)
for the complete SSE table and the [format data](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/raw/main/assets/data/qwen38-flash-next-base-format-sse.json)
for the supplied values and their source.

<details>
<summary>Full-model precision heatmaps</summary>

These **Qwen3.8-Flash-Next V3-XXS** heatmaps show the overall expert average,
followed by Gate, Up and Down, on a shared color scale.

<p align="center">
  <img src="https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/resolve/main/assets/qwen38-flash-next-expert-precision.png" alt="Qwen3.8-Flash-Next V3-XXS: average expert precision and separate Gate, Up and Down heatmaps" width="1100">
</p>

</details>

Our calibration method is open source; see
[Beyond Expert Rankings: Contrast Calibration for Mixed-Precision MoE Quantization](https://arxiv.org/abs/2610.20353).
For BibTeX, see [Citation](#citation).

<details>
<summary>How calibration chooses profiles</summary>

1. **Evaluate candidate configurations.** Compare scalar and vector quantizers, including different precisions for the three projections of an expert.
2. **Calibrate their importance.** Combine the estimated errors with calibrated expert sensitivity weights, so an error in a more sensitive expert carries more weight in the allocation.
3. **Allocate the global budget.** Select configurations across the model, spending additional bytes where they provide the most useful error reduction. An expert's three projections do not have to share a format or bit-width.

</details>

## Benchmarks

WikiText-2 evaluation: lower KLD is better; higher Top1 agreement with the
reference model is better.

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

[Model-quality data](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/raw/main/assets/data/qwen38-flash-next-quality.json).

## Local inference speed

The fine-grained **V3-XXS** tier reaches **105.4 tok/s on M5 Max** and
**61.0 tok/s on RTX 3090 Ti + 32 GiB RAM**, with MTP enabled.

The chart compares MFQ, Strata, llama.cpp and oMLX on the same **M5 Max (40-core GPU,
128 GB)**, using **Qwen3.8-Flash-Next** and a **4,096-token Python-code input**.
Each runtime's tested quantization tiers are shown separately.

<p align="center">
  <img src="https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/resolve/main/assets/qwen38-flash-next-metal-runtime-comparison.svg" alt="Qwen3.8-Flash-Next on M5 Max: MFQ, Strata, llama.cpp and oMLX prompt processing and generation with MTP off and on, across 17 runtime/checkpoint configurations" width="1100">
</p>

Shading and bold labels mark **MFQ V3-XXS (82.0 GB)** and the selected
comparison tiers: **GSQ-RCO IQ3_S (83.6 GB)** and **Jundot oQ3.5e (about 93.9 GB)**.
IQ3_S has a similar weight size to V3-XXS; oQ3.5e is larger.
These weight sizes exclude the vision encoder and MTP head.
With MTP enabled, V3-XXS reaches **105.4 tok/s**, while IQ3_S reaches
**99.6 tok/s** in Strata and **59.3 tok/s** in llama.cpp.
oQ3.5e reaches **111.0 tok/s** in oMLX with adaptive MTP.

The MFQ source gives one prefill value per tier; all other prefill bars use
MTP-off measurements. oMLX uses the official macOS 26/27 DMG with native NAX
kernels and adaptive MTP; its prefill timing includes first-token overhead.
Quantization tiers, draft configurations and repeat counts differ. See the
[benchmark notes](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/blob/main/assets/benchmarks/qwen38-flash-next-metal-comparison.md)
for complete conditions and tables, and the
[comparison data](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/raw/main/assets/data/qwen38-flash-next-metal-runtime-comparison.json)
for all published values and their sources.
The MFQ tables below give the complete published Metal and CUDA results.

MFQ's Apple Metal backend is built on [MLX](https://github.com/ml-explore/mlx),
with a native C++ runtime and custom Metal kernels for MFQ quantization formats.

CUDA combines VRAM, system RAM and SSD to run models that exceed GPU memory
capacity. Metal supports bounded expert residency and SSD streaming for MoE
experts and PLE tables. CUDA and Metal kernels operate on the packed weights,
with optimized attention, caching and I/O.

Feature availability depends on the model and backend; see the
[runtime support matrix](https://github.com/Tylogi/MFQ/blob/2782c7ee3ed2d8ad6351040739a3f340f0e77601/docs/runtime/support.md). ROCm/HIP support is in
development, and optimized CPU backends are planned.

### Apple / Metal

| Tier | Resident weights | Device | Input Size | Prompt Processing | Generation (no MTP) | Generation (MTP) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| [V1-M](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-M) | 43.36 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,538.5 tok/s | 71.6 tok/s | 101.2 tok/s |
| [S2-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S2-L) | 47.41 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,538.9 tok/s | 71.0 tok/s | 97.9 tok/s |
| [V3-XXS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V3-XXS) | 50.33 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,527.6 tok/s | 70.8 tok/s | 105.4 tok/s |
| [S3-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S3-L) | 57.95 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,565.6 tok/s | 68.1 tok/s | 98.2 tok/s |
| [V4-XS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V4-XS) | 61.48 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,540.1 tok/s | 67.7 tok/s | 96.7 tok/s |
| [S4-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S4-L) | 78.25 GiB | Apple M5 Max · 40-core GPU · 128 GB | 4,096 tokens | 2,481.7 tok/s | 63.6 tok/s | 93.6 tok/s |

### NVIDIA / CUDA

| Tier | Resident weights | Device | Input Size | Prompt Processing | Generation (no MTP) | Generation (MTP) |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| [V1-S](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-S) | ≈41.04 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,779.7 tok/s | 62.9 tok/s | 87.0 tok/s |
| [V1-M](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V1-M) | ≈42.90 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,787.2 tok/s | 63.4 tok/s | 77.5 tok/s |
| [S2-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S2-L) | ≈46.93 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,707.8 tok/s | 58.8 tok/s | 77.8 tok/s |
| [V3-XXS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V3-XXS) | ≈49.81 GiB | 3090Ti + 32GiB RAM | 4,096 tokens | 1,626.2 tok/s | 51.7 tok/s | 61.0 tok/s |
| [S3-L](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/S3-L) | ≈57.28 GiB | 3090Ti + 64GiB RAM | 4,096 tokens | 1,408.5 tok/s | 39.4 tok/s | 42.1 tok/s |
| [V4-XS](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main/V4-XS) | ≈60.72 GiB | 3090Ti + 64GiB RAM | 4,096 tokens | 1,317.1 tok/s | 36.0 tok/s | 36.2 tok/s |

The PLE table stays on SSD; it does not need to occupy resident memory.
For example, V1-M downloads 72.44 GiB, including 29.54 GiB of PLE, while its
resident weights occupy 43.36 GiB.

The tables report resident weights; KV cache and runtime allocations are
additional. [Native speed data](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/raw/main/assets/data/qwen38-flash-next-inference-speed.json).

## Memory planning

**Download size is not the amount of memory the model needs to keep resident.** The PLE table is a sparse lookup table: MFQ can keep it on SSD and read the required rows on demand, with asynchronous prefetching.

For example, **V1-M downloads 72.44 GiB, including a 29.54 GiB PLE table, while its resident weights occupy 43.36 GiB**. Context caches and runtime buffers use additional memory. The local speed tables report the weight footprint for each tested tier.

If the remaining weights do not fit, MFQ also supports SSD-streamed experts with a configurable resident memory budget. MFQ Studio provides model loading, context and cache controls, resource monitoring, and OpenAI- and Anthropic-compatible serving for local applications and agents.

## Run with MFQ

### Desktop and prebuilt releases

Download the package for your platform from [MFQ Releases](https://github.com/Tylogi/MFQ/releases). For desktop use, install **MFQ Studio**; no source build is needed.

In Studio, download your preferred tier from the **Downloads** page and load it from **Models**. Alternatively, download the tier directly from the [files on this model page](https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/tree/main), keep all of its shards in the same directory, and load the first numbered `.mfq` shard in Studio.

These are `.mfq` checkpoints for MFQ, not GGUF files. The vision encoder and MTP head are already included; no separate projector or draft-model download is needed.

**Serving overview**

<p align="center">
  <img src="https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/resolve/main/assets/mfq-studio-overview-en.png" alt="MFQ Studio: local model serving, live performance and resource overview" width="1100">
</p>

**Quantization analysis**

MFQ Studio's built-in quantization visualization shows checkpoint architecture,
model structure and precision allocation across layers, experts and Gate, Up
and Down projections.

<p align="center">
  <img src="https://huggingface.co/Tylogi/Qwen3.8-Flash-Next-EW-MFQ/resolve/main/assets/mfq-studio-analysis-en.png" alt="MFQ Studio Analysis: checkpoint architecture, model structure and projection-level precision details" width="1100">
</p>

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

## Citation

If you use MFQ in your research, please cite our
[paper on mixed-precision MoE calibration](https://arxiv.org/abs/2610.20353):

```bibtex
@misc{xie2026contrastcalibration,
  title         = {Beyond Expert Rankings: Contrast Calibration for Mixed-Precision {MoE} Quantization},
  author        = {Xie, Renjie and Yang, Juncheng and Hu, Aoting and Liang, Yan and
                   Zhang, Mingxi and Hong, Zheheng and Xu, Wei and Chen, Yanzhi},
  year          = {2026},
  eprint        = {2610.20353},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.20353}
}
```

## Benchmark sources

Figures, benchmark snapshots and Studio screenshots are synchronized from
[MFQ revision `9378aeea`](https://github.com/Tylogi/MFQ/commit/9378aeea7a6d44a78b0c8b2975ebaf8f1b1cf4c2).
The linked JSON snapshots and benchmark notes record the underlying values,
measurement conditions and source revisions. This model-card update adds no
new measurements.

## License

Model weights follow the original [Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
