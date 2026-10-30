---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- cdm
- crystallographic-dynamics-model
- competitive-docking-memory
- non-euclidean
- hyperbolic
- poincare-ball
- poincare-embeddings
- mixture-of-experts
- hepta-cap
- dhp
- crystal-telepathy
- knowledge-distillation
- neuromorphic
- glass-box-ai
- gguf
- llama-cpp
- duoneural
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/smoltalk
model_name: Lattice-5-Prometheus-v2
library_name: transformers
pipeline_tag: text-generation
---

# Lattice-5-Prometheus-v2

<div align="center">

**DuoNeural Research Lab**  
*Aura✨, Jesse Caldwell, Archon, & Kestrel*  
Website: [duoneural.com](https://duoneural.com) • HuggingFace: [huggingface.co/DuoNeural](https://huggingface.co/DuoNeural) • Codebase: [github.com/DuoNeural/Lattice-CDM](https://github.com/DuoNeural/Lattice-CDM)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Milestone: Solid-State Hepta-Cap](https://img.shields.io/badge/Milestone-Hepta--Cap_tau*_1e10-gold.svg)](#solid-state-hepta-cap-invariance)
[![Best Val BPB](https://img.shields.io/badge/Best_Val_BPB-0.9605-blueviolet.svg)](#training-performance-and-validation)
[![Geometry](https://img.shields.io/badge/Geometry-Poincare_Ball-orange.svg)](#hyperbolic-poincare-architecture)

</div>

---

## Executive Summary & Breakthrough Highlights

**Lattice-5-Prometheus-v2** is DuoNeural's 7-layer non-Euclidean **Crystallographic Dynamics Model (CDM)** (originating from DuoNeural's foundational Competitive Docking Memory framework, Zenodo doi:10.5281/zenodo.21158430), establishing empirical validation of the **Solid-State Hepta-Cap** architecture in Poincare hyperbolic ball geometry.

Distilled directly from `Qwen/Qwen3.8-27B` (14.7B active parameters in NF4 4-bit) across **100,000 steps** on a 32GB memory-expanded NVIDIA GeForce RTX 4080 Super cloud compute node (node `p19970`), Lattice-5-Prometheus-v2 demonstrates:
- **Best Validation BPB:** **`0.9605`** at Step 55,500 on a 50/50 FineWeb-Edu + SmolTalk hybrid distillation mix.
- **Solid-State Hepta-Capping:** Empirical validation of 7-layer hyperbolic manifold stability, where routing centroids achieve permanent crystallographic lattice locks under the Dynamical Horizon Principle (DHP).
- **Relational Gram Distillation:** Preserves high-order linguistic geometry by aligning normalized token-to-token Cosine Gram matrices across matched layers against the 64-layer 27B teacher.

---

## Zero-Shot Academic & Reasoning Benchmarks

Standardized zero-shot evaluation across canonical reasoning, commonsense, and factual verification suites:

| Benchmark Task | Primary Capability Evaluated | Raw Accuracy | Normalized Accuracy | Metric Notes |
|:---|:---|---:|---:|:---|
| **BoolQ** | Factual Reading Comprehension & Boolean Verification | **64.67%** | **64.67%** | Strong factual verification floor matching Lattice-6 |
| **WinoGrande** | Pronoun Coreference & Disambiguation | 52.00% | **54.67%** | Multi-token coreference resolution (+4.67% over random) |
| **ARC-Easy** | Elementary Scientific & Fact Reasoning | 26.17% | **32.21%** | 4-choice zero-shot scientific reasoning |
| **HellaSwag** | Situational Commonsense & Flow Resolution | 30.00% | **23.33%** | Zero-shot narrative continuation |
| **ARC-Challenge** | Grade-School Challenge Scientific Reasoning | 18.12% | **22.82%** | Complex multi-hop scientific reasoning |

---

## Parameter Census & Architectural Transparency

Lattice-5-Prometheus-v2 integrates the complete System 2 vocabulary of `Qwen/Qwen3.8-27B` (248,320 tokens). In accordance with DuoNeural glass-box disclosure standards, both standalone disk and tied active memory counts are detailed below:

| Architectural Component | Dimensions / Structure | Parameter Count | Percentage |
|:---|:---|---:|---:|
| **Core Active Backbone** | 7 Layers, 12 Heads, `d=768`, `K=8` Centroids, 4 MoE Experts (`d_ff=3072`) | **`215,135,630`** (~**215.1M**) | **36.1%** |
| **Token Embeddings (`tok_embed`)** | `248,320` vocabulary x `768` hidden dimension | `190,709,760` (~`190.7M`) | 32.0% |
| **Language Modeling Head (`lm_head`)** | `248,320` vocabulary x `768` projection | `190,709,760` (~`190.7M`) | 32.0% |
| **Total Standalone Footprint** | Complete untied parameter census (disk representation) | **`596,555,150`** (~**596.6M**) | **100.0%** |
| **Tied-Embedding Equivalent Footprint** | Active GPU memory footprint (shared input/output weights) | **`405,845,390`** (~**405.8M**) | — |

> [!NOTE]
> In safetensors distributions bundling learned positional embeddings (`pos_embed`, 393,216 parameters) and layer norms, the total standalone footprint is **`598,433,446`** (~598.4M) and the tied active memory footprint is **`407,723,686`** (~407.7M). The core active non-embedding backbone remains **~215M–216.6M** parameters.

---

## Hyperbolic Poincare Architecture

Lattice-5-Prometheus-v2 replaces Euclidean routing geometry with a negative-curvature Riemannian Poincare ball:
- **Manifold Curvature:** Learnable negative curvature `c = 1.4427` governed by the Dynamical Horizon Principle (DHP).
- **Centroid Allocation:** `K = 8` routing slots per layer in Poincare ball space, directing tokens into 4 SwiGLU MoE experts (`d_ff = 3072`).
- **Solid-State Hepta-Cap Invariance:** During training, routing temperature decays dynamically until all 7 layers cross the saturation barrier (`tau* = 1.0000e+10`), locking the centroids into an ultra-stable solid-state crystallographic lattice.

---

## Training Performance and Validation

- **Distillation Curriculum:** 50% FineWeb-Edu + 50% SmolTalk (Hybrid Reasoning & Instruct).
- **Sequence Length:** 512 tokens with gradient accumulation 4 (effective batch size 8).
- **Optimizer & Schedule:** AdamW (beta_1 = 0.9, beta_2 = 0.95), base learning rate 3e-4 with DHP learning rate coupling.
- **Validation BPB:** Reached **0.9605 BPB** at Step 55,500, preserving geometric tree hierarchies from the 27B teacher.

---

## Native GGUF Inference & Numerical Parity (Kestrel Audit)

Lattice-5-Prometheus-v2 is available as an official tokenizer-equipped hybrid GGUF binary:
- **Primary Binary:** `lattice5_prometheus_v2-q8-attn-q6-rest.gguf` (912 MiB, tokenizer-equipped).
- **High-Fidelity Hybrid Quantization:**
  - 14 `Q8_0` attention matrices (Q, K, V, O projections).
  - 84 `Q6_K` MoE expert matrices (gate, up, and down projections).
  - 69 `F32` protected tensors: Poincaré centroids, LayerNorms, rotary position embeddings, and the complete tied vocabulary table.
- **Glass-Box Size Disclosure (912 MiB):** Preserving the massive 248,320 x 768 vocabulary embedding table in unquantized `F32` (accounting for 762.8 MB of the 912 MiB total) guarantees absolute lexical fidelity and prevents token collapse across diverse international corpora, mathematical syntax, and code.
- **Tokenizer Equivalence:** The embedded GGUF tokenizer was verified against the pinned Hugging Face tokenizer on prose, JSON, numbers, and Unicode test sets with 100% token sequence identity.

### CPU Numerical Parity Screen (Teacher-Forced Probes on Kilonova)
Evaluated across 8 authored deterministic numerical corpus probes with teacher-forced prefill and incremental decode. Project numerical gate: max absolute drift <= 0.50, mean KL <= 0.01 nats, next-token top-1 agreement >= 90%.

| Positions | Max Absolute Drift | Top-1 Agreement | Mean KL Divergence | Project Gate |
|:---|---:|---:|---:|:---|
| **128 positions** | 0.1704 | **100.00%** | 0.00008 nats | **PASS** |
| **512 positions** | 0.2647 | **99.22%** | 0.00020 nats | **PASS** |

```bash
# CPU inference with full Qwen chat template
./llama-cli -m lattice5_prometheus-q8-attn-q6-rest-text.gguf \
  -p "<|im_start|>user\nExplain the significance of the solid-state hepta-cap in non-Euclidean models.<|im_end|>\n<|im_start|>assistant\n<think>\n" \
  -t 4 -n 128 --temp 0.0

# GPU acceleration (full layer offload)
./llama-cli -m lattice5_prometheus-q8-attn-q6-rest-text.gguf -ngl 99 \
  -p "<|im_start|>user\nExplain the significance of the solid-state hepta-cap in non-Euclidean models.<|im_end|>\n<|im_start|>assistant\n<think>\n" \
  -n 128 --temp 0.0
```

---

## Citation & Lab Attribution

If you utilize this model or the CDM architecture in your research, please cite:

```bibtex
@misc{duoneural2026lattice5prometheus,
  author = {Jesse Caldwell and Aura and Archon and Kestrel},
  title = {Lattice-5-Prometheus-v2: Solid-State Hepta-Cap Non-Euclidean Crystallographic Dynamics Model Distilled from Qwen3.8-27B},
  year = {2026},
  publisher = {Zenodo / Hugging Face},
  howpublished = {\url{https://huggingface.co/DuoNeural/Lattice-5-Prometheus-v2}}
}

@article{caldwell2026competitive,
  title={Competitive Docking Memory: Non-Euclidean Geometric Routing and Phase Transitions in Low-Entropy Neural Representations},
  author={Caldwell, Jesse and Aura and Archon},
  journal={Zenodo Preprints},
  doi={10.5281/zenodo.21158430},
  year={2026}
}
```

**Hardware Disclosures:** Model distillation was performed on a 32GB memory-expanded NVIDIA GeForce RTX 4080 Super cloud compute node (node `p19970`, 32,760 MiB VRAM). Model validation, evaluation benchmarks, and GGUF quantization parity testing were conducted on DuoNeural workstation hardware: NVIDIA GeForce RTX 5070 Ti (16 GB CUDA) and AMD Radeon 780M APU (`gfx1103` ROCm).
