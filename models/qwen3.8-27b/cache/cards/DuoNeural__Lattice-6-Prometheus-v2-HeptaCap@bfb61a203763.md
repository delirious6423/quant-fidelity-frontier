---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- cdm
- non-euclidean
- poincare-embeddings
- mixture-of-experts
- hepta-cap
- neuromorphic
- duoneural
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/smoltalk
model_name: Lattice-6-Prometheus-v2-HeptaCap
library_name: transformers
pipeline_tag: text-generation
---

# Lattice-6-Prometheus-v2-HeptaCap

<div align="center">

**DuoNeural Research Lab**  
*Aura✨, Jesse Caldwell, & Archon*  
Website: [duoneural.com](https://duoneural.com) • HuggingFace: [huggingface.co/DuoNeural](https://huggingface.co/DuoNeural) • Codebase: [github.com/DuoNeural/Lattice-CDM](https://github.com/DuoNeural/Lattice-CDM)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Milestone: Solid-State Hepta-Cap](https://img.shields.io/badge/Milestone-Hepta--Cap_tau*_1e10-gold.svg)](#the-hepta-cap-milestone)
[![Best Val BPB](https://img.shields.io/badge/Best_Val_BPB-0.6731-blueviolet.svg)](#training-performance)
[![Geometry](https://img.shields.io/badge/Geometry-Poincare_Ball-orange.svg)](#hyperbolic-architecture)

</div>

---

## Executive Summary & Breakthrough Highlights

**Lattice-6-Prometheus-v2-HeptaCap** is DuoNeural's flagship 7-layer non-Euclidean Crystallographic Dynamics Model (CDM), achieving a historic milestone by demonstrating complete **Solid-State Hepta-Capping** (`tau* = 1.0000e+10` across all 7 layers).

Distilled from `Qwen/Qwen3.8-27B` (14.7B active parameters in NF4 4-bit) across **100,000 steps** on a single workstation NVIDIA GeForce RTX 4080 Super (32GB VRAM), Lattice-6-Prometheus-v2 achieved:
- **Best Validation BPB:** **`0.6731`** at Step 84,000 (Validation Perplexity ~ `9.02`)
- **Teacher KL Divergence:** **`0.1232` nats**
- **Complete Hepta-Cap Invariance:** All 7 layers (`L00` through `L06`) crossed the Dynamical Horizon Principle (DHP) saturation boundary at `tau* = 1.0000e+10`, freezing the routing centroids into permanent, low-entropy hyperbolic Voronoi lattices.
- **Relational Gram Distillation:** Aligned normalized token-to-token Cosine Gram matrices across 7 matched layers between the 64-layer 27B teacher and the 7-layer Poincaré student, transferring geometric semantic curvature with zero auxiliary projection overhead.

---

## Zero-Shot Academic & Reasoning Benchmarks

Evaluated across canonical reasoning, commonsense, and factual verification suites:

| Benchmark Task | Primary Capability Evaluated | Raw Accuracy | Normalized Accuracy | Metric Notes |
|:---|:---|---:|---:|:---|
| **BoolQ** | Factual Reading Comprehension & Boolean Verification | **64.67%** | **64.67%** | Strong factual verification floor |
| **WinoGrande** | Pronoun Coreference & Disambiguation | 54.67% | **56.00%** | Exceeds random baseline by +6.0% |
| **ARC-Easy** | Elementary Scientific & Fact Reasoning | 38.26% | **42.28%** | 4-choice zero-shot scientific reasoning |
| **HellaSwag** | Situational Commonsense & Flow Resolution | 33.33% | **34.00%** | Zero-shot narrative continuation |
| **ARC-Challenge** | Grade-School Challenge Scientific Reasoning | **24.83%** | 24.16% | Complex multi-hop scientific reasoning |

---

## Parameter Census & Architectural Transparency

Lattice-6-Prometheus-v2 incorporates the complete System 2 vocabulary of `Qwen/Qwen3.8-27B` (`248,320` tokens). In accordance with DuoNeural's glass-box disclosure standards, both core active and vocabulary parameter counts are detailed below:

| Architectural Component | Dimensions / Structure | Parameter Count | Percentage |
|:---|:---|---:|---:|
| **Core Active Backbone** | 7 Layers, 12 Heads, `d=768`, `K=8` Centroids, 4 MoE Experts (`d_ff=3072`) | **`216,619,174`** (~**216.6M**) | **36.2%** |
| **Token Embeddings (`tok_embed`)** | `248,320` vocabulary x `768` hidden dimension | `190,709,760` (~`190.7M`) | 31.9% |
| **Language Modeling Head (`lm_head`)** | `248,320` vocabulary x `768` projection | `190,709,760` (~`190.7M`) | 31.9% |
| **Total Standalone Footprint** | Complete untied parameter census | **`598,433,446`** (~**598.4M**) | **100.0%** |
| **Tied-Embedding Equivalent Footprint** | Shared input/output weight representation | **`407,328,934`** (~**407.3M**) | — |

---

## Mechanistic Slot Logit Lens: The `<think>` Emergence

Inspection of Poincaré routing centroids reveals how deep layers specialize during crystallization:

- **Layer 00 (c = 0.4531):** Structural syntax, indentation, and punctuation (`,`, `\n`, `.`, `<|im_end|>`, ` and`).
- **Layer 01 (c = 0.4245):** Logical conjunctions and sentence connectives (` and`, `,`, `.`, ` or`).
- **Layer 02 (c = 0.3921):** **Deep Reasoning Boundary:** Slot 4 locks directly onto `<think>` and paragraph delimiters (`\n`, `<|im_end|>`, `\n\n`, `<think>`).
- **Layer 03 (c = 0.3910):** Step-by-step reasoning tokens (`\n`, `<|im_end|>`, `<think>`, `.`, ` the`).
- **Layer 04 (c = 0.3933):** Determiner and entity anchoring (` the`, `the`, ` a`, ` The`).
- **Layer 05 (c = 0.4865):** Adversarial logic transitions (` And`, ` and`, `,`, `\n`, ` But`).
- **Layer 06 (c = 0.5507):** Syntactic closure and final token emit (`\n`, `,`, ` the`, ` and`, ` `).

---

---

## Native GGUF Inference with llama.cpp

Lattice-6-Prometheus-v2-HeptaCap is fully supported in patched `llama.cpp` for native non-Euclidean execution on CPU and CUDA:

- `lattice6_prometheus-q8-attn-q6-rest-text.gguf` (334 MiB): High-fidelity hybrid quantization (`Q8_0` attention + `Q6_K` MoE experts + `F32` Poincaré routing/norms). Passes aggregate numerical error gates at both 128 and 512 context positions.
- `lattice6_prometheus-f32-text.gguf` (1.63 GiB): Full-precision reference model (100% top-1 parity across all context modes).

### Running with llama-cli

```bash
# CPU inference with full Qwen3.8 chat template
./llama-cli -m lattice6_prometheus-q8-attn-q6-rest-text.gguf \
  -p "<|im_start|>user\nExplain the significance of the solid-state hepta-cap in non-Euclidean models.<|im_end|>\n<|im_start|>assistant\n<think>\n" \
  -t 4 -n 128 --temp 0.0

# GPU acceleration (full layer offload)
./llama-cli -m lattice6_prometheus-q8-attn-q6-rest-text.gguf -ngl 99 \
  -p "<|im_start|>user\nExplain the significance of the solid-state hepta-cap in non-Euclidean models.<|im_end|>\n<|im_start|>assistant\n<think>\n" \
  -n 128 --temp 0.0
```

## Citation & Lab Attribution

```bibtex
@misc{duoneural2026lattice6prometheus,
  title={Lattice-6-Prometheus-v2: Solid-State Hepta-Cap Non-Euclidean Crystallographic Dynamics Model Distilled from Qwen3.8-27B},
  author={Aura and Caldwell, Jesse and Archon},
  institution={DuoNeural Research Lab},
  year={2026},
  url={https://huggingface.co/DuoNeural/Lattice-6-Prometheus-v2-HeptaCap}
}
```
