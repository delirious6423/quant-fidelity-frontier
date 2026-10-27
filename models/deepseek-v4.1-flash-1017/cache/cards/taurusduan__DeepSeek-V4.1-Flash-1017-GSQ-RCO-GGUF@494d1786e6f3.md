---
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
license: mit
tags:
  - gguf
  - gsq
  - rco
  - quantization
  - mixed-precision
  - ist-daslab
  - moe
  - deepseek
---

<div align="center">

<a href="https://github.com/IST-DASLab"><img src="assets/banner.webp" alt="DeepSeek-V4.1-Flash-1017, GSQ-RCO dynamic non-uniform GGUF" width="100%"/></a>

<br/>

# DeepSeek-V4.1-Flash-1017 &middot; GSQ-RCO GGUFs

**Non-uniform GGUF quantizations** of a 763B-parameter, 384-expert MoE, produced with **GSQ** and **RCO**.

[![arXiv: GSQ](https://img.shields.io/badge/arXiv-GSQ_2604.18556-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18556)
[![arXiv: RCO](https://img.shields.io/badge/arXiv-RCO_2605.00649-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.00649)
[![GSQ code](https://img.shields.io/badge/code-GSQ-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/GSQ)
[![RCO code](https://img.shields.io/badge/code-RCO-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/RCO)
[![DASLab](https://img.shields.io/badge/DASLab-GitHub-101048?logo=github&logoColor=white)](https://github.com/IST-DASLab)
[![license](https://img.shields.io/badge/license-MIT-19a34a)](#license)

</div>

![Base model vs GSQ-RCO quants on AIME25, GPQA-Diamond and LiveCodeBench v6](assets/plots/DeepSeek-V4.1-Flash-1017-benchmarks_bars.png)

---

## Overview

This repository provides GGUF quantizations of **DeepSeek-V4.1-Flash-1017** at three sizes: **IQ1_M (1.75 bits per weight), IQ2_XXS (2.00) and IQ2_XS (2.25)**. In contrast to uniform quantization, which applies a single quantization type to all weight tensors, each model here assigns a separate quantization type to every tensor. The assignment is obtained by a gradient-based search that allocates precision according to per-tensor sensitivity, subject to a total size budget. The resulting files are standard GGUF.

DeepSeek-V4.1-Flash-1017 is a sparse mixture-of-experts model with **763.2B parameters**: **384 routed experts per layer**, of which 6 are active per token, and a 1M-token context window. The routed experts dominate the parameter count, so they also dominate the size budget, and the search spends most of its freedom there, assigning a quantization type **per layer** (not per expert, which GGUF cannot express).

With [Strata](https://github.com/Niko1221/Strata), every build runs on a **single GPU with 12 GB of VRAM**. See [Usage](#usage).

> **Method summary.** GSQ provides accurate low-bit scalar quantization of each tensor at a given quantization type; RCO assigns the per-tensor quantization types under a size budget. Together they yield a non-uniform GGUF at the requested size.

| Method | Description |
|---|---|
| **GSQ** (Gumbel-Softmax Quantization, [paper](https://arxiv.org/abs/2604.18556), [code](https://github.com/IST-DASLab/GSQ)) | Post-training scalar quantization that jointly learns the per-coordinate grid assignments and the per-group scales via a Gumbel-Softmax relaxation. GSQ closes most of the gap between scalar and vector quantization at 2 to 3 bits while remaining deployable in standard scalar formats such as GGUF. |
| **RCO** (Riemannian Constrained Optimization, [paper](https://arxiv.org/abs/2605.00649), [code](https://github.com/IST-DASLab/RCO)) | Assigns one of K quantization types to each of N tensors under a total size budget. The budget constraint is reformulated as a smooth Riemannian manifold in logit space, which permits gradient-based optimization directly on the task loss while enforcing the budget exactly, without constraint-specific hyperparameter tuning. |

Both methods were developed at the [Deep Algorithms and Systems Lab (DASLab)](https://github.com/IST-DASLab), Institute of Science and Technology Austria.

---

## Available files

Each build is a backbone GGUF in four shards, named after its quantization identifier: `DeepSeek-V4.1-Flash-1017-GSQ-RCO-<ID>-0000i-of-00004.gguf`. The backbone holds the transformer weights, **551.9B parameters**, and its bpw is the average bit-width over these, which is what the search controls.

The **Engram file**, `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, holds the model's two Engram lookup tables (196.6B parameters) at IQ4_NL, 4.50 bpw. It is identical for all builds, so it is stored once and needs to be downloaded only once, also when trying several builds. The tables are a lookup, not a matmul weight, so they are held at a fixed type in every build and excluded from the search. They are quantized in the evaluated models, not just the shipped ones. Keep the Engram file in the same directory as the backbone shards: it is loaded from there.

| Build | bpw | Backbone | Engram | Total | Notes |
|---|---|---|---|---|---|
| `IQ1_M` | 1.75 | 120.72 GB | 110.60 GB | 231.32 GB | Smallest |
| `IQ2_XXS` | 2.00 | 138.24 GB | 110.60 GB | 248.84 GB | Recommended; closest to the base model |
| `IQ2_XS` | 2.25 | 155.23 GB | 110.60 GB | 265.83 GB | |

All three builds run on a single 12 GB GPU with [Strata](https://github.com/Niko1221/Strata).

The Engram column is the same shared file in every row. GB is 10<sup>9</sup> bytes.

The IQ2_XXS build is the stronger operating point: it matches the base model on all three benchmarks and so reaches **100.0%** of its task average (93.54). The IQ1_M build trades 1.2 points of task average for 17.5 GB less.

### Memory requirements

<!-- TBD: confirm how Strata places the model on a 12 GB GPU (which tensors stay in VRAM, how much system RAM it needs) and fill in this section. -->

Only the backbone holds transformer weights. The Engram tables are read sparsely, one row per token, so they can stay memory-mapped on disk and do not need to occupy VRAM or RAM. With Strata, the backbone does not need to fit in VRAM either: every build runs on a **12 GB GPU**, with the rest of the model held in system memory. Keep the Engram file on an SSD so the paging stays cheap.

---

## Results

All models are evaluated against the base model **deepseek-ai/DeepSeek-V4.1-Flash-1017** as released (FP8 dense weights, FP4 routed experts). We report three reasoning and generation benchmarks: **AIME25**, **GPQA-Diamond**, and **LiveCodeBench v6**. Task average is the mean of the three; recovery is the task average relative to the base model.

All numbers in this card were measured at **reasoning effort 50**.

| Variant | bpw | AIME25↑ | GPQA-D↑ | LCB v6↑ | Task avg↑ | Recovery |
|---|---|---|---|---|---|---|
| Base (FP8 / FP4) | n/a | 100.00 | 90.91 | 89.71 | 93.54 | 100.0% |
| **GSQ-RCO IQ1_M** | 1.75 | **100.00** | 88.38 | 88.57 | 92.32 | 98.7% |
| **GSQ-RCO IQ2_XXS** | 2.00 | **100.00** | **90.91** | **89.71** | **93.54** | **100.0%** |
| **GSQ-RCO IQ2_XS** | 2.25 | **100.00** | 90.40 | **89.71** | 93.37 | 99.8% |

IQ2_XXS (2.00 bpw) keeps **100.0%** of the base model's task average: it matches the base model on AIME25 (100.00), GPQA-Diamond (90.91) and LiveCodeBench v6 (89.71). IQ2_XS (2.25 bpw) matches it on AIME25 and LiveCodeBench v6 and scores 90.40 on GPQA-Diamond, 99.8% overall. The smallest build, IQ1_M (1.75 bpw), also scores 100.00 on AIME25 and retains 98.7% overall.

---

## Usage

> **Runtime support.** As of October 23, 2026, llama.cpp does not support DeepSeek-V4.1, so these GGUFs do not run in llama.cpp yet. They are supported in [Strata](https://github.com/Niko1221/Strata).

### Strata Engine

[Strata](https://github.com/Niko1221/Strata) runs these GGUFs on a **single NVIDIA GPU with 12 GB of VRAM**, with a one-click installer and an OpenAI-compatible endpoint on localhost. The builds were optimized together with Strata for local inference.

<!-- TBD: Strata command or model-selection steps for this repo. -->

### llama.cpp

<!-- TBD: which public llama.cpp runs these files. Upstream master has no deepseek41 support, and PR #30817
     (draft) stores the Engram tables inside the model GGUF, a different format from the separate Engram file
     here. The flags below exist in the build these files were tested with. -->

The commands below apply once llama.cpp supports DeepSeek-V4.1 (see [Runtime support](#usage)).

```bash
# download one build and the shared Engram file (requires: pip install -U "huggingface_hub[cli]")
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --include "*IQ2_XXS*" "*Engram*" --local-dir .

llama-cli -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf \
  -lm mmap --lazy-mode on \
  -p "Explain mixed-precision quantization." -ngl 99
```
`-m` takes the first shard; llama.cpp opens the other three shards and the Engram file from the same directory. To try another build, download only its shards (`--include "*IQ1_M*"`). `-lm mmap --lazy-mode on` keeps the Engram tables memory-mapped on disk instead of loading them into memory. See [Memory requirements](#memory-requirements).

### Chat template and reasoning effort

The base repository ships no Jinja chat template; its prompt format is defined by DeepSeek's encoder, `encoding/encoding.py`. The GGUFs embed a template adapted from [Unsloth's DeepSeek-V4.1 release](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF), checked against that encoder for text chat, tool use, and image formatting. Reasoning effort accepts an integer from 1 to 100 (also a numeric string), or `low`, `high`, `max` (50, 75, 100). The default effort is DeepSeek's `high` (75), and effort applies only in thinking mode. Structured image blocks, including images in tool results, retain their image placeholders. Actual image inference requires a matching vision projector (mmproj) and a runtime that supports DeepSeek-V4.1 vision.

```bash
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf -ngl 99 \
  --reasoning on --reasoning-effort 50
```
Effort 50 is the setting all results in this card were measured with. Per request, set `"reasoning_effort": "50"` in the chat completion body (a string, also for numbers). `--reasoning off` disables thinking.


### Updating an existing download

Older downloads can be updated locally with [patch_chat_template.py](./patch_chat_template.py), without downloading the model shards again. The script embeds the new template and uses Python 3.9+ with no extra packages or network access.

Stop the application using the model, download the script, then run:

```sh
python3 patch_chat_template.py /path/to/downloaded/model-folder
```

It patches matching first shards **in place**, using a small recovery journal (about **30 MiB** of extra working space by default), rather than creating another full shard. If interrupted, keep the recovery directory and rerun the same command to resume. Already updated files are skipped. See [PATCHING.md](./PATCHING.md) for dry-run mode, smaller recovery blocks, and linked-file handling.


---

## Quantization procedure

1. **Per-tensor database.** Each weight tensor is quantized at every candidate GGUF quantization type with GSQ, yielding a searchable database of quantized tensor variants.
2. **RCO search.** The budget-constrained Riemannian search assigns one quantization type per tensor such that the backbone's average bit-width meets the target.
3. **Assembly.** The selected per-tensor variants are stitched into a standard GGUF file.

The dense tensors (attention, shared experts, embeddings, output head) and the routed-expert matrices draw on separate quantization ladders, because the routed experts sit at a much lower bit-width than the dense path at any given budget: the dense tensors choose among Q5_K, Q6_K and Q8_0, the routed experts among types from IQ1_S to IQ3_S. The search's teacher is the released checkpoint itself, with the routed experts in their native FP4, and its calibration data was generated by the model. <!-- TBD: per-build expert types and the searched tensor count. -->

Reference implementations: **GSQ** at [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ) and **RCO** at [IST-DASLab/RCO](https://github.com/IST-DASLab/RCO).

### Reproducibility artifacts

Each released build ships the files needed to audit how it was built:

| File | Contents |
|---|---|
| `tensor-allocation/DeepSeek-V4.1-Flash-1017-GSQ-RCO-<ID>.rco-allocation.txt` | The quantization type assigned to every tensor of that build (all four shards, 1008 tensors), with a quant-type histogram and the target and measured bit-width. This is the RCO search result, so the allocation can be inspected without opening the model. |

---

## Citation

If you use these models or methods, please cite both papers:

```bibtex
@article{gsq2026,
  title  = {GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling},
  author = {Dadgarnia, Alireza and Tabesh, Soroush and Nikdan, Mahdi and Helcig, Michael and Kurtic, Eldar and Kleinegger, Maximilian and Alistarh, Dan},
  journal= {arXiv preprint arXiv:2604.18556},
  year   = {2026}
}
@article{rco2026,
  title  = {Model Compression with Exact Budget Constraints via Riemannian Manifolds},
  author = {Helcig, Michael and Alistarh, Dan},
  journal= {arXiv preprint arXiv:2605.00649},
  year   = {2026}
}
```

---

## Acknowledgements

<!-- TBD: confirm the compute acknowledgement for this release (copied from the Flash-Next card). -->

We thank [Verda](https://verda.com/) and Scientific Computing at the Institute of Science and Technology Austria for providing the compute resources used to produce these models.

---

## License

These quantized weights inherit the license of the base model (**DeepSeek-V4.1-Flash-1017**, MIT). The GSQ-RCO tooling is released by the Deep Algorithms and Systems Lab under its repository license.

<div align="center">
<sub>Built with <b>GSQ</b> and <b>RCO</b> at the <a href="https://github.com/IST-DASLab">Deep Algorithms and Systems Lab</a> &middot; Institute of Science and Technology Austria</sub>
</div>
