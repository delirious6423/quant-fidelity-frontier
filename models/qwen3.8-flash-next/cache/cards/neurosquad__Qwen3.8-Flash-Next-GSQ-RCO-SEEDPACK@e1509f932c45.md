---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
language:
- en
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
tags:
- seedpack
- gguf
- lossless-compression
- qwen3.8
- qwen
- gsq
- rco
- moe
---

<p align="center">
  <a href="https://github.com/glmn-ai/seedpack"><img src="https://huggingface.co/neurosquad/Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK/resolve/main/assets/seedpack-banner.png" alt="SeedPack" width="860"></a>
</p>
<p align="center">
  <img src="https://huggingface.co/neurosquad/Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK/resolve/main/assets/seedpack-stats.png" alt="Qwen3.8-Flash-Next GSQ-RCO: 21 files, 294.82 GB, in one 112.75 GB archive, 2.6× smaller" width="860">
</p>

# Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK

**The complete [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) repository in one 112.75 GB archive instead of 294.82 GB.** That is all four sizes, IQ3_S, IQ3_XXS, IQ2_XS, Q2_0, two shards each, plus the BF16 vision projector, the RCO tensor allocations, the plots and the README. The archive is 38.2% of the original, 2.61× smaller.

Every file is restored **byte for byte**. The SHA-256 of each file matches ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF at revision [`ed59f920`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09); the full list is [below](#contents-of-the-archive).

> **Do you need only one size?** Then download it from [the original repository](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF): one size is 66.42 GB to 83.62 GB, less than this archive. The archive is for keeping, mirroring or moving **all four** sizes: together they cost 1.35× the largest one alone.

## What is SeedPack?

[SeedPack](https://github.com/glmn-ai/seedpack) is a lossless GPU archiver for GGUF model repositories, made by [NeuroSquad](https://huggingface.co/neurosquad). For repositories that ship a BF16 model it grows the quants back from it. This one has no BF16 model, so SeedPack 0.4.0 codes the four sizes against each other (method M30):

- **The same bytes are stored once.** Shard 2 of every size is the same 28.80 GB file: the n-gram embedding table, IQ4_NL. Most dense tensors that two sizes keep at the same type are the same bytes too.
- **The rest are versions of one tensor.** ISTA-DASLab's GSQ quantizes every tensor anew for each size, so even two IQ3_S versions of an expert tensor in two sizes differ. SeedPack sorts the versions of each tensor by size and codes the largest alone (96.0% of its bytes: GSQ's codes are nearly uniform, so there is little to take without a reference).
- **Each later version is coded given the earlier ones.** SeedPack dequantizes them, takes their precision-weighted mean, quantizes it to the next version's type with a bit-exact CUDA port of llama.cpp's quantizers, and codes the published file against that prediction: scales first, then each weight's code given the true scale and the estimate. All later versions together cost 26.5% of their bytes.
- **Big tensors are cut in slices.** The tensors of a 512-expert MoE run to several GB; slices of 64M weights kept packing within 2.7 GB of GPU memory and 5.5 GB of RAM.
- **Everything is verified.** Unpacking checks the BLAKE3 hash of every file and the CRC32 of every chunk.

### Where the bytes go

| Part of the repository | Original | In the archive | |
|---|---:|---:|---|
| The n-gram table (`per_layer_token_embd`, IQ4_NL): shard 2, the same file in all 4 sizes | 115.20 GB | 27.25 GB | stored once; 94.6% of one copy |
| Tensors that are the same bytes in several sizes (most dense tensors of the same type) | 7.49 GB | 0 | stored once |
| The first (largest) version of every other tensor, coded alone | 56.55 GB | 54.28 GB | 96.0% |
| Every later version, coded given the earlier ones | 114.43 GB | 30.34 GB | 26.5% |
| The vision projector (`mmproj-Qwen3.8-Flash-Next-BF16.gguf`) | 905.9 MB | 699.8 MB | 77.2% |
| GGUF headers, small tensors, the allocations, the README, the plots | 239.7 MB | 178.8 MB | 74.6% |
| **Total** | **294.82 GB** | **112.75 GB** | **38.2%** |

The largest groups of later versions:

| Version | Chain starts at | Tensors | Original | In the archive | |
|---|---|---:|---:|---:|---:|
| Q2_0 | IQ4_NL | 103 | 24.30 GB | 6.27 GB | 25.8% |
| IQ2_S | IQ3_S | 50 | 13.43 GB | 2.50 GB | 18.6% |
| Q2_0 | IQ3_S | 44 | 10.38 GB | 3.35 GB | 32.3% |
| IQ2_S | IQ3_XXS | 30 | 8.06 GB | 2.25 GB | 27.9% |
| IQ4_NL | IQ4_NL | 17 | 8.02 GB | 2.05 GB | 25.5% |
| Q2_0 | IQ2_S | 28 | 6.61 GB | 2.79 GB | 42.3% |

## How to use

### 1. Install SeedPack

```bash
pip install "seedpack>=0.4.0"
```

Version 0.4.0 or newer is needed: it is the first with method M30 and the Q2_0 type. Requirements:
- An NVIDIA GPU, Turing (RTX 20xx) or newer, with a CUDA 13 driver (R580 or newer). The model does not have to fit in GPU memory.
- Python 3.11+.
- Windows 10/11 or Linux. Tested on an RTX 5070 (12 GB) under Windows 10. No CUDA Toolkit is needed.
- Disk space for the archive (112.75 GB) plus the files you extract.

### 2. Extract sizes or the whole repository

```bash
seedpack list hf://neurosquad/Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK                                    # what is inside
seedpack x hf://neurosquad/Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK "IQ3_S/*" mmproj-Qwen3.8-Flash-Next-BF16.gguf -o models   # one size and the vision projector
seedpack x hf://neurosquad/Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK -o models                             # the whole repository
seedpack test hf://neurosquad/Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK                                    # decode and verify everything, write nothing
```

`list` fetches only the index; `x` and `test` download the archive once into the Hugging Face cache. A downloaded copy works the same way:

```bash
hf download neurosquad/Qwen3.8-Flash-Next-GSQ-RCO-SEEDPACK Qwen3.8-Flash-Next-GSQ-RCO.seedpack --local-dir .
seedpack x Qwen3.8-Flash-Next-GSQ-RCO.seedpack "IQ3_XXS/*" -o models
```

### 3. Run the model

The extracted files are ISTA-DASLab's GGUFs, so everything in the original card below applies, for example:

```bash
llama-cli -m models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf -lm mmap --lazy-mode on -ngl 99
```

[Strata](https://github.com/Niko1221/Strata) takes a folder of shards you already have: `strata launch --model-path models/IQ3_S`, or `strata models import models/IQ3_S --plan` to see what it would make of it first.

## Speed

Measured on an NVIDIA RTX 5070 (12 GB), an Intel Core i7-8700K (6 cores / 12 threads), 32 GB RAM and Windows 10, with the source files on two SATA SSDs in a stripe (about 1 GB/s) and the archive written to another SATA SSD. Times include process start and file writes.

| Operation | Time | Speed |
|---|---:|---:|
| Pack the whole repository (294.82 GB) | 1753 s | 168 MB/s |
| Test the whole repository (decode and verify, no writes) | 1564 s | 189 MB/s |

Peak GPU memory is 2.7 GB when packing and 2.8 GB when testing; process memory 5.5 GB and 4.1 GB.

Unpacking several files writes them all at once: each tensor's chain is decoded once for every size that holds one of its versions. One size alone needs the versions before its own in each chain, so the smaller sizes take longer to extract than their bytes alone would.

The archive was made with SeedPack 0.4.0.

## Contents of the archive

<details>
<summary>21 files, 294.82 GB. Click to expand.</summary>

| File | Size | SHA-256 |
|---|---:|---|
| `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` | 39.23 GB | `92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7` |
| `IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` | 28.80 GB | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54.82 GB | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28.80 GB | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | 47.04 GB | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` | 28.80 GB | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` | 37.62 GB | `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720` |
| `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf` | 28.80 GB | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907.5 MB | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |
| `.gitattributes` | 2.4 KB | `git blob a1d42795eb1386d78d495d569a6a80b19991909f` |
| `README.md` | 15.7 KB | `git blob 5e40a3fea99859d23fe175a3d2b18065a2621833` |
| `assets/banner.png` | 309.5 KB | `42ce97689e7497dcc46a2fe4701e70ee39ad856f9edffaae3b3220ea5628e076` |
| `assets/plots/Qwen3.8-Flash-Next-aime25_vs_avg_bit_width.png` | 63.3 KB | `git blob b49d69eee85fd9cfa0c66d27e73b3fe1d31870b1` |
| `assets/plots/Qwen3.8-Flash-Next-gpqa_diamond_vs_avg_bit_width.png` | 70.5 KB | `git blob 790a72bc1dba1cd50bdd116a235f01c32c5e2526` |
| `assets/plots/Qwen3.8-Flash-Next-lcb_vs_avg_bit_width.png` | 68.1 KB | `git blob 7d2962dea361ba840e8e00e140cad4aa7559cbc6` |
| `assets/plots/Qwen3.8-Flash-Next-prefill_throughput.png` | 86.3 KB | `git blob ce56b6e105ab007dd3b7f9cc1f0fe9b6547f4fd2` |
| `assets/plots/Qwen3.8-Flash-Next-task_avg_vs_avg_bit_width.png` | 73.7 KB | `git blob e2882a782a5ba0da65a4ba6df7c466af654e805a` |
| `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.rco-allocation.txt` | 39.6 KB | `git blob 7c05839d6316e350643e0375b86dd3bec68fa565` |
| `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt` | 39.4 KB | `git blob 8d567cadd225d8b5133a7ee2ad7ef1c1f01d33ba` |
| `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.rco-allocation.txt` | 39.4 KB | `git blob beb5e7c0bbe56aed2a29121a27baddc1fb89407a` |
| `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt` | 39.1 KB | `git blob 12e14056630b318d14d43ce9570ffcf5490c8152` |

</details>

## License

The weights are [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team, under the Qwen Community License 1.0 ([LICENSE](LICENSE), copied from the base model: Copyright (c) 2026 Qwen). The GGUF quantizations are by the [Deep Algorithms and Systems Lab](https://github.com/IST-DASLab), ISTA, made with GSQ and RCO. SeedPack only packs their files, unchanged.

## Original model card

> The original card of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) follows, unchanged except that its images are linked from that repository. Its `<this-repo>` is ISTA-DASLab's repository.

---


<!--
  GSQ-RCO GGUF release card for Qwen3.8-Flash-Next.
  Mirrors the Qwen3.8-27B card.
  NB: the YAML block must remain the very first bytes of the file (HF requirement).
-->

<div align="center">

<a href="https://github.com/IST-DASLab"><img src="https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/assets/banner.png" alt="GGUF, GSQ-RCO dynamic non-uniform quantization" width="100%"/></a>

<br/>

# Qwen3.8-Flash-Next &middot; GSQ-RCO GGUFs

**Non-uniform GGUF quantizations** of a 512-expert MoE, produced with **GSQ** and **RCO**, with a vision projector for multimodal use.

[![arXiv: GSQ](https://img.shields.io/badge/arXiv-GSQ_2604.18556-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18556)
[![arXiv: RCO](https://img.shields.io/badge/arXiv-RCO_2605.00649-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.00649)
[![GSQ code](https://img.shields.io/badge/code-GSQ-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/GSQ)
[![RCO code](https://img.shields.io/badge/code-RCO-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/RCO)
[![DASLab](https://img.shields.io/badge/DASLab-GitHub-101048?logo=github&logoColor=white)](https://github.com/IST-DASLab)
[![license](https://img.shields.io/badge/license-apache--2.0-19a34a)](#license)

</div>

![Task average vs bit-width](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/assets/plots/Qwen3.8-Flash-Next-task_avg_vs_avg_bit_width.png)

---

## Overview

This repository provides GGUF quantizations of **Qwen3.8-Flash-Next** at four sizes, together with the model's vision projector (`mmproj`) for multimodal use. In contrast to uniform quantization, which applies a single quantization type to all weight tensors, each model here assigns a separate quantization type to every tensor. The assignment is obtained by a gradient-based search that allocates precision according to per-tensor sensitivity, subject to a total size budget. The resulting files are standard GGUF and run unmodified in `llama.cpp`, Ollama, and LM Studio.

A code-specialized variant is released separately as [Qwen3.8-Flash-Next-GSQ-RCO-Coder](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF), which retains 256 of the 512 routed experts per layer at an effective 1.89 bpw over the original parameters, for a 58.4 GB total.

Flash-Next is a sparse mixture-of-experts model: **512 routed experts per layer across 48 layers**, of which 10 are active per token. The routed expert matrices dominate the parameter count, so they also dominate the size budget: 95% of the searchable weights live in the experts. The search therefore spends most of its freedom there, assigning a quantization type **per layer** (not per expert, which GGUF cannot express).

> **Method summary.** GSQ provides accurate low-bit scalar quantization of each tensor at a given quantization type; RCO assigns the per-tensor quantization types under a size budget. Together they yield a non-uniform GGUF at the requested size.

| Method | Description |
|---|---|
| **GSQ** (Gumbel-Softmax Quantization, [paper](https://arxiv.org/abs/2604.18556), [code](https://github.com/IST-DASLab/GSQ)) | Post-training scalar quantization that jointly learns the per-coordinate grid assignments and the per-group scales via a Gumbel-Softmax relaxation. GSQ closes most of the gap between scalar and vector quantization at 2 to 3 bits while remaining deployable in standard scalar formats such as GGUF. |
| **RCO** (Riemannian Constrained Optimization, [paper](https://arxiv.org/abs/2605.00649), [code](https://github.com/IST-DASLab/RCO)) | Assigns one of K quantization types to each of N tensors under a total size budget. The budget constraint is reformulated as a smooth Riemannian manifold in logit space, which permits gradient-based optimization directly on the task loss while enforcing the budget exactly, without constraint-specific hyperparameter tuning. |

Both methods were developed at the [Deep Algorithms and Systems Lab (DASLab)](https://github.com/IST-DASLab), Institute of Science and Technology Austria.

---

## Available files

Files follow the convention **`<model>-GSQ-RCO-<type>.gguf`**, where the suffix names the quantization class; the table lists each file's average bit-width over the transformer weights.

The `mmproj` file carries the vision encoder and projector at BF16; one copy serves all quantizations. Each quantized model ships as **two shards**. The second shard is identical across all four variants: it holds the per-layer n-gram embedding table (`per_layer_token_embd`, 51.2B parameters) at IQ4_NL. That table is a lookup, not a matmul weight, so it is held at a fixed 4.5 bpw in every build and excluded from the search. It is quantized in the evaluated models, not just the shipped ones.

| Directory | bpw | Shard 1 (weights) | Shard 2 (n-gram) | Total | Notes |
|---|---|---|---|---|---|
| `Q2_0/` | 2.40 | 37.6 GB | 28.8 GB | **66.4 GB** | Fastest; avoids lookup-table formats |
| `IQ2_XS/` | 2.50 | 39.2 GB | 28.8 GB | **68.0 GB** | Smallest at equal quality |
| `IQ3_XXS/` | 3.00 | 47.0 GB | 28.8 GB | **75.8 GB** | Matches the base model on AIME25 |
| `IQ3_S/` | 3.50 | 54.8 GB | 28.8 GB | **83.6 GB** | Recommended; matches or exceeds the base model on every task |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 16 | 0.91 GB | n/a | **0.91 GB** | Vision encoder + projector, for multimodal use |

**Q2_0** is built for speed. It avoids the quantization formats that rely on large lookup tables: those formats pack more accuracy into a given bit-width, but decoding them costs real time, and on this model that cost dominates inference. Q2_0 delivers **3.4x the prompt throughput** and **1.9x lower end-to-end latency** than IQ2_XS at a slightly smaller file size, and its decode rate stays flat across workloads instead of varying with the content. The trade is a little quality: 89.07 task average against 89.16 for IQ2_XS, and 3.5 points below IQ3_XXS. Pick it when throughput matters most, and see [Performance](#performance) for the measurements.

The IQ3_S model is the strongest operating point: it matches the base model exactly on AIME25 (100.00), exceeds it on GPQA-Diamond (92.93 against 91.92) and is within 0.57 points on LiveCodeBench v6, at under a quarter of the BF16 size. IQ3_XXS remains the better choice when memory is tight: it gives up 1.52 points on GPQA-Diamond and 0.57 on LiveCodeBench v6 for 7.8 GB less.

### Memory requirements

Only the first shard holds transformer weights and needs to be resident. The second shard is the n-gram lookup table: it is read sparsely, one row per token, so it can stay memory-mapped on disk and does not need to occupy VRAM. Run with `-lm mmap --lazy-mode on` to get this behaviour, plan for **shard 1 in VRAM**, and keep shard 2 on an SSD so the paging stays cheap.

| Model | Must be resident (shard 1) | Can stay on disk (shard 2) | Total download |
|---|---|---|---|
| `Q2_0` | 37.6 GB | 28.8 GB | 66.4 GB |
| `IQ2_XS` | 39.2 GB | 28.8 GB | 68.0 GB |
| `IQ3_XXS` | 47.0 GB | 28.8 GB | 75.8 GB |
| `IQ3_S` | 54.8 GB | 28.8 GB | 83.6 GB |

Add headroom for the KV cache and the vision projector (0.91 GB) if used. Keeping the n-gram table in RAM rather than on disk removes the paging cost entirely and is worth it where memory allows.

---

## Performance

Measured with `llama.cpp` over 55 prompts spanning eleven categories (coding, humanities, math, QA, RAG, reasoning, STEM, writing, multilingual, summarization, roleplay).

| Model | Prompt t/s | Decode t/s | Avg latency |
|---|---|---|---|
| `Q2_0` | **367.49** | **93.79** | **6.70 s** |
| `IQ2_XS` | 108.19 | 70.30 | 12.68 s |

![Prefill throughput by category](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/assets/plots/Qwen3.8-Flash-Next-prefill_throughput.png)

The gap comes from the quantization formats rather than the size: the two files differ by only 1.6 GB, and Q2_0 is the smaller of the two. Q2_0's decode rate is also far more stable across categories, 92.5 to 94.5 t/s (a 2.0 t/s spread), where IQ2_XS swings from 49.4 to 94.5 t/s (a 45.1 t/s spread), because the lookup cost varies with how much of each layer a request touches.

The prefill advantage is uneven, and largest where prompts are long: 9.6x on RAG, 6.8x on writing and 6.2x on coding. On short-prompt, reasoning-heavy categories the two are level or Q2_0 is marginally behind (0.8x to 0.9x on stem, math and reasoning), where prefill is too brief for the format's decoding cost to matter.

---

## Results

All models are evaluated against the **BF16** base model. We report the average over five zero-shot tasks (arc_easy, arc_challenge, hellaswag, winogrande, piqa), recovery (zero-shot average relative to BF16), and three reasoning and generation benchmarks: **AIME25**, **GPQA-Diamond**, and **LiveCodeBench v6**. Task average is the mean of the three. Sizes are the full download, including the n-gram shard; the bit-width column is the average over the transformer weights, which is what the search controls.

| Variant | bpw | GB | ZS avg↑ | recovery | AIME25↑ | GPQA-D↑ | LCB v6↑ | Task avg↑ |
|---|---|---|---|---|---|---|---|---|
| BF16 | 16.00 | 354 | 76.94 | 100.0% | 100.00 | 91.92 | 87.43 | 93.12 |
| **GSQ-RCO Q2_0** | 2.40 | 66.4 | **78.00** | **101.4%** | 96.67 | 89.39 | 81.14 | 89.07 |
| **GSQ-RCO IQ2_XS** | 2.50 | 68.0 | 77.16 | 100.3% | 96.67 | 87.37 | 83.43 | 89.16 |
| **GSQ-RCO IQ3_XXS** | 3.00 | 75.8 | 77.23 | 100.4% | **100.00** | 91.41 | 86.29 | 92.57 |
| **GSQ-RCO IQ3_S** | 3.50 | 83.6 | n/a | n/a | **100.00** | **92.93** | **86.86** | **93.26** |

All three models exceed the BF16 base on the zero-shot average (100.3% to 101.4%), which is common for careful low-bit quantization on these five tasks: the differences are within roughly one standard error and should be read as parity rather than improvement. The same applies where IQ3_S exceeds the base on GPQA-Diamond and on the task average: recoveries slightly above 100% reflect benchmark variance, not a model that is better than the one it was quantized from. The reasoning and generation benchmarks separate the models far more clearly.

At 3.00 bpw, IQ3_XXS reaches **99.4%** of the base model's task average (92.57 against 93.12) while being **4.7x smaller** (75.8 GB against 354 GB). It matches the base exactly on AIME25 (100.00), and trails by 0.51 on GPQA-Diamond and 1.14 on LiveCodeBench v6. At 2.50 bpw, IQ2_XS retains 95.7% of the task average at 68.0 GB, and the smallest build, Q2_0, holds 96.67 on AIME25 at 66.4 GB, a 5.3x reduction.

The two sub-2.5 bpw builds land within 0.1 points of each other on task average despite different allocations: Q2_0 is stronger on GPQA-Diamond (89.39 against 87.37) while IQ2_XS is stronger on LiveCodeBench v6 (83.43 against 81.14).

![AIME25 vs bit-width](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/assets/plots/Qwen3.8-Flash-Next-aime25_vs_avg_bit_width.png)

![GPQA-Diamond vs bit-width](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/assets/plots/Qwen3.8-Flash-Next-gpqa_diamond_vs_avg_bit_width.png)

![LiveCodeBench v6 vs bit-width](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/assets/plots/Qwen3.8-Flash-Next-lcb_vs_avg_bit_width.png)

---

## Usage

Both shards must be downloaded; `llama.cpp` loads the split automatically when given the first one.

### llama.cpp
```bash
# download (requires: pip install -U "huggingface_hub[cli]")
hf download <this-repo> --include "IQ3_XXS/*" --local-dir .

llama-cli -m IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  -lm mmap --lazy-mode on \
  -p "Explain mixed-precision quantization." -ngl 99
```
`-lm mmap --lazy-mode on` keeps the n-gram table memory-mapped on disk instead of loading it into memory, which removes 28.8 GB from the resident footprint. See [Memory requirements](#memory-requirements).

### Strata Engine

Many users report strong results with [Strata](https://github.com/Niko1221/Strata), a third-party engine that runs these GGUFs on a single NVIDIA GPU from 8 GB of VRAM, with a one-click installer and an OpenAI-compatible endpoint on localhost.

### Vision (multimodal)
```bash
hf download <this-repo> mmproj-Qwen3.8-Flash-Next-BF16.gguf --local-dir .

llama-mtmd-cli -m IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-BF16.gguf \
  -lm mmap --lazy-mode on \
  --image photo.jpg -p "Describe this image."
```
The projector was converted directly from the base checkpoint and is shared by all four quantizations.

### Ollama
```bash
ollama run hf.co/<this-repo>   # pick the directory matching your memory budget
```

### LM Studio
Search the repo name, then pick a `GSQ-RCO-*` build from the file list.

---

## Quantization procedure

1. **Per-tensor database.** Each weight tensor is quantized at every candidate GGUF quantization type with GSQ, yielding a searchable database of quantized tensor variants.
2. **RCO search.** The budget-constrained Riemannian search assigns one quantization type per tensor such that the whole-file average bit-width meets the target.
3. **Assembly.** The selected per-tensor variants are stitched into a single standard GGUF file.

For this model the search covers **352 tensors**: the 304 dense tensors (attention, SSM projections, shared experts, embeddings, output head) and the 48 fused routed-expert matrices, one per layer. Two classes draw on separate quantization ladders, because the routed experts sit at a much lower bit-width than the dense path at any given budget. `ffn_down_exps` searches over a much smaller ladder than the other expert matrices. Its 640 rows are not divisible by 256, which rules out every block-256 K and I quant format and leaves only Q2_0 (block 64) and IQ4_NL (block 32). In the two smaller builds every layer lands on Q2_0; in IQ3_XXS the search lifts 18 of the 48 layers to IQ4_NL.

Reference implementations: **GSQ** at [IST-DASLab/GSQ](https://github.com/IST-DASLab/GSQ) and **RCO** at [IST-DASLab/RCO](https://github.com/IST-DASLab/RCO).

### Reproducibility artifacts

Each released GGUF ships the files needed to audit how it was built:

| File | Contents |
|---|---|
| `tensor-allocation/<model>.rco-allocation.txt` | The quantization type assigned to every tensor in that file, with a quant-type histogram and the target bit-width. This is the RCO search result, so the allocation can be inspected without opening the model. |

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

We thank [Verda](https://verda.com/) and Scientific Computing at the Institute of Science and Technology Austria for providing the compute resources used to produce these models.

---

## License

These quantized weights inherit the license of the base model (**Qwen3.8-Flash-Next**). The GSQ-RCO tooling is released by the Deep Algorithms and Systems Lab under its repository license.

<div align="center">
<sub>Built with <b>GSQ</b> and <b>RCO</b> at the <a href="https://github.com/IST-DASLab">Deep Algorithms and Systems Lab</a> &middot; Institute of Science and Technology Austria</sub>
</div>
