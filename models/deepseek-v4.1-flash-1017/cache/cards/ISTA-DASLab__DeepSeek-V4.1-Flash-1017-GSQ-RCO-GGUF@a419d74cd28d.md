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

> **Update, October 27, 2026.** The files now use llama.cpp's own DeepSeek-V4.1 layout and load in llama.cpp [b11870](https://github.com/ggml-org/llama.cpp/releases/tag/b11870) or newer, including images.
> - **Downloaded the earlier files?** You do not need to download the model again. Run [convert_ds41_upstream.py](./convert_ds41_upstream.py) on your model folder: it rewrites the files locally into exactly the current ones and fetches only 80 KB. See [Updating an existing download](#updating-an-existing-download).
> - **New download?** The shared Engram file is shard 6 of every build, so link it once per build you use (`<ID>` is `IQ1_M`, `IQ2_XXS` or `IQ2_XS`; see [llama.cpp](#llamacpp)):
>
>   ```sh
>   ln -s DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf DeepSeek-V4.1-Flash-1017-GSQ-RCO-<ID>-00006-of-00006.gguf
>   ```

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

Each build is a backbone GGUF in five shards, named after its quantization identifier: `DeepSeek-V4.1-Flash-1017-GSQ-RCO-<ID>-0000i-of-00006.gguf`, i = 1 to 5. Shard 1 holds the metadata only, shards 2 to 5 the transformer weights, **551.9B parameters**. The bpw is the average bit-width over these, which is what the search controls.

The **Engram file**, `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`, holds the model's two Engram lookup tables (196.6B parameters) at IQ4_NL, 4.50 bpw. It is identical for all builds, so it is stored once and needs to be downloaded only once, also when trying several builds. llama.cpp loads it as shard 6 of a build, so each build needs a link to it named `...-<ID>-00006-of-00006.gguf` (one command, see [llama.cpp](#llamacpp)). The tables are a lookup, not a matmul weight, so they are held at a fixed type in every build and excluded from the search. They are quantized in the evaluated models, not just the shipped ones.

| Build | bpw | Backbone | Engram | Total | Notes |
|---|---|---|---|---|---|
| `IQ1_M` | 1.75 | 120.72 GB | 110.60 GB | 231.32 GB | Smallest |
| `IQ2_XXS` | 2.00 | 138.24 GB | 110.60 GB | 248.84 GB | Recommended; closest to the base model |
| `IQ2_XS` | 2.25 | 155.23 GB | 110.60 GB | 265.83 GB | |

All three builds run on a single 12 GB GPU with [Strata](https://github.com/Niko1221/Strata).

The Engram column is the same shared file in every row. The vision projector, `mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf` (0.97 GB), also serves all builds. GB is 10<sup>9</sup> bytes.

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

> **Runtime support.** llama.cpp supports DeepSeek-V4.1, including images, since release [b11870](https://github.com/ggml-org/llama.cpp/releases/tag/b11870) (October 25, 2026), and the files here use its layout since October 27, 2026. Downloads from before that date can be converted locally, see [Updating an existing download](#updating-an-existing-download). The GGUFs are also supported in [Strata](https://github.com/Niko1221/Strata).

<!-- TBD: confirm that Strata reads the six-shard layout of October 27, 2026. -->

### Strata Engine

[Strata](https://github.com/Niko1221/Strata) runs these GGUFs on a **single NVIDIA GPU with 12 GB of VRAM**, with a one-click installer and an OpenAI-compatible endpoint on localhost. The builds were optimized together with Strata for local inference.

<!-- TBD: Strata command or model-selection steps for this repo. -->

### llama.cpp

Requires llama.cpp [b11870](https://github.com/ggml-org/llama.cpp/releases/tag/b11870) or newer.

```bash
# download one build and the shared Engram file (requires: pip install -U "huggingface_hub[cli]")
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --include "*IQ2_XXS*" "*Engram*" --local-dir .

# the Engram file is shard 6 of every build
ln -s DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf

llama-cli -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf \
  -p "Explain mixed-precision quantization." -ngl 99
```
`-m` takes the first shard; llama.cpp opens shards 2 to 6 from the same directory. To try another build, download only its shards (`--include "*IQ1_M*"`) and add its link. On Windows, `mklink /H <link> DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` makes the link; [convert_ds41_upstream.py](./convert_ds41_upstream.py), run on the folder, also creates missing links.

The Engram table stays memory-mapped on disk and is read one row per token (llama.cpp's default `--lazy-mode auto` does this for tensors over 4 GiB), so it needs neither VRAM nor RAM. See [Memory requirements](#memory-requirements).

### Images

Download the vision projector and pass it with `--mmproj`:

```bash
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --include "mmproj-*" --local-dir .

llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf \
  --mmproj mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf -ngl 99
```
With `llama-mtmd-cli`, also pass `--jinja` so that the chat template embedded in the GGUF is used.

### Chat template and reasoning effort

The base repository ships no Jinja chat template; its prompt format is defined by DeepSeek's encoder, `encoding/encoding.py`. The GGUFs embed a template adapted from [Unsloth's DeepSeek-V4.1 release](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF), checked against that encoder for text chat, tool use, and image formatting. Reasoning effort accepts an integer from 1 to 100 (also a numeric string), or `low`, `high`, `max` (50, 75, 100). The default effort is DeepSeek's `high` (75), and effort applies only in thinking mode. Structured image blocks, including images in tool results, retain their image placeholders. Image inference needs the vision projector, see [Images](#images).

```bash
llama-server -m DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf -ngl 99 \
  --reasoning on --reasoning-effort 50
```
Effort 50 is the setting all results in this card were measured with. Per request, set `"reasoning_effort": "50"` in the chat completion body (a string, also for numbers). `--reasoning off` disables thinking.


### Updating an existing download

Files downloaded before October 27, 2026 (four shards per build and a separate `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`) were written for an earlier llama.cpp branch and do not load in llama.cpp. [convert_ds41_upstream.py](./convert_ds41_upstream.py) turns them into the current files of this repository, byte for byte, without downloading the model again. It needs Python 3.8+ and no extra packages, and fetches only two small files of this repository (`chat_template.jinja` and `exp_probs_b_vl.f32`, 80 KB).

```sh
cd /path/to/downloaded/model-folder
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF convert_ds41_upstream.py --local-dir .
python3 convert_ds41_upstream.py .
```

It converts every build in the folder and the mmproj, and writes the new backbone shards next to the old ones; every written tensor is read back and compared with its source. The Engram file is converted in place: only its header changes, its 110 GB of tables stay as they are, and the script adds the shard-6 links. The free space needed is the size of the new shards (121 to 155 GB per build). With `--delete-old`, the old shards of a build are deleted as soon as its new ones are verified, so space for one build suffices. The new first shards take the chat template from `chat_template.jinja`, so `patch_chat_template.py` is not needed.


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
| `tensor-allocation/DeepSeek-V4.1-Flash-1017-GSQ-RCO-<ID>.rco-allocation.txt` | The quantization type assigned to every tensor of that build (all six shards, 1043 tensors), with a quant-type histogram and the target and measured bit-width. This is the RCO search result, so the allocation can be inspected without opening the model. |
| `convert_ds41_upstream.py`, `chat_template.jinja`, `exp_probs_b_vl.f32` | The converter for downloads from before October 27, 2026 (see [Updating an existing download](#updating-an-existing-download)), the chat template the GGUFs embed, and the vision router bias the converter adds (40 x 384 F32, `ffn.gate.bias_vl` of the original checkpoint). |

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
