---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
license: other
license_name: qwen-community-1.0
license_link: LICENSE
tags:
  - gguf
  - gsq
  - rco
  - quantization
  - mixed-precision
  - multimodal
  - vision
---

> **TL;DR:** Run Qwen's 180B model on 2× RTX 3090s, with embeddings on disk and the KV cache in RAM. With BF16 embeddings, it scores 2.85 points above the base model on MMLU-Pro, with 1.72% higher perplexity.

<div align="center">

# Qwen3.8-Flash-Next &middot; GSQ-RCO GGUFs

**Non-uniform GGUF quantization** produced with **GSQ** and **RCO**, with a vision projector for multimodal use.

[![arXiv: GSQ](https://img.shields.io/badge/arXiv-GSQ_2604.18556-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18556)
[![arXiv: RCO](https://img.shields.io/badge/arXiv-RCO_2605.00649-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.00649)
[![GSQ code](https://img.shields.io/badge/code-GSQ-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/GSQ)
[![RCO code](https://img.shields.io/badge/code-RCO-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/RCO)
[![DASLab](https://img.shields.io/badge/DASLab-GitHub-101048?logo=github&logoColor=white)](https://github.com/IST-DASLab)
[![license](https://img.shields.io/badge/license-Qwen_Community_1.0-19a34a)](#license)

</div>

> Independent community reproduction. These files were produced by a third party using the published GSQ and RCO methods. They are not an IST-DASLab release and carry no endorsement from the authors of either paper.

![MMLU-Pro: GSQ-RCO 3.5-bit and BF16 reference](assets/plots/qwen-mmlu-updated.png)

---

## Overview

This repository provides a GGUF quantization of **[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** at a 3.5-bit target, together with the model's BF16 vision projector (`mmproj`) for multimodal use. Each weight tensor is assigned its own quantization type by a search that allocates precision by per-tensor sensitivity under an exact total size budget.

This is a reduced-budget adaptation of GSQ/RCO to native GGUF formats. The model uses a separate shard for the token and n-gram embeddings. Choose the BF16 shard or the smaller Q4_0/Q8_0 shard.

| Method | Description |
|---|---|
| **GSQ** (Gumbel-Softmax Quantization, [paper](https://arxiv.org/abs/2604.18556), [code](https://github.com/IST-DASLab/GSQ)) | Post-training scalar quantization that jointly learns per-coordinate grid assignments and per-group scales through a Gumbel-Softmax relaxation. |
| **RCO** (Riemannian Constrained Optimization, [paper](https://arxiv.org/abs/2605.00649), [code](https://github.com/IST-DASLab/RCO)) | Assigns one of K quantization types to each of N tensors under an exact total size budget, reformulated as a smooth Riemannian manifold in logit space. |

Both methods were developed at the [Deep Algorithms and Systems Lab (DASLab)](https://github.com/IST-DASLab), Institute of Science and Technology Austria.

---

## Read this before using these files

**Download the main weights and one embedding shard.** The `3.5bit` filename retains the original target for the complete model with Q4_0/Q8_0 embeddings. The complete model averages 3.500 bits per weight with those embeddings, or 6.855 with BF16 embeddings, including file headers. File identities and the BF16 shard layout are recorded in [manifest.json](manifest.json); the quantized alternative is recorded in [embeddings-Q4_0.json](embeddings-Q4_0.json).

---

## Available files

| File | File bpw | Size | Notes |
|---|---:|---:|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf` | 3.065 | 47.94 GB | Main weights; requires embedding shard |
| `Qwen3.8-Flash-Next-ngram-embeddings.gguf` | 16.002 | 103.68 GB | BF16 token and n-gram embeddings |
| `Qwen3.8-Flash-Next-ngram-embeddings-Q4_0.gguf` | 4.551 | 29.49 GB | Q4_0 n-gram table + Q8_0 token embeddings |
| `Qwen3.8-Flash-Next-mmproj-BF16.gguf` | 16.173 | 0.91 GB | Vision encoder + projector |

---

## MMLU-Pro

The evaluations below use the BF16 embedding shard.

**No reasoning; 2,048-token context limit.** Scores use the log probabilities of single-token answers ` A` through ` J`, with zero-shot prompts and no chat template. The 2,000 questions use a fixed seed and are stratified across all 14 categories. Quant and BF16 use matching layer placement. Scorer, question IDs, per-question predictions and runtime settings are in the [evaluation package](eval/2026-09-13/README.md).

| Build | Correct | Accuracy | SE |
|---|---:|---:|---:|
| GSQ-RCO 3.5-bit | 1,165/2,000 | **58.25%** | 1.10 pp |
| BF16 reference | 1,108/2,000 | **55.40%** | 1.11 pp |

The paired difference is +2.85 percentage points (226 quant-only / 169 BF16-only correct; exact two-sided McNemar p = 0.0047752).

---

## Results

Perplexity: eight 1,024-token contexts, 4,088 scored tokens, document-disjoint held-out text. Greedy decoding; IFEval answered directly; GSM8K with xhigh thinking; 1,024/2,048-token caps.

| Variant | Native PPL ↓ | Approx. KL/token ↓ | IFEval strict | IFEval completed correct | GSM8K completed correct |
|---|---:|---:|---:|---:|---:|
| BF16 reference | **3.0533** | — | 13/16 | 13/16 | 8/8 |
| **GSQ-RCO** | 3.1058 | 0.111225 | 13/16 | 12/16 | 8/8 |

KL is approximate, computed from the native uint16 reference cache. One quant IFEval response was truncated.

[Raw results](eval/2026-09-13/README.md) · [Reproducibility](eval/2026-09-13/reproducibility/README.md)

---

## Usage

### Runtime

`llama.cpp` with Qwen4Exp and Q2_0 support, pinned here to upstream [`f3f1a8f`](https://github.com/ggml-org/llama.cpp/commit/f3f1a8f2760f28325a5ec20c05b171e5b7c83a29). The runtime and numerical settings used for evaluation are documented separately in the [reproduction instructions](eval/2026-09-13/reproducibility/README.md#rebuild-the-runtime).

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout f3f1a8f2760f28325a5ec20c05b171e5b7c83a29
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

### Text and vision

For quantized embeddings, replace `Qwen3.8-Flash-Next-ngram-embeddings.gguf` with `Qwen3.8-Flash-Next-ngram-embeddings-Q4_0.gguf` in the download and symlink commands below. The main weights and quantized embeddings total 77.42 GB.

```bash
hf download pfeifferj/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf \
  Qwen3.8-Flash-Next-ngram-embeddings.gguf \
  Qwen3.8-Flash-Next-mmproj-BF16.gguf --local-dir .

ln -s Qwen3.8-Flash-Next-GSQ-RCO-3.5bit.gguf qwen3.8-flash-next-00001-of-00002.gguf
ln -s Qwen3.8-Flash-Next-ngram-embeddings.gguf qwen3.8-flash-next-00002-of-00002.gguf

llama-server -m qwen3.8-flash-next-00001-of-00002.gguf \
  --mmproj Qwen3.8-Flash-Next-mmproj-BF16.gguf \
  --override-tensor '^per_layer_token_embd\.weight$=CPU,^token_embd\.weight$=CPU' \
  --load-mode mmap -ngl all -c 4096 -np 1 --jinja
```

The numbered symlinks let `llama.cpp` discover both shards. The tensor overrides keep both embedding matrices on CPU, with mmap allowing their pages to be backed by disk.

For two 24 GB cards such as [RTX 3090s](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/), balance the weights with `--tensor-split` and leave headroom for the KV cache and runtime buffers. `--no-kv-offload` places the KV cache in system RAM when needed.

---

## Quantization procedure

1. Per-tensor candidates. GSQ trained routed expert projections for 8 steps on 64 training and 4 validation sequences of 1024 tokens; rare or unimproved experts keep their initializer.
2. RCO search. 8 steps, 2 Gumbel samples, 8 training and 4 validation sequences, selected by validation teacher KL. No step improved on step 0, so the initial allocation was kept.
3. Assembly. Exact packed bytes, metadata and alignment charged; payload roundtrip verified.

Gate/up candidates Q2_0/Q2_K/Q3_K/Q4_K; down candidates Q2_0/Q4_0 (640-column rows do not fit 256-value K blocks); compatible non-expert matrices Q8_0. The BF16 embedding option retains the source bytes. The Q4_0/Q8_0 option copies the packed embedding tensors from the original quantized release without requantization.

![Quantization type allocation across the expert modules](https://huggingface.co/pfeifferj/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/assets/plots/q38fn-allocation.png)

Source revision: [`de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540).

---

## Citation

If you use these files, please cite this release together with the base model and both methods.

**This release**

```bibtex
@misc{q38fngsqrco2026,
  title        = {Qwen3.8-Flash-Next GSQ-RCO GGUF quantization},
  author       = {Josephine Pfeiffer},
  year         = {2026},
  publisher    = {Hugging Face},
  doi          = {10.57967/hf/10398},
  howpublished = {\url{https://huggingface.co/pfeifferj/Qwen3.8-Flash-Next-GSQ-RCO-GGUF}}
}
```

**Base model**

```bibtex
@techreport{qwen2026design,
    title       = {On the Design of {Qwen3.8-Next} Architecture: Evaluation, Efficiency, and Training Stability},
    author      = {{Qwen Team}},
    institution = {Alibaba Group},
    month       = {August},
    year        = {2026}
}

@misc{qwen3.8flashnext,
    title  = {{Qwen3.8-Flash-Next}: A New Architecture, Towards Ultimate Cost-Efficiency},
    author = {{Qwen Team}},
    month  = {August},
    year   = {2026},
    url    = {https://qwen.ai/blog?id=qwen3.8-flash-next}
}
```

**Methods**

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

Huge kudos to the [Deep Algorithms and Systems Lab (DASLab)](https://github.com/IST-DASLab) at the Institute of Science and Technology Austria for developing GSQ and RCO and for releasing the papers and reference implementations publicly.

Thanks also to ella for giving me the idea and supporting me at every step of the process, and Chris van Hoof (Red Hat) for contributing the compute on which these quantizations were produced and evaluated.

---

## License

These quantized weights inherit the license of the base model, **[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, the Qwen Community License 1.0; the original license text ships as `LICENSE`. The GSQ and RCO tooling is released by DASLab under its own repository licenses.
