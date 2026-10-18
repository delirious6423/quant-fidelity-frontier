---
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
license: mit
tags:
  - gguf
  - gsq
  - rco
  - quantization
  - mixed-precision
  - multimodal
  - vision
---

> **TL;DR:** DeepSeek's 552B-parameter backbone plus 196B-parameter Engram memory, reduced from the original 510.3 GB FP8/FP4 checkpoint to a 426.1 GB package including the DSpark draft model and vision projector.

<div align="center">

# DeepSeek-V4.1-Flash &middot; GSQ-RCO GGUF

**Non-uniform GGUF quantization** produced with **GSQ** and **RCO**, with the DSpark draft model and the vision projector.

[![arXiv: GSQ](https://img.shields.io/badge/arXiv-GSQ_2604.18556-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18556)
[![arXiv: RCO](https://img.shields.io/badge/arXiv-RCO_2605.00649-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.00649)
[![GSQ code](https://img.shields.io/badge/code-GSQ-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/GSQ)
[![RCO code](https://img.shields.io/badge/code-RCO-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/RCO)
[![DASLab](https://img.shields.io/badge/DASLab-GitHub-101048?logo=github&logoColor=white)](https://github.com/IST-DASLab)
[![license](https://img.shields.io/badge/license-mit-19a34a)](#license)

</div>

> Independent community reproduction. These files were produced by a third party using the published GSQ and RCO methods. They are not an IST-DASLab release and carry no endorsement from the authors of either paper.

---

## Overview

This repository provides a GGUF quantization of **[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** at 3.0 bits per weight for the backbone, together with the DSpark speculative-decoding draft and the BF16 vision projector (`mmproj`). RCO assigns quantization types to eligible backbone projections according to their sensitivity, within an exact total size budget. The Engram conditional-memory tables use Q8_0 and are excluded from the search.

| Method | Description |
|---|---|
| **GSQ** (Gumbel-Softmax Quantization, [paper](https://arxiv.org/abs/2604.18556), [code](https://github.com/IST-DASLab/GSQ)) | Post-training scalar quantization that jointly learns per-coordinate grid assignments and per-group scales through a Gumbel-Softmax relaxation. |
| **RCO** (Riemannian Constrained Optimization, [paper](https://arxiv.org/abs/2605.00649), [code](https://github.com/IST-DASLab/RCO)) | Assigns one of K quantization types to each of N tensors under an exact total size budget, reformulated as a smooth Riemannian manifold in logit space. |

Both methods were developed at the [Deep Algorithms and Systems Lab (DASLab)](https://github.com/IST-DASLab), Institute of Science and Technology Austria.

---

## Read this before using these files

The file is 415.76 GB. Engram (208.90 GB at Q8_0) is intended for host memory (`-ot 'engram_embd=CPU'`); the remaining 206.86 GB contains the backbone and file metadata. Allow additional memory for the KV cache, runtime buffers, draft model and vision projector.

---

## Available files

| File | bpw | Size | Notes |
|---|---:|---:|---|
| `DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf` | 2.9986 (backbone) | 415.76 GB | 206.86 GB backbone + 208.90 GB Engram at Q8_0 |
| `DeepSeek-V4.1-Flash-DSpark-draft.gguf` | | 9.32 GB | DSpark speculative-decoding draft |
| `DeepSeek-V4.1-Flash-mmproj-BF16.gguf` | 16 | 0.97 GB | Vision encoder and projector |

---

## Results

**MMLU-Pro: no reasoning; 2,048-token context limit.** Scores use a fixed 2,000-question subset with raw zero-shot prompts and the log probabilities of single-token answer letters. These evaluations used F32 matrix multiplication and F32 KV caches in the supplied runtime, with Flash Attention disabled.

| Test | GSQ-RCO 3.0-bit result |
|---|---:|
| MMLU-Pro | **61.00%** (1,220/2,000) |
| Held-out PPL, 4,088 scored positions | **3.4845570141** |
| IFEval, 16 prompts | 16 passed the strict checker; 15 completed, 1 truncated |
| GSM8K, 8 problems | 8 completed correctly |

All 23 completed responses passed official parsing. The truncated IFEval response was not parsed. The [evaluation guide](eval/2026-09-13/README.md) includes the full protocol and saved responses.

---

## Download

```bash
hf download pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF --local-dir .
```

For text inference, follow the supplied [runtime build guide](runtime/README.md) and [reproduction instructions](runtime/REPRODUCE.md). Vision and speculative decoding have not been validated.

---

## Quantization procedure

1. GSQ generated a candidate bank for backbone projections across 40 layers and 384 experts. The bank is 1.1 TB; Engram tables and the vision tower were excluded from the search.
2. RCO selected the allocation by validation teacher KL. No step improved on step 0, so it kept the initial allocation.
3. The size budget included packed weights, metadata and alignment. A roundtrip check verified the payload.

Source revision: [`dba1be0a`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277).

---

## Citation

If you use these files, please cite this release together with the base model and both methods.

**This release**

```bibtex
@misc{dsv41flashgsqrco2026,
  title        = {DeepSeek-V4.1-Flash GSQ-RCO GGUF quantization},
  author       = {Josephine Pfeiffer},
  year         = {2026},
  publisher    = {Hugging Face},
  doi          = {10.57967/hf/10402},
  howpublished = {\url{https://huggingface.co/pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF}}
}
```

**Base model**

```bibtex
@misc{deepseekai2026deepseekv41flash,
  title  = {DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression},
  author = {DeepSeek-AI},
  year   = {2026},
  url    = {https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash}
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

Thanks also to ella for giving me the idea and supporting me at every step of the process, and Chris van Hoof (Red Hat) for contributing the compute on which this quantization was produced and evaluated.

---

## License

These quantized weights inherit the license of the base model, **[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)** (MIT); the original license text ships as `LICENSE`. The GSQ and RCO tooling is released by DASLab under its own repository licenses.
