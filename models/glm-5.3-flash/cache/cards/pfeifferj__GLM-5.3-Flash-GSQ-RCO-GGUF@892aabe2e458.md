---
base_model: zai-org/GLM-5.3-Flash
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

> **TL;DR:** GLM's 320B-parameter multimodal model in 118.6 to 138.2 GB including vision, compared with 328.3 GB for the FP8 checkpoint.

<div align="center">

# GLM-5.3-Flash &middot; GSQ-RCO GGUFs

**Non-uniform GGUF quantizations** produced with **GSQ** and **RCO**, with a vision projector for multimodal use.

[![arXiv: GSQ](https://img.shields.io/badge/arXiv-GSQ_2604.18556-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18556)
[![arXiv: RCO](https://img.shields.io/badge/arXiv-RCO_2605.00649-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.00649)
[![GSQ code](https://img.shields.io/badge/code-GSQ-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/GSQ)
[![RCO code](https://img.shields.io/badge/code-RCO-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/RCO)
[![DASLab](https://img.shields.io/badge/DASLab-GitHub-101048?logo=github&logoColor=white)](https://github.com/IST-DASLab)
[![license](https://img.shields.io/badge/license-mit-19a34a)](#license)

</div>

> Independent community reproduction. These files were produced by a third party using the published GSQ and RCO methods. They are not an IST-DASLab release and carry no endorsement from the authors of either paper.

![Native perplexity against the Q8_0 reference](assets/plots/glm-perplexity-current.png)

---

## Overview

This repository contains two GGUF quantizations of **[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** and its BF16 vision projector (`mmproj`). RCO assigns each weight tensor a quantization type according to its sensitivity, within an exact total size budget.

| Method | Description |
|---|---|
| **GSQ** (Gumbel-Softmax Quantization, [paper](https://arxiv.org/abs/2604.18556), [code](https://github.com/IST-DASLab/GSQ)) | Post-training scalar quantization that jointly learns per-coordinate grid assignments and per-group scales through a Gumbel-Softmax relaxation. |
| **RCO** (Riemannian Constrained Optimization, [paper](https://arxiv.org/abs/2605.00649), [code](https://github.com/IST-DASLab/RCO)) | Assigns one of K quantization types to each of N tensors under an exact total size budget, reformulated as a smooth Riemannian manifold in logit space. |

Both methods were developed at the [Deep Algorithms and Systems Lab (DASLab)](https://github.com/IST-DASLab), Institute of Science and Technology Austria.

---

## Read this before using these files

Evaluation uses `llama.cpp` PR 27773 with `native-f32-mmf.patch` and `NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_MMF_F32_DISABLE=1`. The [runtime instructions](#runtime) include the build and launch commands.

---

## Available files

| File | bpw | Size | Notes |
|---|---:|---:|---|
| `GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf` | 3.499816 | 137.07 GB | Better perplexity of the two |
| `GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf` | 2.999595 | 117.48 GB | Smaller |
| `GLM-5.3-Flash-mmproj-BF16.gguf` | 16.52 | 1.16 GB | Vision encoder and projector |

---

## MMLU-Pro

**No reasoning; 2,048-token context limit.** MMLU-Pro uses 2,000 questions sampled with a fixed seed and stratified across all 14 categories. Questions are scored zero-shot using the log probabilities of single-token answers ` A` through ` J`, without a chat template. All three builds use the same patched runtime and matching layer placement. Raw scores: [Q8_0](eval/2026-09-13/results/mmlu/glm-q8.tsv), [3.5-bit](eval/2026-09-13/results/mmlu/glm3.5-three-gpu.tsv), [3.0-bit](eval/2026-09-13/results/mmlu/glm3-three-gpu.tsv).

The eight-bit Q8_0 reference is converted from the pinned FP8 source, excluding the unused MTP layer. The [reference record](eval/2026-09-13/baseline/glm-q8/reference-evaluation.json) contains the conversion details and hashes.

![MMLU-Pro accuracy by build](assets/plots/glm-mmlu-rerun.png)

| Build | Accuracy | SE | vs Q8_0, paired | Discordant (Q8_0 right / build right) | Exact p |
|---|---:|---:|---:|---:|---:|
| Q8_0 reference | 61.95% | 1.09 | | | |
| GSQ-RCO 3.5-bit | 60.55% | 1.09 | -1.40 pp | 86 / 58 | 0.0241 |
| GSQ-RCO 3.0-bit | 60.00% | 1.10 | -1.95 pp | 128 / 89 | 0.00973 |

Correct counts are 1,239, 1,211 and 1,200 out of 2,000. Chance is 11.2%. The 3.5-bit model leads 3.0-bit by 0.55 pp (118 / 107 discordant, exact p = 0.505). Q8_0 and 3.5-bit agree on 1,739 predictions. P-values are two-sided and unadjusted.

Per category:

| Category | n | Q8_0 | 3.5-bit | 3.0-bit |
|---|---:|---:|---:|---:|
| biology | 119 | 93.3% | 93.3% | 88.2% |
| business | 131 | 48.1% | 46.6% | 48.1% |
| chemistry | 188 | 45.2% | 44.7% | 41.5% |
| computer science | 68 | 76.5% | 76.5% | 75.0% |
| economics | 140 | 80.7% | 80.0% | 79.3% |
| engineering | 161 | 49.7% | 48.4% | 52.8% |
| health | 136 | 71.3% | 66.9% | 72.1% |
| history | 63 | 73.0% | 74.6% | 71.4% |
| law | 183 | 58.5% | 57.9% | 56.3% |
| math | 225 | 49.8% | 44.9% | 44.0% |
| other | 154 | 69.5% | 68.8% | 65.6% |
| philosophy | 83 | 66.3% | 69.9% | 71.1% |
| physics | 216 | 45.4% | 42.6% | 42.1% |
| psychology | 133 | 85.0% | 84.2% | 83.5% |

---

## Results

Perplexity is measured on the same held-out text for all three builds: eight 1,024-token contexts and 4,088 scored tokens. Generation scores cover 16 IFEval and 8 GSM8K items. Completed-correct counts exclude responses that hit the token cap.

| Variant | bpw | GB | Native PPL↓ | vs Q8_0 | Approx. KL↓ | IFEval strict | IFEval completed correct | GSM8K completed correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 reference | n/a | n/a | **3.3909** | n/a | n/a | 4/16 | 2/16 | 5/8 |
| **GSQ-RCO 3.5-bit** | 3.499816 | 137.07 | 3.5431 | +4.49% | 0.071346 | 7/16 | 4/16 | 5/8 |
| GSQ-RCO 3.0-bit | 2.999595 | 117.48 | 3.6985 | +9.07% | 0.142370 | 10/16 | 4/16 | 5/8 |

Q8_0 emitted reasoning on five IFEval requests despite the thinking-off adapter. KL is approximate because the native reference cache stores quantized log probabilities. Raw responses: [Q8_0](eval/2026-09-13/results/diagnostics/glm-q8.mini.json), [3.5-bit](eval/2026-09-13/results/diagnostics/glm3.5-three-gpu.mini.json), [3.0-bit](eval/2026-09-13/results/diagnostics/glm3-three-gpu-isolated.mini.json).

---

## Runtime

1. The base is upstream [`8134115f`](https://github.com/ggml-org/llama.cpp/commit/8134115f88ed8018474e7db69afcfe97fb097fc4).
2. llama.cpp PR 27773 adds support for the `glm5-next` architecture.
3. `native-f32-mmf.patch` from this repository: rejects F32 candidates in the MMF path so no TF32 tensor-core instructions are emitted. `NVIDIA_TF32_OVERRIDE=0` alone does not do this. No behaviour changes unless `GGML_CUDA_MMF_F32_DISABLE=1` is set.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch origin de25343596a2c924ba62d35b126e811be0d38a49
git checkout de25343596a2c924ba62d35b126e811be0d38a49
git apply /path/to/native-f32-mmf.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j
```

```bash
NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_MMF_F32_DISABLE=1 ./build/bin/llama-server \
  -m GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf \
  --mmproj GLM-5.3-Flash-mmproj-BF16.gguf \
  -ngl all -c 16384 -np 4 --kv-unified --load-mode none --lazy-mode off -fa off --jinja
```

```bash
hf download pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF \
  GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf GLM-5.3-Flash-mmproj-BF16.gguf --local-dir .
```

---

## Quantization procedure

1. GSQ trained expert projection candidates across 45 layers and 288 experts for 16 steps, using 64 training and 4 validation sequences of 1024 tokens.
2. RCO selected the allocation by validation teacher KL. Neither target improved on step 0, so both kept their initial allocation.
3. The size budget included packed weights, metadata and alignment. A roundtrip check verified the payload.

Source revision: [`eb9eb208`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a).

| Target | Q4_K | Q3_K | Q2_K |
|---|---:|---:|---:|
| 3.5-bit | 2 | 105 | 19 |
| 3.0-bit | 9 | 9 | 108 |

![Quantization type mix at each budget](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF/resolve/main/assets/plots/glm-allocation.png)

---

## Citation

If you use these files, please cite this release together with the base model and both methods.

**This release**

```bibtex
@misc{glm53flashgsqrco2026,
  title        = {GLM-5.3-Flash GSQ-RCO GGUF quantizations},
  author       = {Josephine Pfeiffer},
  year         = {2026},
  publisher    = {Hugging Face},
  doi          = {10.57967/hf/10400},
  howpublished = {\url{https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF}}
}
```

**Base model**

```bibtex
@misc{glm53flash,
  title  = {GLM-5.3-Flash},
  author = {{Zhipu AI}},
  year   = {2026},
  url    = {https://huggingface.co/zai-org/GLM-5.3-Flash}
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

These quantized weights inherit the license of the base model, **[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** (MIT); the original license text ships as `LICENSE`. The GSQ and RCO tooling is released by DASLab under its own repository licenses.
