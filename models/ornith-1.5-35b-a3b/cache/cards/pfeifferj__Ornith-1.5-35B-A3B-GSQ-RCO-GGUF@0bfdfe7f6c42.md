---
base_model: ornith-ai/Ornith-1.5-35B-A3B
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
---

> **TL;DR:** Text-only Ornith-1.5-35B-A3B at 13.0 GB (3-bit) or 15.2 GB (3.5-bit), down from 69.4 GB in BF16.

<div align="center">

# Ornith-1.5-35B-A3B &middot; GSQ-RCO GGUFs

3-bit and 3.5-bit GGUFs made with GSQ and RCO.

[![arXiv: GSQ](https://img.shields.io/badge/arXiv-GSQ_2604.18556-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.18556)
[![arXiv: RCO](https://img.shields.io/badge/arXiv-RCO_2605.00649-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.00649)
[![GSQ code](https://img.shields.io/badge/code-GSQ-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/GSQ)
[![RCO code](https://img.shields.io/badge/code-RCO-181717?logo=github&logoColor=white)](https://github.com/IST-DASLab/RCO)
[![DASLab](https://img.shields.io/badge/DASLab-GitHub-101048?logo=github&logoColor=white)](https://github.com/IST-DASLab)
[![license](https://img.shields.io/badge/license-MIT-19a34a)](#license)

</div>

> Independent community reproduction. These files were produced by a third party using the published GSQ and RCO methods. They are not an IST-DASLab release and carry no endorsement from the authors of either paper.

![Ornith-1.5 perplexity for BF16, GSQ-RCO and both controls](assets/plots/ornith-perplexity.png)

---

## Overview

[Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) is a Qwen3.5-family mixture-of-experts model with about 35B parameters and 3B active per token. Both GGUFs include the text model, tokenizer and chat template.

| Method | Description |
|---|---|
| **GSQ** (Gumbel-Softmax Quantization, [paper](https://arxiv.org/abs/2604.18556), [code](https://github.com/IST-DASLab/GSQ)) | Post-training scalar quantization that jointly learns per-coordinate grid assignments and per-group scales through a Gumbel-Softmax relaxation. |
| **RCO** (Riemannian Constrained Optimization, [paper](https://arxiv.org/abs/2605.00649), [code](https://github.com/IST-DASLab/RCO)) | Assigns one of K quantization types to each of N tensors under an exact total size budget, reformulated as a smooth Riemannian manifold in logit space. |

Both methods were developed at the [Deep Algorithms and Systems Lab (DASLab)](https://github.com/IST-DASLab), Institute of Science and Technology Austria.

---

## Results

**MMLU-Pro: no reasoning, 2,048-token context, zero generated tokens.** The test uses 2,000 questions across all 14 categories. Zero-shot scoring chooses the answer with the highest log likelihood for its full letter continuation (` A`, ` B`, etc.), without a chat template. [Question IDs and scores](eval/mmlu-pro/).

Perplexity covers 4,088 next-token predictions across eight 1,024-token contexts of held-out text. Lower is better.

| Build | Size | PPL | MMLU-Pro |
|---|---:|---:|---:|
| BF16 reference | 69.377 GB | 3.4112 | 47.60% |
| GSQ-RCO 3-bit | 12.992 GB | 3.5369 | 43.70% |
| Native imatrix 3-bit | 12.984 GB | 3.5320 | 41.90% |
| Matched initializer 3-bit | 12.992 GB | 3.5353 | 45.35% |
| GSQ-RCO 3.5-bit | 15.163 GB | 3.4575 | 43.20% |
| Native imatrix 3.5-bit | 15.156 GB | 3.4795 | 43.95% |
| Matched initializer 3.5-bit | 15.163 GB | 3.4575 | 44.65% |

Perplexity rises 3.7% at 3-bit and 1.4% at 3.5-bit over BF16. Both GSQ builds score below their matched initializers on MMLU-Pro.

Matched initializers use the same tensor formats and file sizes as the GSQ builds, with native quantized weights before GSQ optimization. The native imatrix controls have separate allocations.

GSM8K uses eight questions, thinking enabled and a 2,048-token output cap; scoring checks the numerical answer after `####`. IFEval uses 16 prompts, thinking disabled, strict scoring and a 1,024-token cap. Both use temperature 0 and an 8,192-token context.

| Build | GSM8K | IFEval strict | IFEval at token cap |
|---|---:|---:|---:|
| BF16 reference | 4/8 | 14/16 | 2/16 |
| GSQ-RCO 3-bit | 8/8 | 11/16 | 3/16 |
| GSQ-RCO 3.5-bit | 7/8 | 14/16 | 1/16 |

Two BF16 IFEval passes and one 3-bit pass reached the cap. Excluding capped passes gives 12/16, 10/16 and 14/16, respectively. No GSM8K answer reached the cap.

---

## Usage

### llama.cpp

Tested with upstream [`llama.cpp` at `58367713`](https://github.com/ggml-org/llama.cpp/commit/58367713a6935c0810103378144008df32e3d5db).

```bash
hf download pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF Ornith-1.5-35B-A3B-GSQ-RCO-3.5bit.gguf --local-dir .

llama-server -m Ornith-1.5-35B-A3B-GSQ-RCO-3.5bit.gguf \
  -ngl 999 -c 8192 -fa on -ctk f32 -ctv f32 --jinja \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  --chat-template-kwargs '{"preserve_thinking":true}'
```

These are Ornith's recommended sampling settings. Thinking is on by default; disable it with `--chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true}'`.

---

## Quantization procedure

Source: [ornith-ai/Ornith-1.5-35B-A3B at `10fbf86f`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/tree/10fbf86fed7ecee4a061f8b499a618f46001cac1). Routed experts mix Q2_K, Q3_K and Q4_K. Embeddings and the output matrix use Q4_K; attention and shared-expert matrices use Q8_0. Routers, norms and remaining convolution and state tensors use F32.

1. Capture expert activations across 40 layers from one million tokens: 45% calibration_mixture, 40% OpenThoughts and 15% FineWeb. Keep up to 128 training and 64 validation rows per expert.
2. Run 40 GSQ updates per projection and format with 32 rows per update. Fit gate, up and down outputs separately, using the original gate/up activations for the down projection. Scales and offsets stay fixed; keep the initializer when it validates better.
3. Run 50 RCO steps per size on four calibration sequences and choose the allocation on four separate validation sequences. Assemble the selected packed tensors into GGUF without requantization. [Quantization settings](quantization.json).

---

## Citation

**This release** · [DOI: 10.57967/hf/10579](https://doi.org/10.57967/hf/10579)

```bibtex
@misc{ornithgsqrco2026,
  title        = {Ornith-1.5-35B-A3B GSQ-RCO GGUF quantizations},
  author       = {Josephine Pfeiffer},
  year         = {2026},
  publisher    = {Hugging Face},
  doi          = {10.57967/hf/10579},
  howpublished = {\url{https://huggingface.co/pfeifferj/Ornith-1.5-35B-A3B-GSQ-RCO-GGUF}}
}
```

**Base model**

```bibtex
@misc{ornith_1_5,
  title  = {{Ornith-1.5}: From Self-Scaffolding to Self-Improvement},
  url    = {https://ornith.ai/ornith_1_5.html},
  author = {{Ornith Team}},
  year   = {2026}
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

Thanks to the [Ornith team](https://github.com/ornith-ai/Ornith-1) for Ornith-1.5 and to the [Deep Algorithms and Systems Lab (DASLab)](https://github.com/IST-DASLab) for GSQ, RCO and their reference implementations.

---

## License

The weights use the base model's MIT license. [LICENSE](LICENSE) comes from the [official Ornith repository](https://github.com/ornith-ai/Ornith-1/blob/9c800446621ebad1172f517a1d0f9b683fb8615e/LICENSE).
