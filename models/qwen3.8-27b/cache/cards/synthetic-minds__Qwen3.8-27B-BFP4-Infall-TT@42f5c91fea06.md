---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
  - tenstorrent
  - blackhole
  - bfp4
  - gptq
  - infall-tt
  - qwen3.8
---

<h1 align="center">Qwen3.8-27B-BFP4-Infall-TT</h1>

<br>

<p align="center">
  <img src="assets/qwen-banner.svg" alt="Qwen" width="420">
</p>

<br>

<p align="center">
  <img src="assets/made-with-infall-tt.png" alt="Synthetic Minds: made with Infall-TT" width="240">
</p>

<br>

<p align="center">
  <a href="https://www.syntheticmindslab.com/"><img src="https://img.shields.io/badge/Website-syntheticmindslab.com-111111?style=for-the-badge&logo=safari&logoColor=white&labelColor=555555" alt="Website"></a>
  <a href="https://x.com/syntheticminds2"><img src="https://img.shields.io/badge/X-Synthetic%20Minds-111111?style=for-the-badge&logo=x&logoColor=white&labelColor=555555" alt="X (Synthetic Minds)"></a>
  <a href="https://x.com/tudoriustin_"><img src="https://img.shields.io/badge/X-Tudor-111111?style=for-the-badge&logo=x&logoColor=white&labelColor=555555" alt="X (Tudor)"></a>
</p>

<p align="center">
  <a href="https://www.linkedin.com/company/synthetic-minds-lab/"><img src="https://img.shields.io/badge/LinkedIn-Synthetic%20Minds-1F5BB5?style=for-the-badge&labelColor=555555" alt="LinkedIn"></a>
</p>

<br>

<p align="center">
  <a href="https://huggingface.co/synthetic-minds"><img src="https://img.shields.io/badge/Hugging%20Face-synthetic--minds-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=555555" alt="Hugging Face"></a>
  <a href="https://huggingface.co/collections/synthetic-minds/infall-tt-optimised-gguf-20-6b01ce6d8f799e1cde015b58"><img src="https://img.shields.io/badge/GGUF%20Collection-Infall--TT%20Optimised-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=555555" alt="GGUF collection"></a>
</p>

<p align="center">
  <a href="https://github.com/syntheticminds-lab"><img src="https://img.shields.io/badge/GitHub-syntheticminds--lab-111111?style=for-the-badge&logo=github&logoColor=white&labelColor=555555" alt="GitHub"></a>
  <a href="https://buymeacoffee.com/syntheticminds_lab"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-syntheticminds__lab-FFD21E?style=for-the-badge&logo=buymeacoffee&logoColor=white&labelColor=555555" alt="Buy Me a Coffee"></a>
</p>

<br>

> ***"Quantized with the new beta preview of Infall-TT,** our custom framework for training, quantizing and
> running large language models on Tenstorrent Neural Accelerator Hardware.
> [Read the introduction](https://www.syntheticmindslab.com/).*
>
> ***Follow us and [sign up for the Infall-TT waitlist](https://www.syntheticmindslab.com/) for more quants like this."***

## Model details

| | |
|---|---|
| **Original model** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Model provider** | Qwen |
| **Release date** | 21 November 2026 |
| **Languages** | Multilingual (as the original model) |
| **Model type** | Causal language model with vision encoder |
| **Parameters** | 27B |
| **License** | Apache-2.0 |
| **Quantization** | Tenstorrent BFP4 (`bfloat4_b`), quantized with Infall-TT by Synthetic Minds |
| **Model developer** | Synthetic Minds |
| **Target hardware** | Tenstorrent Blackhole: P150A/B, P300C, Galaxy Blackhole |

## Quantization quality

Measured against the original BF16 model on held-out text the quantizer never saw: 40 windows of 2,048
tokens per set (81,880 predicted tokens each). "Tenstorrent default" is what tt-metal produces from the
original weights: every weight rounded to the nearest value on the same BFP4 grid.

| | WikiText-2 perplexity | C4 perplexity | KL divergence (C4) | Top-1 agreement (C4) |
|---|---|---|---|---|
| BF16 (original) | 6.489 | 12.193 | 0 | 100% |
| Tenstorrent default BFP4 | 7.205 (+11.0%) | 12.839 (+5.3%) | 0.059 | 85.7% |
| **Infall-TT BFP4** | **6.750 (+4.0%)** | **12.600 (+3.3%)** | **0.042** | **87.7%** |

Lower perplexity and KL divergence are better; higher top-1 agreement (how often the quantized model's most
likely next token matches the original's) is better.

**Why Infall-TT is better.** Round-to-nearest treats every weight on its own, so the small errors add up
through the network. Infall-TT solves for the weights layer by layer with GPTQ directly on Tenstorrent's
BFP4 grid: each rounding error is compensated by adjusting the weights not yet rounded, using real text
to see which errors matter, and the shared exponent of each block is chosen to minimize the error. On the
same format and the same memory, this cuts the perplexity increase from 11.0% to 4.0% on WikiText-2 and
from 5.3% to 3.3% on C4, and the KL divergence by 42% (WikiText-2: 0.093 → 0.054) and 29% (C4).

## How to use it

This repository contains the full weights in the standard Hugging Face format: the same files, tensor
names and shapes as the original, about 56 GB to download. The quantized values are stored in BF16, so
any tool that reads the original checkpoint reads this one.

When tt-metal loads the weights, it automatically compresses every projection and the LM head to BFP4
during loading. Every value in this checkpoint already lies exactly on that BFP4 grid, so the compression
keeps the Infall-TT weights unchanged instead of rounding the originals. In BFP4 the weights take about
4.5 bits each, roughly 15 GB for the model.

Download it and point tt-metal's Qwen3.8-27B demo (or its vLLM integration) at it instead of the original
weights. No code changes are needed:

```bash
huggingface-cli download synthetic-minds/Qwen3.8-27B-BFP4-Infall-TT --local-dir Qwen3.8-27B-BFP4-Infall-TT
export MODEL_WEIGHTS_DIR=$PWD/Qwen3.8-27B-BFP4-Infall-TT
python models/demos/qwen38_27b_qb2/demo/text_demo.py --full --length 128 --generate 128
```

The weight layout was checked against tt-metal's
[`models/demos/qwen38_27b_qb2`](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/qwen38_27b_qb2)
implementation.

## What was quantized

- Every decoder projection (Gated DeltaNet and full-attention layers, MLP) and the LM head: BFP4, 4.5 bits
  per weight including shared exponents. 497 tensors in total.
- Embeddings, norms, convolution and recurrence parameters: unchanged (BF16), as in tt-metal.
- The vision tower and the multi-token prediction layers are unchanged; this quantization targets the
  text model.

## Method

- GPTQ solved directly on Tenstorrent's BFP4 grid, with shared-exponent blocks along the output dimension
  (the layout tt-metal packs after transposing each weight) and a per-block exponent search.
- Calibration: 128 sequences of 2,048 tokens from the C4 training split.
- Every block-float value is bit-identical to what tt-metal's host packer (`ttnn.from_torch`) produces.
- Verified with tt-metal's own weight preparation for this model, run on the host over all 64 layers:
  2,048 of 2,048 device shards came out already on the grid. Every other tensor is identical to the
  original checkpoint.

## Limitations

- Quantized and evaluated on our development server, using Infall-TT's bit-exact emulation of
  Tenstorrent's block-float formats. It has not yet been run on Tenstorrent hardware.
- No speed claims: speed depends on tt-metal and the hardware, not on this checkpoint.
- Accuracy is reported as perplexity, KL divergence and top-1 agreement on text; no task benchmarks and
  no image or video evaluation yet.

## Support our work

Infall-TT is built by Synthetic Minds without access to Tenstorrent hardware: the framework is developed
and tested on Tenstorrent's ttsim simulator. Your support will help us get the funding we need to acquire
a TT-QuietBox 2, so we can run and verify models like this one on real silicon, which the simulator cannot
do at this size, and accelerate our work for the Tenstorrent open-source community and hardware.

<a href="https://buymeacoffee.com/syntheticminds_lab"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-syntheticminds__lab-FFD21E?style=for-the-badge&logo=buymeacoffee&logoColor=white&labelColor=555555" alt="Buy Me a Coffee"></a>

Get in touch: hello@syntheticmindslab.com.

## License

Apache-2.0, the same license as the original model.

- Original model: [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) © the Qwen team.
- Quantization: © Synthetic Minds | Infall-TT Framework.
