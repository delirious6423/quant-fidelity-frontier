---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
tags:
- gguf
- llama.cpp
- lfm2.5
- quantized
- text-generation
---

# LFM2.5-24B-A2B GGUF

Community GGUF quantizations of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).

<div align="center" style="background-color:#f59e0b;color:#ffffff;padding:16px 20px;border-radius:10px;line-height:1.7;">
☕ If this GGUF made your day easier, a coffee would make mine.<br>
<a href="https://ko-fi.com/ngquocvinh" style="color:#ffffff;"><strong style="color:#ffffff;">Send a coffee ☕</strong></a><br>
I build and test these releases myself. Your coffee helps keep me going.<br>
Thank you for supporting this work.
</div>

## About LFM2.5-24B-A2B

[LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) is Liquid AI's text-only reasoning Mixture-of-Experts model. It has 24 billion total parameters and 2.45 billion active parameters, with 40 layers, 64 experts per MoE block, top-4 routing, and a 131,072-token context window. The upstream card lists English, Arabic, Chinese, French, German, Italian, Japanese, Korean, Portuguese, Spanish, Vietnamese, Thai, Indonesian, Hindi, Russian, and Polish. It describes the source model as a reasoning model with tool use. This release's runtime checks cover short text generation; long-context and tool-call behavior have not been evaluated here.

This release targets [llama.cpp](https://github.com/ggml-org/llama.cpp) and compatible GGUF runtimes. See the [official model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for the model's capabilities and usage guidance.

[![LFM2.5-24B-A2B benchmarks](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/resolve/main/figures/LFM2.5-24B-A2B-benchmark.png)](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)

*Benchmark figure from the [upstream model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B); results shown are upstream measurements.*

## Quick start

After downloading `LFM2.5-24B-A2B-Q4_K_M.gguf` and the included `chat_template.jinja`, run:

```bash
llama-cli \
  -m LFM2.5-24B-A2B-Q4_K_M.gguf \
  --chat-template-file chat_template.jinja \
  --jinja \
  -p 'Answer briefly in English: What is GGUF and why is it useful for running language models locally?' \
  -n 128 -c 4096 -ngl 99 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05
```

Choose a context size that fits your available memory. The 131,072-token context is the model's supported maximum; using it requires substantially more memory than this example.

## Reproducibility and validation

Each GGUF is built directly from the pinned official BF16 GGUF source with a model-specific imatrix. Files are added to this repository after llama.cpp loads them and produces a non-empty response. Exact file sizes in decimal GB, including the BF16 reference and imatrix, and per-file validation status are recorded in [`reproducibility/artifact-measurements.tsv`](reproducibility/artifact-measurements.tsv); SHA-256 checksums are in [`SHA256SUMS.txt`](SHA256SUMS.txt), and the build inputs and commands are summarized in [`reproducibility/manifest.md`](reproducibility/manifest.md).

### Size measurements

Percentages compare each file's size with the BF16 source (47.962095 GB). `Relative size (%) = file size / BF16 size × 100`. This percentage measures storage size, not model quality or fidelity.

| Quantization | Size (GB) | Relative size vs BF16 (%) |
| --- | ---: | ---: |
| BF16 source | 47.962095 | 100.00% |
| Q8_0 | 25.493733 | 53.15% |
| Q6_K | 19.689406 | 41.05% |
| Q5_K_M | 17.029603 | 35.51% |
| Q5_K_S | 16.541204 | 34.49% |
| Q4_K_M | 14.526259 | 30.29% |
| Q4_K_S | 13.660242 | 28.48% |
| IQ4_XS | 12.837437 | 26.77% |
| Q3_K_L | 12.428312 | 25.91% |
| Q3_K_M | 11.465719 | 23.91% |
| Q3_K_S | 10.429988 | 21.75% |
| IQ3_XXS | 9.282732 | 19.35% |
| Q2_K | 8.809758 | 18.37% |
| IQ2_XS | 7.090126 | 14.78% |
| IQ2_XXS | 6.370082 | 13.28% |
| TQ2_0 | 6.355845 | 13.25% |
| IQ1_M | 5.471666 | 11.41% |
| Q1_0 | 3.578019 | 7.46% |

No fixed hold-out fidelity comparison is published in this release. Size and runtime compatibility checks do not measure next-token fidelity.

## License and attribution

The upstream model is released under the [LFM Open License v1.0](LICENSE). Review its terms, including the commercial-use conditions, before using the model. These are community GGUF quantizations, not an official Liquid AI release or endorsement.
