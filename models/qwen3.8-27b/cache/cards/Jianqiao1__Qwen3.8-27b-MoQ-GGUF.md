---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - gguf
  - llama.cpp
  - moq
  - mtp
  - quantized
---

# Qwen3.8-27B-MoQ-GGUF

This repository contains imatrix-aware, layer-aware Mixture-of-Quantization (MoQ) GGUF files for the original [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) model. These files preserve the model's MTP tensors and are quantizations of the base Qwen model. In terms of overall quality, MoQ is comparable to Unsloth's Dynamic Quantization 3.0; however, our quantization series offers a more granular range of options (4.0 bpw – 4.75 bpw) than the Unsloth team's offerings, while maintaining a slight performance edge over Dynamic Quantization 3.0 within this range.

The complete series was evaluated locally alongside the Unsloth Dynamic GGUF series under identical conditions. Lower is better in all three charts.

![p999 KLD comparison](p999_kld.png)

![Mean KLD comparison](mean_kld.png)

![WikiText-2 PPL comparison](ppl.png)

The [interactive comparison report](Qwen3.8-27B-MTP-MoQ-vs-Unsloth-KLD.html) supports pan, zoom, view reset, and per-series visibility controls. The [companion CSV](Qwen3.8-27B-MTP-MoQ-vs-Unsloth-KLD.csv) contains all measured values and tensor-composition summaries.

## Comparison Summary

All 17 measured GGUF files contain 866 tensors and 27,320,697,856 parameters. They were evaluated against the same BF16 reference logits on WikiText-2 with a fixed context length of 512.

At the exact size of each of the 9 MoQ files, linearly interpolating the Unsloth curve favors this MoQ series on:

- PPL: 8 of 9 points
- p999 KLD: 7 of 9 points
- Mean KLD: 2 of 9 points

The results show the intended tradeoff clearly. The layer-aware MoQ recipes are especially effective on PPL and tail divergence, while the Unsloth Dynamic recipes remain very strong on Mean KLD, particularly at low and middle file sizes. The full table is included below so that users can choose based on the metric that matters for their workload.

## Full Quality Results

`Actual BPW` is computed from the complete GGUF file size, including metadata and alignment. GB is decimal. PPL and all KLD values are lower-is-better; same top-p is higher-is-better.

| Series | Recipe | Actual BPW | GB | PPL | Mean KLD | p999 KLD | p99 KLD | RMS delta-p | Same top-p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jianqiao1 MoQ | 3.2 | 3.0956 | 10.572 | 7.408463 | 0.111637 | 3.348125 | 1.084846 | 9.634% | 85.679% |
| Jianqiao1 MoQ | 3.6 | 3.4821 | 11.892 | 7.221080 | 0.072244 | 2.351531 | 0.717370 | 7.682% | 88.722% |
| Jianqiao1 MoQ | 3.8 | 3.7232 | 12.715 | 7.122706 | 0.048190 | 1.624749 | 0.437946 | 6.077% | 90.109% |
| Jianqiao1 MoQ | 4.1 | 4.1395 | 14.137 | 7.068801 | 0.031460 | 1.056327 | 0.285069 | 4.915% | 91.872% |
| Jianqiao1 MoQ | 4.3 | 4.3600 | 14.890 | 7.045628 | 0.024500 | 0.805452 | 0.216612 | 4.332% | 92.648% |
| Jianqiao1 MoQ | 4.6 | 4.4302 | 15.129 | 7.014436 | 0.020171 | 0.758995 | 0.200817 | 3.936% | 93.809% |
| Jianqiao1 MoQ | 4.8 | 4.6961 | 16.038 | 7.008832 | 0.017282 | 0.650152 | 0.169743 | 3.584% | 94.339% |
| Jianqiao1 MoQ | 4.9 | 4.8055 | 16.411 | 7.021034 | 0.016342 | 0.651529 | 0.162814 | 3.530% | 94.423% |
| Jianqiao1 MoQ | 5.1 | 5.0772 | 17.339 | 7.020549 | 0.012857 | 0.487299 | 0.123491 | 3.134% | 94.968% |
| Unsloth | UD-IQ2_XXS | 2.6383 | 9.010 | 7.957195 | 0.163689 | 4.714475 | 1.739405 | 11.822% | 82.389% |
| Unsloth | UD-IQ2_M | 3.0219 | 10.320 | 7.507406 | 0.103564 | 3.590751 | 1.076759 | 9.332% | 85.496% |
| Unsloth | UD-Q2_K_XL | 3.1263 | 10.676 | 7.403969 | 0.091579 | 3.107871 | 0.925859 | 8.708% | 86.160% |
| Unsloth | UD-IQ3_XXS | 3.4885 | 11.914 | 7.246240 | 0.054517 | 2.259040 | 0.627471 | 6.766% | 90.359% |
| Unsloth | UD-Q3_K_XL | 3.9358 | 13.441 | 7.111811 | 0.031355 | 1.248320 | 0.353498 | 5.129% | 92.396% |
| Unsloth | UD-Q4_K_XL | 5.2483 | 17.923 | 6.976017 | 0.008640 | 0.422880 | 0.090719 | 2.614% | 96.073% |
| Unsloth | UD-Q5_K_XL | 5.9203 | 20.218 | 6.967300 | 0.004526 | 0.200576 | 0.040594 | 1.849% | 97.153% |
| Unsloth | UD-Q6_K_XL | 7.5911 | 25.924 | 6.953912 | 0.001331 | 0.062800 | 0.011536 | 1.107% | 98.493% |

The Unsloth rows are local comparison measurements only. Unsloth GGUF files are not redistributed in this repository.

## Available Models

The recipe number is a series label. `Actual BPW` below is calculated from the complete file and is the value to use for exact size comparisons.

| File | Actual BPW | Size GB | Size GiB |
|---|---:|---:|---:|
| `Qwen3.8-27B-MTP-MoQ-3.2.gguf` | 3.0956 | 10.572 | 9.846 |
| `Qwen3.8-27B-MTP-MoQ-3.6.gguf` | 3.4821 | 11.892 | 11.075 |
| `Qwen3.8-27B-MTP-MoQ-3.8.gguf` | 3.7232 | 12.715 | 11.842 |
| `Qwen3.8-27B-MTP-MoQ-4.1.gguf` | 4.1395 | 14.137 | 13.166 |
| `Qwen3.8-27B-MTP-MoQ-4.3.gguf` | 4.3600 | 14.890 | 13.867 |
| `Qwen3.8-27B-MTP-MoQ-4.6.gguf` | 4.4302 | 15.129 | 14.090 |
| `Qwen3.8-27B-MTP-MoQ-4.8.gguf` | 4.6961 | 16.038 | 14.936 |
| `Qwen3.8-27B-MTP-MoQ-4.9.gguf` | 4.8055 | 16.411 | 15.284 |
| `Qwen3.8-27B-MTP-MoQ-5.1.gguf` | 5.0772 | 17.339 | 16.148 |

The repository also includes the locally generated calibration imatrix and a BF16 multimodal projector.

## Quantization Approach

This series combines the advantages of the MoQ (Mixed-precision Quantization) scheme at the layer level with tensor-specific strategies derived from Unsloth Dynamic 3.0's low-bit GGUF implementations. Since Qwen3.8 and Qwen3.6 share a matching tensor-level architecture, we were able to transfer and fine-tune layer-specific quantization strategies without being forced to apply a uniform quantization type across all tensor families.

The specific process is as follows:

1. Generate a calibration imatrix specifically for Qwen3.8 using a calibration corpus.
2. Transfer the layer-wise MoQ allocation scheme from Qwen3.6 to the corresponding tensors in Qwen3.8.
3. Compare each tensor family and layer against the original weights.
4. Retain MoQ layer-wise precision adjustments (whether increasing or decreasing precision) if they improve perplexity (PPL) or KLD tail performance.
5. While maintaining the layer-wise structure, optimize average KLD performance for the low-BPW (bits per weight) schemes of versions 3.2 and 3.6 by referencing Unsloth's low-bit quantization strategies.

This is not a single, global quantization preset; different precision settings can be applied to specific tensor families and layers based on their measured or inferred sensitivity.

## MTP Precision

Compared to the MTP layer in Qwen 3.6, the MTP layer in Qwen 3.8 employs lower-bit quantization; this is because our tests revealed that the prediction performance of Qwen 3.8's MTP layer—even at Q8 precision—falls short of that of Qwen 3.6, rendering the maintenance of high precision for the MTP layer less meaningful.

## Evaluation Conditions

The comparison used llama.cpp build 10276, commit `6ea215d17`, with CUDA on an NVIDIA GeForce RTX 5090. WikiText-2 `wiki.test.raw` was evaluated against BF16 logits from the original Qwen3.8-27B GGUF. Context length was fixed at 512; logical batch was 2048, ubatch was 8192, GPU layers were selected automatically with fit enabled, op offload and flash attention were enabled, and 16 CPU threads were used. GPU evaluations were run strictly one at a time.

The BF16 reference PPL reported by the evaluator was 6.950282.

## Usage

Use a recent llama.cpp build with Qwen3.8 and MTP support. The files can be used for ordinary generation or MTP speculative decoding. Choose a file according to available memory and the quality curves above; the nominal recipe label is useful for navigating the series, while the actual file BPW and GB columns provide precise memory-planning values.

## License and Acknowledgements

Released under the Apache License 2.0, following the base model license metadata.

Thanks to:

- the Qwen team for Qwen3.8-27B and its MTP architecture;
- the llama.cpp project and contributors for GGUF quantization, MTP support, and evaluation tooling;
- the Unsloth team for the Dynamic GGUF series.
