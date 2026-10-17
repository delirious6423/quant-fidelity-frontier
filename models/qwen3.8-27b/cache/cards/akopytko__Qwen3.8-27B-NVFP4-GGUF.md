---
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- nvfp4
- blackwell
- q4_n
---

# Qwen3.8-27B NVFP4 GGUF Fast

This is a Blackwell-native 4-bit float quant designed to get the most out of your Blackwell card. It is tuned for maximum prefill speed, running 50% faster than conventional 4-bit quants with a comparable footprint.

It is different from the NVIDIA-style NVFP4 quant:

- It does not include global per-tensor scales, so it requires less compute to run (4-7% faster).
- It uses quantize-time MSE scale search to find the best UE4M3 code minimizing the reconstruction error.

It is similar to other NVFP4 quants in that it requires no changes to llama.cpp, so it runs with the current released version. You can download it and run it right away.

This NVFP4-exclusive quant requires NVIDIA Blackwell GPUs like the RTX 50 series, DGX Spark, B200, or B300, so you need one to use it fully.

## Speed

We benchmarked it and the contenders using the default flags (`-b 2048 -ub 512`) and peak-throughput flags (`-b 8192 -ub 4096`).

| model         | pp2048, ubatch 512 | pp2048, ubatch 4096 | tg128          |
|---------------|--------------------|---------------------|----------------|
| this NVFP4    | 6258.74 +/- 15.37  | 6432.25 +/- 143.73  | 87.45 +/- 0.27 |
| unsloth NVFP4 | 6018.71 +/- 13.97  | 5998.72 +/- 95.34   | 87.69 +/- 0.27 |
| NVIDIA+Q6_K   | 4643.32 +/- 66.89  | 4555.66 +/- 66.20   | 79.51 +/- 0.08 |
| Q4_0          | 4139.00 +/- 4.31   | 4173.82 +/- 28.23   | 87.90 +/- 0.29 |
| unsloth Q4_0  | 4156.58 +/- 7.61   | 4146.54 +/- 38.64   | 87.01 +/- 0.09 |
| unsloth Q6_K  | 3215.89 +/- 3.89   | n/a                 | 62.49 +/- 0.33 |

For benchmarks, we used RTX 5090 running at stock 575 W.

## MTP

NVFP4-quantized MTP draft head is embedded. Recommended settings for that sweet 1.6x boost:

```
--spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-p-min 0.8
```

With these settings, we observed 152 t/s decode at 0.75 draft acceptance on a typical prompt.

## Vision

Image input works out of the box when loading the model with `-hf`, else do:

```
--mmproj mmproj-BF16.gguf
```

## Quality

For accuracy benchmarks, we ran KLD and Top P agreement against the base bf16 model on the wikitext test set. Base perplexity: 6.7945.

[Unsloth's Q4_0](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) is a mixed quant (14.95 GiB) optimized with imatrix on a proprietary calibration dataset we don't have, so for an additional comparison, we added a straightforward Q4_0 quant: it matches the size and density of our quant byte by byte (14.63 GiB, 4.60 BPW).

| model            | KLD mean              | same top-p        |
|------------------|-----------------------|-------------------|
| N4_0             | 0.066254 +/- 0.000702 | 88.525 +/- 0.141% |
| unsloth NVFP4    | 0.070824 +/- 0.000733 | 88.061 +/- 0.144% |
| NVIDIA+Q6_K      | 0.039314 +/- 0.000432 | 90.784 +/- 0.128% |
| Q4_0 w/o imatrix | 0.037877 +/- 0.000448 | 91.461 +/- 0.124% |
| unsloth Q4_0     | 0.028317 +/- 0.000330 | 92.698 +/- 0.115% |
| unsloth Q6_K     | 0.002857 +/- 0.000051 | 97.506 +/- 0.069% |

Overall, our quant shows a pattern common to all NVFP4 quants: it falls behind Q4_0, so no surprises here. That said, we found it edging ahead of the Unsloth's NVFP4 quant.

Although this quant is just as smart as other NVFP4 quants, it is 4-7% faster.

If you want to make no compromises, use the near-lossless Q6_K quant from this repository with the same NVFP4-quantized MTP head: since the main model verifies every drafted token, it cannot degrade outputs, so you get to benefit from both speed and quality.

## NVIDIA NVFP4 with Q6_K

For a fair comparison and to provide options, I included `Qwen3.8-27B-NVFP4-NVIDIA-Q6_K.gguf` which a build of [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4): the NVFP4 tensors and scales of that checkpoint with Q6_K instead of FP8. Use the full tag to select it:

```
-hf akopytko/Qwen3.8-27B-NVFP4-GGUF:NVIDIA-Q6_K
```

## Reproduction

To recreate this quant, or quantize other models using this technique, you will need to pick up a pair of PRs ([#23572](https://github.com/ggml-org/llama.cpp/pull/23572), [#23692](https://github.com/ggml-org/llama.cpp/pull/23692)) with the relevant changes, then use the NVFP4 quantization type.

```
llama-quantize Model-bf16.gguf Model-M4_0.gguf NVFP4
```
