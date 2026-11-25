---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- tensorfold
- tfq3
- qwen
- quantized
- apple-silicon
- metal
---

# Qwen3.8-27B-TFQ3

A **14.37 GiB** TensorFold checkpoint based on [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It combines 4-bit TFQ3 weights with a BF16 base. The weight files and tables account for 14.34 GiB of the download.

<p align="center">
  <img src="qwen-logo.png" alt="Qwen" width="200" />
</p>

## Before you download

This checkpoint requires a TensorFold native build with TFQ3 support for Qwen3.8-27B on Apple silicon (Metal). Keep the Safetensors file, TFQ3 shards and tables together. The results below cover text generation on an Apple M5 Ultra.

## Checkpoint and quality comparison

| Metric | MLX 8-bit reference | MLX 4-bit | TFQ3 with BF16 base |
| --- | ---: | ---: | ---: |
| Weight size | 27.50 GiB | 14.98 GiB | **14.34 GiB** |
| Perplexity, 4,790 tokens | 5.2468 | 5.3694 | **5.3680** |
| Held-out perplexity, 3,071 tokens | 6.4485 | 6.6021 | **6.5829** |
| Mean KL divergence to MLX 8-bit | — | 0.03352 | **0.02316** |
| 99th-percentile KL divergence to MLX 8-bit | — | 0.27518 | **0.15037** |

TFQ3 recorded slightly lower perplexity than MLX 4-bit on both sets. Across 1,022 evaluated token positions, it also recorded **31% lower mean KL divergence** and **45% lower 99th-percentile KL divergence** to the MLX 8-bit reference compared with MLX 4-bit.

## Live serving on Apple M5 Ultra

The reported live test measured **44.4 tok/s** streaming decode with no drafter, **145 ms** to first token and **2.1 seconds** to load. The server was configured for a 32,768-token context and eight stream slots, with a 43.29 GiB startup budget.

## Licence and credit

The original model is by the Qwen team. TensorFold modified the checkpoint weights for this TFQ3 release and distributes the result under the upstream [Apache License 2.0](LICENSE).
