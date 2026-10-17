---
license: apache-2.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags: [exl3, exllamav3, quantized, qwen3.8, tandemllm, speculative-decoding]
---
# Qwen3.8-27B in EXL3, 2.5 bits per weight

These are EXL3 weights of Qwen3.8-27B at 2.5 bits per weight, 12.0 GB in total. We made them with exllamav3 1.6.0 from Qwen's BF16 release, [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), on one H200. The calibration recipe is in `calibration/`, and the raw quality output is in `quality/`.

We serve them with [TandemLLM](https://github.com/0xBakeer/TandemLLM), our inference engine for one DGX Spark, with speculative decoding and StairCut on top.

## Quality and speed of every size

We measured quality against the BF16 model on wikitext-2, where BF16's perplexity is 6.849. Speed is tokens per second on one DGX Spark in TandemLLM 0.6.2: one request at a time, greedy, 512 tokens, thinking off, speculative decoding on.

| size | GB | KL vs BF16 | top-1 agreement | prose | chat | code |
|-|-|-|-|-|-|-|
| **2.5** | 12.0 | 0.1112 | 86.3 % | 56.2 | 89.0 | 124.2 |
| [3.0](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-3.0) | 13.5 | 0.0621 | 89.8 % | 56.9 | 69.6 | 119.1 |
| [3.5](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-3.5) | 15.2 | 0.0298 | 93.1 % | 47.2 | 74.8 | 113.6 |
| [4.0](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-4.0) | 16.9 | 0.0171 | 94.8 % | 45.6 | 53.4 | 89.1 |
| [4.5](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-4.5) | 18.4 | 0.0096 | 96.1 % | 41.2 | 52.5 | 84.4 |
| [6.0](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-6.0) | 23.3 | 0.0018 | 98.5 % | 33.2 | 39.8 | 67.7 |

In our check of 12 prompts per size, every answer with speculative decoding matched plain greedy decoding, or left it only at a near-tie between two tokens.

## How to run it

In TandemLLM, switch to this size with `ops/use-profile.sh exl3-2.5`. Since TandemLLM 0.8.1 the script downloads everything this size needs, checks every file against the hub and restarts the engine: this quant, the [EXL3 drafters](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-EXL3-drafters), our BF16 drafters and the base checkpoint. All six sizes are in the [collection](https://huggingface.co/collections/0xBakeer/tandemllm-qwen38-27b-exl3-6acdd787008264fa4f35eab4), and quants made by others are listed as [community weights](https://github.com/0xBakeer/TandemLLM/blob/main/docs/community-weights.md).

The files are in the standard EXL3 format, so other EXL3 loaders such as exllamav3 and TabbyAPI should read them too. We have not tested those.

Everything else about the model, and its Apache 2.0 license, is Qwen3.8-27B's. See [its model card](https://huggingface.co/Qwen/Qwen3.8-27B).
