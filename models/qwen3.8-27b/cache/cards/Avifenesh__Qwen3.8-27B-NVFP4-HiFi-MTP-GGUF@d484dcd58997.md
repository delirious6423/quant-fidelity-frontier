---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: Avifenesh
pipeline_tag: text-generation
tags:
  - nvfp4
  - gguf
  - mtp
  - speculative-decoding
  - blackwell
  - qwen3
  - conversational
---

# Qwen3.8-27B NVFP4 HiFi GGUF, with MTP head

The high-fidelity NVFP4 build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). 16.59 GB. Same speed as a mixed NVFP4/Q6_K file, closer to BF16 than any NVFP4 build we measured. The MTP head is in the file.

The standard build is [Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF](https://huggingface.co/Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF) (15.71 GB). Take this one when you have the extra 0.9 GB.

File: `Qwen3.8-27B-NVFP4-HiFi-mtp.gguf`, 16,587,252,768 bytes, sha256 `972390ab1c6cdee39e572f36991ab7e9a4cd63cfdff519947a64f69cfcc17e07`.

## Requirements

- A llama.cpp build that reads per-tensor `.scale` tensors: mainline `969d733` or newer. The weights are stored divided by their scale, so a build that skips the scales gives wrong outputs.
- NVFP4 kernels: Blackwell (sm_100/sm_120/sm_121). Other GPUs fall back to slower paths.

## Recipe

1. BF16 GGUF from the official checkpoint.
2. Every trunk NVFP4 weight gets a per-tensor FP32 scale, a power of two: `alpha = 2^ceil(log2(amax / (6 x 256)))`. The weight is stored as `W / alpha` (exact in BF16) plus a sibling F32 `<name>.scale`. This is the NVFP4 tensor scale with the 256 limit from Four Over Six (arXiv 2512.02010). Without it, llama.cpp's NVFP4 block scales are tiny E4M3 values: 99.5% of them are subnormals with 2 to 3 bits of precision. With it, none are.
3. llama-quantize NVFP4 with an imatrix-weighted block-scale search (llama.cpp PR #25153, 25 E4M3 codes per block). The imatrix is 204,800 tokens of wikitext-2 train plus C source, disjoint from the eval texts.
4. Q6_K (imatrix) where it pays most per byte, from per-layer probes and a knapsack at 16.6 GB:
   - full-attention q/k/v/o and linear-attention `ssm_out` in layers 0-15, 20-31 and 44-55;
   - `ffn_up` and `ffn_down` in layers 56-63.
5. Embedding and LM head Q5_K. MTP block NVFP4, unchanged.

Four Over Six itself adds nothing here. Its amax/4 scale is one of the 25 codes the search already tries, and the output is byte-identical with or without it.

## Fidelity vs BF16

KL divergence of the full next-token distribution against BF16 logits. Wikitext-2 test, 72 chunks of 4,096 tokens (147,384 scored). Code: llama.cpp `src/`, 40 chunks (81,880 scored). Mean KLD is driven by a few positions, so the median and top-1 are the steadier columns.

llama.cpp `969d733`, default path (NVFP4 weights with 4-bit activations on Blackwell):

| file | GB | wiki mean KLD | wiki median | wiki top-1 % | code mean KLD | code top-1 % |
|---|---|---|---|---|---|---|
| standard (NVFP4-MTP-GGUF) | 15.71 | 0.0843 | 0.0229 | 89.96 | 0.0323 | 96.08 |
| NVFP4 + Q6_K attention (mixC) | 16.53 | 0.0763 | 0.0194 | 90.59 | 0.0257 | 96.51 |
| **this file** | **16.59** | **0.0647** | **0.0141** | **91.95** | **0.0206** | **96.86** |
| unsloth UD-Q4_K_M | 16.56 | 0.0254 | 0.0030 | 95.97 | 0.0057 | 98.31 |

memra (our engine; prefill with 8-bit activations, NVFP4 weights):

| file | wiki mean KLD | wiki median | wiki top-1 % | code mean KLD | code top-1 % |
|---|---|---|---|---|---|
| standard | 0.0570 | 0.0131 | 92.23 | 0.0189 | 97.00 |
| mixC | 0.0521 | 0.0113 | 92.72 | 0.0155 | 97.25 |
| **this file** | **0.0415** | **0.0079** | **93.87** | **0.0109** | **97.58** |

UD-Q4_K_M is closer to BF16. It is also 26% slower to prefill in llama.cpp on Blackwell (below). NVFP4's 4-bit activation path costs fidelity; an engine with 8-bit activations recovers about a third of the KL.

## Size and speed

One DGX Spark (GB10), llama.cpp `969d733`, `llama-bench -ngl 99 -fa 1 -r 5`:

| file | GB | pp512 | pp4096 | tg128 |
|---|---|---|---|---|
| standard | 15.71 | 1508 | 1502 | 14.7 |
| this file | 16.59 | 1295 | 1298 | 13.8 |
| UD-Q4_K_M | 16.56 | 959 | 963 | 13.0 |

## MTP

The embedded MTP head works as a draft (K=3). It is byte-identical to the standard build's, as are the embedding and the LM head. memra, greedy, chat template, 256 new tokens per prompt, one GB10:

| file | acceptance, 8 prompts | acceptance, 6 more prompts | all 14 | MTP speedup over plain decode |
|---|---|---|---|---|
| standard | 67.1% (1370/2043) | 71.4% (1048/1467) | 68.9% | 2.08x / 2.30x |
| this file | 64.1% (1350/2106) | 72.7% (1054/1449) | 67.6% | 2.14x / 2.31x |

The 8 prompts are code, chat, math and agentic; the 6 more add long code and agentic contexts (up to 5,420 tokens), prose and structured output. Greedy runs are deterministic, so the spread is prompt to prompt: per-prompt differences run from -11 to +8 points, and the 14-prompt difference (-1.3 points) is within that noise. With the standard build, MTP took decode from 14.3 to 31.3 tok/s on an idle GB10.

## Not measured

Contexts beyond 4,096 tokens in the KL runs. Task benchmarks. Non-Blackwell GPUs.
