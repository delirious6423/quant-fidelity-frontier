---
base_model:
- Qwen/Qwen3.8-27B
- Qwen/Qwen3.8-27B-FP8
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- fp8
- vllm
- speculative-decoding
---

# Qwen3.8-27B-FP8 with a fused RMS kernel

The parameters are the public checkpoint [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) at revision `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`. They are unchanged. This repository adds the fused add + RMS + UE8M0 kernel and the speculative-decoding measurements from 16 Nov 2026 on one NVIDIA H200 NVL, vLLM 0.29.0, tensor parallel 1.

License: Apache 2.0, the same license as the upstream weights. The upstream model card is [`README_UPSTREAM.md`](README_UPSTREAM.md).

## Kernel

`kernel/` contains the CUDA source, the runtime, and a patch for vLLM 0.29.0. The patch does nothing unless both variables are set:

- `XIRCUIT_RMS_UE8M0=1`
- `XIRCUIT_RMS_UE8M0_PATH` pointing at the `kernel/` directory

For K=5120 rows of at least 256, the kernel fuses the residual add, RMS norm, and UE8M0 block quant, and DeepGEMM consumes that FP8 activation. It does not cover the down projection (K=17408).

Isolated, bit-identical to CUDA `fused_add_rms_norm` plus column-major UE8M0 quant, 100 reps:

| Rows | Split | Fused | Cold gain |
| ---: | ---: | ---: | ---: |
| 8192 | 0.201 ms | 0.110 ms | 45% |
| 2048 | 0.056 ms | 0.035 ms | 37% |

The kernel uses the CUDA bf16-weight contract. This server compiles Gemma RMS with an fp32 weight, so hooked generations are not the same tokens as stock MTP-4.

## Speculative decoding

MTP-4 is the serving configuration. Stock is the reference. Raw `result.json` files, arXiv output ids, GSM8K scores, and the prefill census are under `results/`. The arXiv article text is not included.

### ShareGPT chat, 504 requests, prefix cache on

| Arm | Wall | Output tok/s | Acceptance | Prefix hit |
| --- | ---: | ---: | ---: | ---: |
| Stock | 47.6 s | 138.2 | — | 60.3% |
| MTP-3 | 44.2 s | 148.7 | 3.00 | 16.7% |
| DFlash, 7 tokens | 46.6 s | 141.1 | 2.99 | 15.8% |
| DSpark, 8 tokens | 47.7 s | 137.8 | 3.17 | 15.8% |

### 256 output tokens, same 504 chats

| Arm | Wall | Output tok/s | Acceptance |
| --- | ---: | ---: | ---: |
| Stock | 236.0 s | 547 | — |
| MTP-4 | 133.9 s | 964 | 3.60 |
| MTP-5 | 136.9 s | 942 | 3.89 |
| MTP-8 | 163.8 s | 788 | 4.40 |
| DFlash, 7 tokens | 210.5 s | 613 | 2.22 |
| DSpark, 8 tokens | 211.0 s | 611 | 2.68 |

MTP-4 is 1.76× stock. With the v2 model runner off, MTP-4 is 947 tok/s. On 64 short prompts decoded for 512 tokens, stock is 3,362 tok/s and MTP-4 is 3,413 tok/s.

### GSM8K, temperature 0, 1,319 items

| Arm | Strict | Flexible | Wall | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Stock | 1,203 / 1,319 (91.2%) | 1,214 | 89.04 s | — |
| MTP-4 | 1,206 / 1,319 (91.4%) | 1,214 | 89.03 s | 4.29 |

Strict McNemar p = 0.743 (both correct 1,186, stock only 17, MTP-4 only 20). Flexible counts match, but 19 items swap each way (p = 1.000).

### arXiv 1,000, 6,142,812 prompt tokens, 32 output tokens, prefix cache off

| Arm | Reps | Mean wall | Stdev | Prompt tok/s | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock | 4 | 427.00 s | 0.93 s | 14,386 | — |
| MTP-4 | 4 | 408.50 s | 0.42 s | 15,037 | 3.450 |
| MTP-4 + fused kernel | 3 | 402.59 s | 0.27 s | 15,258 | 3.490 |

MTP-4 is 18.5 s (4.3%) ahead of stock. The fused kernel is another 5.9 s (1.45%) ahead of MTP-4. Every hooked run beat every MTP-4 run by at least 5.4 s.

Token agreement on the 1,000 summaries:

| Comparison | Full sequence | First token |
| --- | ---: | ---: |
| MTP-4 vs stock | 508 or 512 | 892 or 894 |
| Fused kernel vs MTP-4 | 482 | 893 |
| Fused kernel vs stock | 504 | 879 |

The four MTP-4 runs match each other. The three hooked runs match each other. Hook counters on every hooked run: 101,400 fused RMS calls, 99,840 consumed by DeepGEMM, 1,447 pointer misses, 17,116 small-batch fallbacks.
