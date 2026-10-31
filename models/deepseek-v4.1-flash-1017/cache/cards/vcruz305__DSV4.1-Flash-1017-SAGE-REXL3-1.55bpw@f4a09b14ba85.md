---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - rexl3
  - quantized
  - sage
  - deepseek-v4.1
---

# DeepSeek-V4.1-Flash-1017 — SAGE REXL3 1.55 bpw

Loadable REXL3 / ExLlamaV3 pack of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), encoded with the native `REXL3` CUDA quantizer and contiguous per-expert record layout (`w1, w2, w3` × `trellis, mul1, suh, svh`).

Quantized by vcruz305 with **SAGE**, our own mixed-precision quantization method for EXL3. Tensors use different EXL3 `K` values; the kernels read `K` from each trellis tensor at load time. The method is not published.

## Pack Summary

| Property | Value |
| :--- | :--- |
| Base model | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Routed expert average bitwidth | `1.5462 bpw` (`98.50 GiB` across 40 layers × 384 experts) |
| Non-routed `wq_b` / `wo_b` / `shared_experts` | `K=5/6` EXL3 (`3.09 GiB`) |
| MTP speculative draft (`mtp.0..2`) | `1.50 bpw` mixed `K1/K2/K3` (`2.39 GiB`, Shard 15) |
| LM head (`head`) | `K=5` EXL3 (`0.386 GiB`) + final RMSNorm (`norm.weight`) |
| Engram tables (`layers.1`, `layers.14`) | Native int8 + scale (`201.18 GiB`, Shards 16–17) |
| Total GPU-resident footprint | `108.52 GiB` Text-Only / `109.42 GiB` VLM (fits 1× DGX Spark 128 GiB at 262K context) |
| Shards | `17` (`model-00001-of-00017.safetensors` .. `model-00017-of-00017.safetensors`) |

## Canonical `float64` Fidelity vs. Teacher (`129,280` Vocab)

| Split | Tokens | Top-1 Agreement | Top-5 Agreement | Top-5 Overlap | KL Mean | KL Median (`p50`) | KL `p95` | Teacher PPL | Student PPL | $\Delta$ PPL |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e2e_eval` (Non-Header) | 29,492 | **93.48%** | **99.46%** | 80.64% | `0.1245` | `0.000237` | `0.6608` | `1.9981` | `1.9736` | `-0.0245` |
| `e2e_eval` (Assistant Turn) | 21,458 | **93.95%** | **99.53%** | 81.01% | `0.1058` | `0.000163` | `0.5398` | `1.8575` | `1.8653` | `+0.0078` |
| `e2e_eval` (Answer Only) | 9,305 | **92.90%** | **99.41%** | 78.02% | `0.1499` | `0.000044` | `0.8296` | `1.5794` | `1.6254` | `+0.0461` |
| `wiki1100` (Non-Header) | 1,016 | **89.57%** | **99.02%** | 74.29% | `0.2273` | `0.001960` | `1.3351` | `2.6041` | `2.7162` | `+0.1121` |
| `long17k` (17,408-tok Non-Header) | 15,700 | **92.18%** | **99.28%** | 77.42% | `0.1878` | `0.000212` | `1.1103` | `2.0717` | `1.9378` | `-0.1339` |
