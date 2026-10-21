---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
tags:
  - gguf
  - quantized
  - apex
  - moe
  - mixture-of-experts
  - liquidai
  - lfm2
  - hybrid
---

# LFM2.5-24B-A2B APEX GGUF

**APEX (Adaptive Precision for EXpert Models)** quantizations of [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) by LiquidAI.

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team** | [APEX Project](https://github.com/mudler/apex-quant) | [Technical Report](https://github.com/mudler/apex-quant/blob/main/paper/APEX_Technical_Report.pdf)

## Benchmark Results

Perplexity is measured on wikitext-2-raw. KL divergence is measured against the BF16 model on the same text. HellaSwag and Winogrande use 400 tasks each, zero-shot. The LiquidAI rows are the quantizations published by LiquidAI, measured the same way. Perplexity differences of about 0.2 are close to the measurement error (+/- 0.14).

| Model | Size (GiB) | Perplexity | KL vs BF16 (mean) | HellaSwag | Winogrande |
|---|---|---|---|---|---|
| APEX I-Quality | 14.3 | 15.99 | 0.077 | 69.5 | 61.8 |
| APEX Quality | 14.3 | 16.36 | 0.083 | 67.2 | 60.2 |
| APEX I-Balanced | 16.1 | 15.93 | 0.060 | 67.5 | 62.5 |
| APEX Balanced | 16.1 | 16.20 | 0.065 | 66.2 | 62.0 |
| APEX I-Compact | 10.4 | 16.20 | 0.219 | 66.8 | 62.8 |
| APEX Compact | 10.4 | 16.83 | 0.266 | 65.2 | 65.0 |
| APEX I-Mini | 8.5 | 16.43 | 0.436 | 63.0 | 62.2 |
| APEX I-Nano | 7.1 | 17.55 | 0.567 | 62.2 | 63.8 |
| LiquidAI Q8_0 | 23.7 | 15.94 | n/a | 67.0 | 63.0 |
| LiquidAI Q5_K_M | 15.8 | 15.70 | 0.092 | 68.0 | 62.5 |
| LiquidAI Q4_K_M | 13.5 | 15.94 | 0.164 | 65.5 | 63.0 |

Only these benchmarks were run. No MMLU, ARC or TruthfulQA numbers are claimed.

## What is APEX?

APEX is a quantization strategy for Mixture-of-Experts (MoE) models. It classifies tensors by role (routed expert, shared expert, attention) and applies a layer-wise precision gradient: edge layers get higher precision, middle layers get more aggressive compression. The I-variants use imatrix-guided quantization.

See the [APEX project](https://github.com/mudler/apex-quant) for details and the technical report.

## Files

| File | Size |
|---|---|
| LFM2.5-24B-A2B-APEX-I-Quality.gguf | 14.3 GiB |
| LFM2.5-24B-A2B-APEX-Quality.gguf | 14.3 GiB |
| LFM2.5-24B-A2B-APEX-I-Balanced.gguf | 16.1 GiB |
| LFM2.5-24B-A2B-APEX-Balanced.gguf | 16.1 GiB |
| LFM2.5-24B-A2B-APEX-I-Compact.gguf | 10.4 GiB |
| LFM2.5-24B-A2B-APEX-Compact.gguf | 10.4 GiB |
| LFM2.5-24B-A2B-APEX-I-Mini.gguf | 8.5 GiB |
| LFM2.5-24B-A2B-APEX-I-Nano.gguf | 7.1 GiB |

## Architecture

- **Model**: LFM2.5-24B-A2B (lfm2_moe) by LiquidAI
- **Layers**: 40 (30 convolutional + 10 full attention, hybrid), the first 2 dense
- **Experts**: 64 routed, 4 active per token
- **Total parameters**: about 24B
- **Active parameters**: about 2B per token

## Run with LocalAI

```bash
local-ai run mudler/LFM2.5-24B-A2B-APEX-GGUF@LFM2.5-24B-A2B-APEX-I-Balanced.gguf
```

## Credits

APEX is brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team. Developed through human-driven, AI-assisted research. Built on [llama.cpp](https://github.com/ggerganov/llama.cpp).
