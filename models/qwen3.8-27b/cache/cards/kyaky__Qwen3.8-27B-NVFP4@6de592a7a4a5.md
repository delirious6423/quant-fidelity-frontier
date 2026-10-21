---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - qwen
  - vision-language
  - nvfp4
  - vllm
  - blackwell
pipeline_tag: image-text-to-text
library_name: transformers
---

# Qwen3.8-27B-NVFP4

A competitive, efficient NVFP4 build that matches or leads the alternative build across the measured benchmarks, stays measurably closer to the original BF16 model, and delivers faster single-stream generation on the tested hardware.

## Highlights

- Matches or leads `unsloth/Qwen3.8-27B-NVFP4` across CMMLU, C-Eval, and non-thinking MMLU-Pro, with MMLU-Pro essentially at parity.
- Records 0.0265 KL divergence from BF16 versus 0.0391 for the competing build—about 32% lower and measurably closer to the original BF16 output distribution.
- Reaches 61.2 tokens/s single-stream versus 58.5 for the competing build and 27.4 for BF16 on the tested configuration—about 2.2× BF16.
- Delivers near-parity aggregate throughput at 32 concurrent requests: 1284 tokens/s versus 1292; higher concurrency tends to favor the competing build's FP4 kernel path.
- Downloads at 23.29 GB versus 23.4 GB for the competing build.
- NVFP4 4-bit weights for vLLM serving on NVIDIA Blackwell GPUs.

![Accuracy, fidelity, and size comparison](compare-qwen3.8-27b-nvfp4.svg)

## Results

| Metric | BF16 baseline | `kyaky/Qwen3.8-27B-NVFP4` | `unsloth/Qwen3.8-27B-NVFP4` | Result |
|---|---:|---:|---:|---|
| CMMLU accuracy ↑ | 72.03 | **73.55** | 70.76 | +2.80 pp vs competitor |
| C-Eval accuracy ↑ | 74.07 | **75.93** | 73.63 | +2.30 pp vs competitor |
| MMLU-Pro accuracy ↑, non-thinking | 75.43 | **77.14** | 76.57 | Essentially at parity with competitor |
| KL divergence vs BF16 ↓ | 0 (reference) | **0.0265** | 0.0391 | About 32% lower than competitor |
| Download size ↓ | — | **23.29 GB** | 23.4 GB | 0.11 GB smaller |

All comparisons use the same evaluation conditions. Small differences from the BF16 reference should be treated as evaluation variation, not as evidence that this build surpasses the base model's capabilities.

## Performance (measured on 1 × NVIDIA RTX PRO 6000 only)

Measured on 1 × NVIDIA RTX PRO 6000 (Blackwell) using vLLM 0.25.1 with greedy decoding and an FP8 KV cache at 8K context. Workload: 256-token generations; single-stream is the mean of 3 runs, concurrent is 32 simultaneous requests. Values are tokens/second and apply only to this tested configuration; they should not be assumed to generalize to other hardware or serving setups.

| Metric | BF16 original | `kyaky/Qwen3.8-27B-NVFP4` | `unsloth/Qwen3.8-27B-NVFP4` | Result |
|---|---:|---:|---:|---|
| Single-stream generation speed (tokens/s) ↑ | 27.4 | **61.2** | 58.5 | +4.6% vs competitor; about 2.2× BF16 |
| Aggregate throughput at 32 concurrent requests (tokens/s) ↑ | 648 | 1284 | 1292 | Near-parity at this concurrency; both roughly 2× BF16; higher concurrency tends to favor the competing build's FP4 kernel path |

## Serve with vLLM

```bash
vllm serve kyaky/Qwen3.8-27B-NVFP4 --trust-remote-code
```

NVFP4 requires an NVIDIA Blackwell GPU, such as an RTX 50-series or RTX PRO 6000-class GPU, and is served through vLLM.

## Intended use

Choose this release for a well-rounded 23.29 GB NVFP4 build with competitive benchmark retention, closer measured BF16 fidelity, and strong serving performance on the tested Blackwell configuration. It leads the competing build on CMMLU and C-Eval, remains essentially at parity on non-thinking MMLU-Pro, is faster in single-stream generation, and provides near-parity aggregate throughput at 32 concurrent requests.

---

`kyaky/Qwen3.8-27B-NVFP4` is an independent NVFP4 release derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Review the base model card for capabilities, acceptable use, and license terms.
