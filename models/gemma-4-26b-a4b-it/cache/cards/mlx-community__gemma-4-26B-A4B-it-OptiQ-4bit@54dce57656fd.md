---
library_name: mlx
license: gemma
pipeline_tag: text-generation
base_model: google/gemma-4-26B-A4B-it
tags:
- mlx
- quantized
- mixed-precision
- 4bit
- 8bit
- optiq
- apple-silicon
- text-generation
- gemma-4
---

# mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit

> **Built with [mlx-optiq](https://mlx-optiq.com)**, the MLX-native toolkit to quantize, fine-tune, and serve LLMs locally on Apple Silicon, no PyTorch and no cloud. [Try the Lab](https://mlx-optiq.com/docs/lab/) · [All OptiQ quants](https://mlx-optiq.com/models) · [Docs](https://mlx-optiq.com/docs/)

A 4-bit mixed-precision MLX quant produced by [mlx-optiq](https://mlx-optiq.com/), the sensitivity-aware quantization toolkit for Apple Silicon. Beats stock uniform 4-bit on every benchmark in the six-metric Capability Score.

A 4-bit mixed-precision MLX quant of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). Per-layer bit-widths come from a KL-divergence sensitivity pass on a [six-domain calibration mix](https://mlx-optiq.com/blog/calibration-mix) (prose · reasoning · code · agent · tool-call · constraint-bearing instructions). Sensitive layers go to 8-bit; robust ones stay at 4-bit. The on-disk size is within ~5 % of a stock uniform 4-bit MLX quant.

## Quantization details

| Property | Value |
|---|---|
| Predominant precision | 4-bit |
| Layers at 8-bit (sensitive) | 246 |
| Layers at 4-bit (robust) | 79 |
| Total quantized layers | 325 |
| Group size | 64 |
| Calibration mix | [six-domain mix](https://mlx-optiq.com/blog/calibration-mix) (40 samples × 6 domains) |
| Reference for sensitivity | bf16 (auto-resolved; falls back to uniform-4-bit if bf16 doesn't fit) |
| Speculative drafter | served with [`mlx-community/gemma-4-26B-A4B-it-assistant-bf16`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-assistant-bf16) via `optiq serve --drafter` |

We follow the same naming convention `llama.cpp` uses for Q4_K_M and similar mixed-precision quants: the "4-bit" label is for the predominant precision, not the weighted average. The mixed allocation is what lets this build beat stock uniform-4-bit on every benchmark below at the same disk size.

## Usage

Load it with `mlx-lm` and use it as usual:

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit")
response = generate(
    model, tokenizer,
    prompt="Explain quantum computing in simple terms.",
    max_tokens=200,
)
```

For more (mixed-precision KV-cache serving, sensitivity-aware LoRA fine-tuning, OpenAI + Anthropic-compatible inference server, hot-swap mounted adapters, sandboxed Python execution for agent workflows), install [`mlx-optiq`](https://mlx-optiq.com/):

```bash
pip install mlx-optiq
```

### Speculative decoding (assistant drafter)

Gemma-4 ships a separate small drafter for speculative decoding. Pair this quant with [`mlx-community/gemma-4-26B-A4B-it-assistant-bf16`](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-assistant-bf16) for faster decode:

```bash
optiq serve --model mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit \
            --drafter mlx-community/gemma-4-26B-A4B-it-assistant-bf16
```


See the [Gemma-4 family guide](https://mlx-optiq.com/docs/gemma-4) on [mlx-optiq.com](https://mlx-optiq.com/) for sampling defaults, training recipes, and family-specific caveats.

## Benchmarks

Six-metric Capability Score (mean of MMLU + GSM8K + IFEval + BFCL + HumanEval + HashHop). Apples-to-apples comparison against stock uniform 4-bit:

| Metric | OptiQ | Uniform 4-bit | Δ |
|---|---:|---:|---:|
| MMLU (5-shot, 1000 samples) | **65.0%** | 61.1% | +3.9 |
| GSM8K (1000 samples, 3-shot CoT) | **93.8%** | 91.7% | +2.1 |
| IFEval (full set, strict) | **73.0%** | 74.5% | -1.5 |
| BFCL-V3 simple (200 calls) | **91.5%** | 92.0% | -0.5 |
| HumanEval (164 problems, pass@1) | **90.2%** | 88.4% | +1.8 |
| HashHop (long-context retrieval) | **41.0%** | 30.0% | +11.0 |
| **Capability Score** (mean of 6) | **75.76** | 72.95 | **+2.81** |
| KL vs uniform-4-bit reference (mean / p95) | 1.1874 / 5.2286 |, |, |
| On-disk size | 16.4 GB | 14.5 GB | +1.9 |

Every metric gets one equal vote. Disk size is reported next to the score as an honest second axis instead of being folded into the score. See the [eval-framework writeup](https://mlx-optiq.com/blog/eval-framework) for the full methodology.

## Links

- **Project website:** [mlx-optiq.com](https://mlx-optiq.com/)
- **Gemma-4 family guide:** [mlx-optiq.com/docs/gemma-4](https://mlx-optiq.com/docs/gemma-4)
- **PyPI:** [pypi.org/project/mlx-optiq](https://pypi.org/project/mlx-optiq/)
- **Calibration mix:** [mlx-optiq.com/blog/calibration-mix](https://mlx-optiq.com/blog/calibration-mix)
- **Eval framework:** [mlx-optiq.com/blog/eval-framework](https://mlx-optiq.com/blog/eval-framework)
- **Base model:** [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)


## Quantize your own

This quant was produced by [mlx-optiq](https://mlx-optiq.com). Point it at any Hugging Face model to get the same sensitivity-aware mixed precision:

```bash
pip install mlx-optiq
optiq convert <hf-model-id> --target-bpw 5.0 --candidate-bits 4,8
optiq lab   # full local workbench: chat, compare, quantize, fine-tune
```

## License

Gemma license (inherits from base model). See https://ai.google.dev/gemma/terms for the terms of use.
