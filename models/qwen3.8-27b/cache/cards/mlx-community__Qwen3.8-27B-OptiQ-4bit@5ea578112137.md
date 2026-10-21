---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- quantized
- mixed-precision
- 4bit
- 8bit
- optiq
- apple-silicon
- image-text-to-text
- vision-language
- qwen3.8
---

# mlx-community/Qwen3.8-27B-OptiQ-4bit

> **Built with [mlx-optiq](https://mlx-optiq.com)**, the MLX-native toolkit to quantize, fine-tune, and serve LLMs locally on Apple Silicon, no PyTorch and no cloud. [Try the Lab](https://mlx-optiq.com/docs/lab/) · [All OptiQ quants](https://mlx-optiq.com/models) · [Docs](https://mlx-optiq.com/docs/)

A 4-bit mixed-precision MLX quant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), an image+text model. Per-layer bit-widths come from a KL-divergence sensitivity pass on a [six-domain calibration mix](https://mlx-optiq.com/blog/calibration-mix) (prose · reasoning · code · agent · tool-call · constraint-bearing instructions). Sensitive layers go to 8-bit; robust ones stay at 4-bit.

The vision tower is kept at bf16 in a sidecar, so this repo loads text-only under stock `mlx-lm` and image+text under OptiQ.

## Quantization details

| Property | Value |
|---|---|
| Predominant precision | 4-bit |
| Layers at 8-bit (sensitive) | 261 |
| Layers at 4-bit (robust) | 237 |
| Total quantized layers | 498 |
| Group size | 64 |
| Vision tower | bf16, `optiq/optiq_vision.safetensors` (0.9 GB) |
| Calibration mix | [six-domain mix](https://mlx-optiq.com/blog/calibration-mix) (40 samples × 6 domains) |
| Reference for sensitivity | uniform-4-bit (bf16 at this size does not fit 36 GB) |
| On-disk size | 19.0 GB |

Qwen3.8-27B is a hybrid: three quarters of its 64 blocks use linear attention (five projections each) and the rest full attention, which is why the sweep scores 498 tensors rather than the ~450 a dense model of this size would have.

We follow the same naming convention `llama.cpp` uses for Q4_K_M and similar mixed-precision quants: the "4-bit" label is for the predominant precision, not the weighted average.

## Usage

Load it with `mlx-lm` for text:

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3.8-27B-OptiQ-4bit")
response = generate(
    model, tokenizer,
    prompt="Explain quantum computing in simple terms.",
    max_tokens=200,
)
```

For image+text, mixed-precision KV-cache serving, sensitivity-aware LoRA fine-tuning, an OpenAI + Anthropic-compatible inference server, hot-swap mounted adapters and sandboxed Python execution, install [`mlx-optiq`](https://mlx-optiq.com/):

```bash
pip install mlx-optiq
optiq serve --model mlx-community/Qwen3.8-27B-OptiQ-4bit
```

See the [Qwen family guide](https://mlx-optiq.com/docs/qwen) on [mlx-optiq.com](https://mlx-optiq.com/) for sampling defaults, training recipes, and family-specific caveats.

## Benchmarks

| Metric | Score |
|---|---|
| MMLU (5-shot, 1000) | 84.1% |
| GSM8K (3-shot CoT) | 96.1% |
| IFEval (prompt-level, strict) | 77.1% |
| BFCL-V3 simple (AST) | 93.5% |
| HumanEval (pass@1) | 92.1% |
| HashHop (long-context retrieval) | 85.0% |
| **Capability Score** (mean of 6) | **87.98** |

18.9 GB on disk.

Every metric gets one equal vote, and disk size is reported next to the score as
an honest second axis rather than folded into it. See the
[eval-framework writeup](https://mlx-optiq.com/blog/eval-framework) for the
methodology.

## Links

- **Project website:** [mlx-optiq.com](https://mlx-optiq.com/)
- **PyPI:** [pypi.org/project/mlx-optiq](https://pypi.org/project/mlx-optiq/)
- **Calibration mix:** [mlx-optiq.com/blog/calibration-mix](https://mlx-optiq.com/blog/calibration-mix)
- **Eval framework:** [mlx-optiq.com/blog/eval-framework](https://mlx-optiq.com/blog/eval-framework)
- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

## Quantize your own

This quant was produced by [mlx-optiq](https://mlx-optiq.com). Point it at any Hugging Face model to get the same sensitivity-aware mixed precision:

```bash
pip install mlx-optiq
optiq convert Qwen/Qwen3.8-27B --target-bpw 5.0 --candidate-bits 4,8
optiq lab   # or drive the whole thing from the browser
```
