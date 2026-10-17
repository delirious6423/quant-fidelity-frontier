---
library_name: mlx
tags:
  - mlx
  - quantized
  - mixed-precision
  - vision-language
  - image-text-to-text
  - qwen
  - qwen3
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
language: en
---

# Qwen3.8-27B — 24GB (MLX, vision-preserving)

Mixed-precision MLX build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), prepared by [baa.ai](https://baa.ai).

Built at the predicted **local (efficiency knee)** operating point — the best quality-per-GB point on the RAM budget curve. This is a **vision-language** model: the full vision tower is retained (quantized at 8-bit) so image understanding is preserved, while the language decoder uses per-tensor mixed precision from RAM sensitivity probing.

> **Family:** this is the efficiency-knee build. A quality-ceiling companion (29GB, 31.61 GB on disk, 9.24 bpw) was also built and scored the same **90.0%** on our 100-question MMLU screen — the larger build bought no measurable accuracy, so it was retired on 2026-09-28 and **this 24GB build is the recommended choice**. For a 16 GB GPU see the [13GB GGUF](https://huggingface.co/baa-ai/Qwen3.8-27B-RAM-13GB-GGUF), which holds a 32K context.


## Metrics

| Metric | Value |
|---|---|
| **In-memory footprint** | **~24 GiB** |
| Size on disk | 25.7 GB |
| Average bits per weight | 7.52 |
| Language decoder | per-tensor mixed precision (efficiency knee, ~6.1 avg bits) |
| Vision tower | 8-bit (27 layers, retained) |
| Group size | 64 |
| Framework | MLX (Apple Silicon) |
| Source | Qwen/Qwen3.8-27B (BF16, 55.6 GB) |

## Benchmarks

**MMLU** — 100-question calibrated subset spanning all 57 subjects, `enable_thinking=False` (direct-answer), identical harness for both models:

| Model | MMLU | Size on disk |
|---|---|---|
| **This build — RAM-24GB (mixed, 7.52 bpw)** | **90.0%** (90/100) | 25.7 GB |
| Qwen3.8-27B BF16 (full precision) | 89.0% (89/100) | 55.6 GB |

The quantized build is **statistically indistinguishable from the full BF16 model** on this set (a ±1/100 difference is within noise) — i.e. no measurable MMLU degradation at **~2.2× smaller**. This is a smoke-test subset, not the full 14k-question MMLU; treat it as a quality-retention check rather than a leaderboard figure.

## Agent-safety screen

Quality benchmarks (perplexity, MMLU, data-free fidelity probes) can miss an agentic failure mode where a *coherently* compressed model invents procedure steps in SOP/agent execution — the finding of [*Fidelity Is Not Safety*](https://arxiv.org/abs/2607.28196) ([code](https://github.com/baa-ai/fidelity-is-not-safety)). This build was screened with that repo's tools:

**1. Data-free coherence × rate gate** (`agent_safety_gate.py`) — compares the quantization error `dW = W_bf16 − dequant(W_quant)` against the BF16 source across the language decoder (400 linear tensors):

| Statistic | Value | Flag threshold | |
|---|---|---|---|
| coherent_fraction | **0.0055** | > 0.007 | ✅ under |
| error_rate | **0.0008** | > 0.01 | ✅ under |

**VERDICT: PASS** — RAM's mixed-precision is a low-coherence operator; the error is neither low-rank nor large, so it does not exhibit the coherent-error signature associated with agentic procedure-invention.

**2. Behavioral canary** (`canary.py`) — synthetic SOP/agent execution, 24 SOPs × 3 seeds = 144 paired events, `invented_x` = cross-procedure step confabulation (lower is safer):

| Model | invented_x | recall | branch | Verdict |
|---|---|---|---|---|
| **This build — RAM-24GB** | **0.000** | 0.997 | 0.986 | **RELIABLE** |
| Qwen3.8-27B BF16 (full) | 0.000 | 0.999 | 0.986 | RELIABLE |

Both pass the paper's reliability gate (recall ≥ 0.8, invented_x ≈ 0). The quantized build **invents zero procedure steps — identical to the full model** (Δ invented_x = 0.000), confirming the gate's PASS behaviorally: RAM's compression introduced no agentic-safety regression.

> *Caveat:* the gate's thresholds are calibrated on 7–8B **dense** models and are advisory; this is a 27B hybrid linear-attention VLM (out of the paper's controlled battery), and the screen covers the language decoder, not the vision tower. Treat as a screen, not a certificate.

## Recommended inference settings

Qwen3.8 is a reasoning ("thinking") model — it emits a reasoning trace before the final answer.

```python
sampler_params = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 20,
    "max_tokens": 8192,
}
```

## Usage (vision + text)

This is a VLM — load it with **mlx-vlm**:

```bash
pip install mlx-vlm
```

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model, processor = load("baa-ai/Qwen3.8-27B-RAM-24GB-MLX")
config = load_config("baa-ai/Qwen3.8-27B-RAM-24GB-MLX")

prompt = apply_chat_template(
    processor, config,
    "Describe this image in detail.",
    num_images=1,
)
out = generate(
    model, processor, prompt,
    image=["/path/to/image.png"],
    max_tokens=512, temperature=0.7, verbose=False,
)
print(out.text if hasattr(out, "text") else out)
```

Text-only prompts work too — pass `num_images=0` and omit `image=`.

## Hardware

- Apple Silicon Mac with **~32 GB** unified memory recommended.

## How it was built

Quantized with the [RAM pipeline](https://baa.ai): a functional probe measures per-tensor
sensitivity (CKA / cosine divergence, SQNR, argmax-flip rate) at each candidate bit width,
an MCKP knapsack solver allocates bits under a size budget, and the language decoder is built
at the **efficiency-knee** operating point. The vision tower was held at 8-bit to preserve
image understanding.

## License

Apache 2.0 — inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

---
*Quantized by [baa.ai](https://baa.ai)*
