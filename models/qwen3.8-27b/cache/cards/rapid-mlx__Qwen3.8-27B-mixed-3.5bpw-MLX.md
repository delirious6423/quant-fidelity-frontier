---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- mlx
- apple-silicon
- multimodal
- vlm
- qwen
- quantized
- mixed-precision
- rapid-mlx
language:
- en
- zh
---

## Run it on your Mac

```bash
curl -fsSL https://rapidmlx.com/install.sh | bash
rapid-mlx serve qwen3.8-27b-mixed-3.5bpw
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}]}'
```

Use it from OpenCode / Claude Code / Codex: `rapid-mlx agents <name> --setup` · [Agent setup](https://rapidmlx.com/docs/agents)

[Download the Desktop app](https://rapidmlx.com/download) · 24 GB unified memory minimum; 32 GB gives more context headroom.

# Qwen3.8-27B · Activation-Aware Mixed 3.5bpw (MLX)


A **multimodal, mixed-precision MLX build** of
[`Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) that brings this 27B
vision-language model down to a size that runs **comfortably on a 24 GB
Apple-Silicon Mac** — while keeping **4-bit-level reasoning, knowledge, and code**.

**▶ Run it on [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)** (OpenAI-compatible, full continuous batching):

```bash
pip install rapid-mlx
rapid-mlx serve rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX --port 8000            # text
rapid-mlx serve rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX --port 8000 --mllm     # + vision
```

- **~13 GB on disk** (avg **3.5 bits-per-weight**, mixed 2/3/4-bit) — vs ~16 GB for the standard 4-bit build and ~51 GB for bf16.
- **Runs comfortably on 24 GB Macs; easy on 32 GB.** (Measured peak ~14.6–16.5 GB — see *Runtime footprint*.)
- **Native multimodal** — image + text in, text out (vision tower preserved).
- Serves on **[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)** — text with full continuous batching, or vision with `--mllm` — on Apple unified memory.

## Optional MTP acceleration

This repository also contains its paired native Multi-Token Prediction drafter
at `mtp/model.safetensors`. Recent Rapid-MLX versions can opt into lossless
speculative decoding with one self-contained download:

```bash
rapid-mlx serve qwen3.8-27b-mixed-3.5bpw \
  --speculative-config '{"method":"mtp","model":"rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX","num_speculative_tokens":3}'
```

MTP remains off by default because its speed benefit varies by Apple chip. In
paired greedy validation, emitted tokens were identical to autoregressive
decoding. The Rapid Mac app exposes the same per-model opt-in under Settings →
Performance.

The optimized MTP artifact layout builds on
[MTPLX](https://github.com/youssofal/mtplx) by Youssof Altoukhi (Apache-2.0).

![Knowledge-per-gigabyte sweet spot: mixed 3.5bpw matches 4-bit knowledge at 13 GB](assets/sweetspot.png)

## Why this build

A **uniform 3-bit** quant of this model loses **~6.6 points of factual knowledge**
(MMLU-Pro) vs 4-bit. This build **recovers essentially all of it** by spending
bits where they matter: an **activation-aware mixed-precision** allocation assigns
each language `Linear` 2/3/4-bit according to how much it actually moves the
model's outputs, rather than quantizing everything uniformly.

> Method note: this is **not** the AWQ algorithm (which does per-channel scaling at
> a uniform bit-width). It borrows AWQ's *salience metric* — activation-weighted
> quantization error — to **drive a mixed-precision bit allocation**. See below.

## How it was made

1. Run a small English calibration set through the model once, capturing the mean
   input-activation magnitude of every language `Linear` (AWQ/GPTQ-style salience).
2. Score each module's quantization error **weighted by that activation scale** —
   `|| (W − dequant(quant_b(W))) · diag(act_scale) ||` — at bit-widths {2,3,4,8}.
3. Greedy-knapsack allocate bits to hit a **3.5 bpw** target, minimizing total
   activation-weighted error under the size budget.
4. Vision tower kept at 4-bit → multimodal preserved. group size 64, affine.

The knowledge loss concentrates in the **MLP** blocks (not the output head); the
allocation promotes the most salient MLP projections to 4-bit and demotes
low-salience ones. **3.5 bpw is the sweet spot** — higher gave no further gain.

## Evaluation (identical harness, greedy decoding)

| Benchmark | Skill | 4-bit (16 GB) | **this — 3.5bpw (13 GB)** | uniform 3-bit (12 GB) |
|---|---|---|---|---|
| MMLU-Pro (n=500) | knowledge | 60.8 | **61.0** | 54.2 |
| GSM8K (n=150) | math reasoning | 95.3 | **94.7** | 92.7 |
| HumanEval+ (n=40, thinking) | code | 77.5 | **80.0** | — |
| MMBench-EN (n=300) | vision | 90.3 | 86.7 | 87.3 |

![Benchmark comparison: mixed 3.5bpw matches 4-bit on knowledge, reasoning, and code](assets/eval.png)

**Reading:** knowledge, reasoning, and code **match the 4-bit build**; the uniform
3-bit knowledge gap is fully closed. **Vision** sits at the uniform-3-bit level
(~3.6 below 4-bit) — the mixed budget was spent on language, not the vision tower.

### vs the full-precision original

Run on an **identical subset** (same 200 questions / 60 problems / 150 images),
bf16, 4-bit, and this mixed build land **within measurement noise** of each other
— at n=200 the 95% CI is ≈ ±7 points, so the 1–3 point spread is individual
questions flipping, not a quality gap:

| same subset | bf16 (51 GB) | 4-bit (16 GB) | **mixed 3.5bpw (13 GB)** |
|---|---|---|---|
| MMLU-Pro (n=200) | 59.5 | 61.5 | **60.5** |
| GSM8K (n=60) | 90.0 | 98.3 | **93.3** |
| MMBench-EN (n=150) | 88.0 | 92.0 | **86.0** |

At **13 GB** this build is **statistically indistinguishable from the bf16
original** on knowledge and reasoning — full-precision brains at ~1/4 the size.

## Runtime footprint (measured, `mlx.get_peak_memory`)

| Workload | **this (3.5bpw)** | 4-bit |
|---|---|---|
| text-only, 200 tok | **14.6 GB** | 18.6 GB |
| + image (672²) | **15.9 GB** | 19.7 GB |
| + image (896²) | **16.5 GB** | 20.4 GB |

![Peak memory: mixed 3.5bpw saves 3.9 GB vs 4-bit, fitting a 24 GB Mac](assets/memory.png)

Saves **~3.9 GB** of peak memory vs 4-bit — enough to move the model down a tier:
**4-bit needs a 32 GB Mac** (or a tuned `iogpu.wired_limit` on 24 GB), while this
build **runs on 24 GB out of the box**. Measured **~11 tok/s** decode on an
M2 Pro (32 GB); load ~5 s. Does **not** fit 16 GB.

## The sweet spot (size vs capability)

| Build | Disk | Peak (w/ image) | MMLU-Pro | Best for |
|---|---|---|---|---|
| bf16 | ~51 GB | ~53 GB | _ref_ | 64 GB+ Macs / reference |
| 4-bit | ~16 GB | ~20 GB | 60.8 | 32 GB Macs / max vision |
| **🍬 this (3.5bpw mixed)** | **~13 GB** | **~16 GB** | **61.0** | **24 GB Macs — 4-bit-level, −4 GB peak** |
| uniform 3-bit | ~12 GB | ~15 GB | 54.2 | absolute smallest, but −6.6 knowledge |

Going from 4-bit → uniform 3-bit saves ~4 GB but **drops 6.6 MMLU-Pro points**.
This build sits ~1 GB above uniform 3-bit yet **keeps all of that knowledge** — the
best capability-per-gigabyte point in the family, and the one that turns a 24 GB
Mac into a comfortable home for Qwen3.8-27B.

## Usage

Serve with **[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX)** (OpenAI-compatible API):

```bash
pip install rapid-mlx

# text — full continuous batching
rapid-mlx serve rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX --port 8000

# vision — add --mllm (serialized one-request-at-a-time lane)
rapid-mlx serve rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX --port 8000 --mllm
```

Recent Rapid-MLX also ships a short alias — `rapid-mlx serve qwen3.8-27b-mixed-3.5bpw --mllm`.

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "default",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}}
  ]}]
}'
```

## Limitations

- **Vision ~3.6 pts below 4-bit** (MMBench-EN); on par with uniform-3-bit. If you
  need maximum vision accuracy and have 32 GB, use the 4-bit build.
- **Does not fit 16 GB** — peak ~14.6 GB text-only. This is a 24 GB / 32 GB build.
- One-shot post-training quantization, no retraining.
- Inherits all capabilities and limitations of the base model.

## Attribution & license

- **Base model:** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B),
  Apache-2.0. This derivative is released under the same **Apache-2.0**.
- **Runtime:** [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) — an MLX inference server for Apple Silicon.
- Quantization method implemented from scratch in MLX (activation-aware
  mixed-precision); no third-party quantization code included.
