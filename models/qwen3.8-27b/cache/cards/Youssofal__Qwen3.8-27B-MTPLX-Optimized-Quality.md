---
license: apache-2.0
library_name: mtplx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- macos
- speculative-decoding
- multi-token-prediction
- qwen
- qwen3.8
- mtp
- mtplx
- local-ai
- coding
- qwen3-8
- qwen-3.8
- local-llm
- llm
- m5
- m5-max
- m4
- m3
- macbook-pro
- mac-studio
- opencode
- claude-code
- 27b
- qwen3.8-27b
- qwen3-8-27b
---

**[MTPLX](https://mtplx.com): the fastest way to run Qwen 3.8 on a Mac. Native multi-token-prediction speculative decoding on Apple Silicon, two to three times the speed of plain decoding, exact at any temperature.**

# Qwen 3.8 27B Optimized Quality

**8-bit dynamic quant. Good coding speeds and perfect quality.**

The closest of the three MTPLX Qwen 3.8 builds to the original bf16 model.
Every weight matrix at 8-bit, sensitive parts at 16-bit, native
multi-token-prediction head kept, so [MTPLX](https://mtplx.com) still drafts
ahead and verifies in one pass. Pick this when you want the answer the full
model would give and still want it fast.

## Measured on MTPLX 2.11.3 (16 September 2026)

This is a Qwen3.8-27B MLX pack for MTPLX, the fastest way to run Qwen 3.8 27B on a Mac. All numbers on a MacBook Pro M5 Max, fans verified at maximum, sampled at the model's own settings. Conditions and sources: [mtplx.com/benchmarks](https://mtplx.com/benchmarks/).

| Run | tok/s |
|---|---|
| Optimized Speed rewriting a file it just wrote, stock settings (MTPLX 2.10.0) | 87.6 |
| Bare Speed on a fresh coding task at official Qwen 3.8 sampling (MTPLX 2.7.0) | 65.2 |
| Optimized Speed on a 3k-token chat answer (MTPLX 2.10.0) | 64.3 |
| The 27B record on a fresh generation: Qwen 3.6 27B Optimized Speed, 192-token bench, 2 July 2026, raw logs published | 81.74 |

Pack quality against the bf16 Qwen3.8-27B checkpoint, teacher-forced over 2,389 positions of code, prose, JSON and a multilingual notice (16 September 2026): Optimized Speed (4-bit dynamic) 96.0 percent top-1 agreement and KL 0.012; Optimized Quality (8-bit dynamic) 99.3 percent and KL 0.0005. Exactness on this release: a thousand four-token draws from the fast path match a thousand from the plain path within the plain path's own noise, at temperature 1, top-p 0.95, top-k 20. Details: [MTPLX 2.11.3 release notes](https://mtplx.com/releases/2.11.3/).

Runs on Apple Silicon Macs with 32 GB of unified memory or more (36 GB for Optimized Quality): MacBook Pro, MacBook Air, Mac mini and Mac Studio on M1 to M5. Guide: [Run Qwen 3.8 27B on a Mac](https://mtplx.com/models/qwen3.8-27b/). Comparison pages: [MTPLX vs mlx-serve](https://mtplx.com/compare/mtplx-vs-mlx-serve/), [MTPLX vs oMLX](https://mtplx.com/compare/mtplx-vs-omlx/), [MTPLX vs LM Studio](https://mtplx.com/compare/mtplx-vs-lm-studio/), [MTPLX vs Ollama](https://mtplx.com/compare/mtplx-vs-ollama/).

## Speeds

Measured on an M5 Max, fans verified at max, single stream, generation running
to the model's own stop, official Qwen 3.8 sampling (temperature 1.0, top-p
0.95, top-k 20).

| Run | tok/s |
|---|---|
| Coding task, medium reasoning, inside the MTPLX Mac app | 48.3 |
| Long reasoning at xhigh, 34k and 46k token answers | 33.2 and 33.1 |

Same night, same task, the 4-bit builds: Qwen 3.6 27B Optimized Speed V2 59.9
to 60.1 tok/s, Qwen 3.8 Optimized Speed 58.7, Bare Speed 65.2. This is the
quality pick, not the speed pick, and it is still well past 40 tok/s while
running the full-precision distribution.

Draft acceptance on the coding task by depth: 0.96, 0.88, 0.79. Verify cost
63.5 ms per round. Depth 3 was measured at +19.9% over depth 2 on the long
reasoning task.

## How it is built

- Every weight matrix at 8-bit with 64-weight groups.
- The GDN convolution kernels and recurrent state parameters, every norm, and
  the whole MTP head stay 16-bit.
- KL divergence to the original bf16 model on our coding battery: 0.00105.
  That is 21x closer than Optimized Speed and 36x closer than Bare Speed. In
  practice you will not tell the outputs apart from the bf16 model.

| | |
|---|---|
| Download | 29.4 GB |
| Peak unified memory (measured, this artifact) | 32.7 GB |
| Context window | 262,144 tokens |
| MTP depth | 3 |
| Sampling | temperature 1.0, top-p 0.95, top-k 20 (the official Qwen 3.8 contract) |

The tuned depth and draft settings ship inside `mtplx_runtime.json`. MTPLX
reads them on load. Speculation is exact: drafts are accepted with the
probability-ratio rule plus residual resampling, so the output follows the
model's own distribution at any temperature. Reasoning effort levels (xhigh,
medium, low) work, and preserved thinking flows through the MTP path.

## Use it

You want 36 GB of unified memory or more for this one. Mac app: download at
[mtplx.com](https://mtplx.com), pick "Qwen 3.8 27B Optimized Quality".

Command line:

```bash
pip install mtplx
mtplx serve --model Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality
```

Siblings: [Optimized Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed)
(recommended for coding) and
[Bare Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed)
(quickest burst chat speeds). On an M1 or M2 Mac use the
[FP16 build](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality-FP16)
of this model.
