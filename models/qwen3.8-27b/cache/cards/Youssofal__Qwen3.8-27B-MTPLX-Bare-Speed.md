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
- chat
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
- coding
---

**[MTPLX](https://mtplx.com): the fastest way to run Qwen 3.8 on a Mac. Native multi-token-prediction speculative decoding on Apple Silicon, two to three times the speed of plain decoding, exact at any temperature.**

# Qwen 3.8 27B Bare Speed

**Quickest burst chat speeds. Lower quality and slower on long coding tasks.**

The fastest of the three MTPLX Qwen 3.8 builds and the smallest download.
Qwen3.8-27B in flat 4-bit with its native multi-token-prediction head kept, so
[MTPLX](https://mtplx.com) drafts ahead and verifies in one pass. If you want
the snappiest chat on a Mac and can live with a rougher quant, this is it. For
coding, pick Optimized Speed.

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
| Coding task, medium reasoning, `mtplx serve` | 65.2 |
| Same task inside the MTPLX Mac app | 64.4 |
| Long reasoning at xhigh, 34k and 37k token answers | 35.7 and 32.0 |
| One 52,740-token answer, 27.2 minutes, ended at the model's own stop | 32.4 sustained |

Same night, same task: the previous MTPLX flagship Qwen 3.6 27B Optimized
Speed V2 ran 59.9 to 60.1 tok/s. oMLX 0.5.7 serving its own Qwen 3.8 4-bit
MTP quant ran 63.3. LM Studio on the 52k-token long answer ran 17.40 tok/s
against 32.4 here.

Draft acceptance on the coding task by depth: 0.95, 0.86, 0.78. Verify cost
44 ms per round.

## How it is built

- Every weight matrix at 4-bit with 64-weight groups. Nothing promoted.
- The GDN convolution kernels and recurrent state parameters, every norm, and
  the whole MTP head stay 16-bit.
- KL divergence to the original bf16 model on our coding battery: 0.0376.
  Optimized Speed is 1.7x closer, Optimized Quality 36x closer. That is the
  trade you make for the speed.

| | |
|---|---|
| Download | 16.0 GB |
| Peak unified memory (measured, this artifact) | 17.0 GB |
| Context window | 262,144 tokens |
| MTP depth | 3 |
| Sampling | temperature 1.0, top-p 0.95, top-k 20 (the official Qwen 3.8 contract) |
| Draft sampler | temperature 0.6 (measured winner for this build, 46.1 vs 42.4 tok/s) |

The tuned depth and draft settings ship inside `mtplx_runtime.json`. MTPLX
reads them on load. The draft sampler is a speed knob only: MTPLX accepts
drafts with the probability-ratio rule plus residual resampling, so the output
follows the model's own distribution at any temperature. Reasoning effort
levels (xhigh, medium, low) work, and preserved thinking flows through the MTP
path.

## Use it

Mac app: download at [mtplx.com](https://mtplx.com), pick "Qwen 3.8 27B Bare
Speed".

Command line:

```bash
pip install mtplx
mtplx serve --model Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed
```

Siblings: [Optimized Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed)
(recommended for coding) and
[Optimized Quality](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality)
(8-bit, perfect quality). On an M1 or M2 Mac use the
[FP16 build](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Bare-Speed-FP16)
of this model.
