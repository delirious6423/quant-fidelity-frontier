---
language:
- en
library_name: mlx
license: mit
pipeline_tag: image-text-to-text
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
- mlx
- jang
- ornith-1.5-35b-a3b-jang-4m
- quantized
- apple-silicon
- ornith
- vision
- video
- reasoning
- thinking
---

<p align="center"><a href="https://osaurus.ai"><img src="./osaurus-x-banner.png" alt="Osaurus AI"></a></p>

# OsaurusAI/Ornith-1.5-35B-A3B-JANG_4M

`Ornith-1.5-35B-A3B-JANG_4M` MLX bundle of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — .

Ornith 1.5 is an agentic coding / reasoning VLM built on a hybrid **gated-delta linear attention + full attention** backbone (3:1), with a 27-layer vision tower and native video support.

## Bundle

| Field | Value |
|---|---|
| Source | `ornith-ai/Ornith-1.5-35B-A3B` |
| Architecture | `qwen3_5_moe` / `Qwen3_5MoeForConditionalGeneration` |
| Size on disk | **20.01 GiB** |
| Layers | 40 |
| Hidden size | 2048 |
| Routed experts | 256 |
| Context | 262,144 |
| Shards | 5 |
| Bit distribution | {4: 1090, 5: 44, 8: 240} |

## How it was quantized

Three calibration methods, all driven by **one** capture pass — the per-input-channel second moment `E[x_c^2]` is simultaneously the Hessian diagonal, the imatrix weighting and the AWQ salient-channel statistic.

| Method | What it does here |
|---|---|
| **Hessian-trace allocation** | Bits assigned by measured `tr(H)·‖W‖²_F` per module, not by tensor name. The vision tower scores *higher* than the text MLP on this model, which a name-based profile gets backwards. |
| **imatrix refit** | Activation-weighted affine fit replacing RTN codes — mean weighted rel-err **0.0735**. |
| **AWQ** | Salient-channel scaling (alpha=0.25), absorbed into the producing RMSNorm across 80 norm groups / 390 projections. |

Tensors whose `in_features` is divisible by no MLX group size (the 27 vision `linear_fc2` at 4304) stay fp16.

## Modalities

| Modality | Status |
|---|---|
| Text | supported |
| Vision | supported — 333 vision-tower tensors, `preprocessor_config.json` + `processor_config.json` ride with the bundle |
| Video | supported — `video_preprocessor_config.json` present; verified end-to-end |
| Audio | **not supported.** The tokenizer defines `<|audio_start|>` / `<|audio_end|>` / `<|audio_pad|>`, but the model has no `audio_config` and no audio-tower weights. These are vestigial tokens; the capability gate is weight-gated to false. |

## Reasoning

Reasoning is **ON by default** — the no-kwarg generation prompt is byte-identical to `enable_thinking=True` and ends `<|im_start|>assistant\n<think>\n`.

It is toggleable, but note *how*: `enable_thinking=False` does not remove the think block, it prefills an **empty closed** one (`<think>\n\n</think>\n\n`). A parser testing merely for the presence of a `<think>` block will find one in both modes — test whether it has content.

There are **no `reasoning_effort` tiers** on this model family (unlike Qwen3.8). History `<think>` blocks are preserved unconditionally. Reasoning parser: `qwen3`; tool parser: `qwen3_coder`.

## Sampling

Both presets from the vendor card are stamped into `jang_config.json`, and the **coding** preset is also written to `generation_config.json` so the two files agree.

> Ornith 1.5 is an agentic coding model (SWE-bench Verified 79, Terminal-Bench 2.1 67.8), so this bundle defaults to the **coding** preset. Upstream's own `generation_config.json` ships the *general* numbers (temp 1.0, presence 1.5) — use `sampling_modes.general` if you want parity with the vLLM/Transformers defaults.

| Preset | temp | top_p | top_k | min_p | presence | repetition |
|---|---|---|---|---|---|---|
| general | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| coding *(default)* | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |

Stop tokens: `[248046, 248044]` (`<|im_end|>`, `<|endoftext|>`).

## Speculative decoding (MTP)

This bundle **preserves the native MTP head** (2341 `mtp.*` tensors). Recommended 1 draft/step on Apple silicon (`vmlx_mtp_tuning.json`); that is a recommendation, not a measured sweep on this artifact.

## Credits

JANG quantization by **Jinho Jang** — <eric@osaurus.ai>

Base model: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) by Ornith AI.
