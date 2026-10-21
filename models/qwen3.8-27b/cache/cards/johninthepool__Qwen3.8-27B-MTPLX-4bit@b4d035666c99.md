---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - mtplx
  - quantized
  - multi-token-prediction
  - speculative-decoding
  - qwen3.5
  - apple-silicon
pipeline_tag: text-generation
---

# Qwen3.8-27B-MTPLX-4bit

> **MTP weights kept.** This build does not strip the model's multi-token-prediction head — it is preserved at full bf16 precision (`mtp_policy: keep_bf16`) alongside the 4-bit quantized body, so native speculative decoding still works.

A 4-bit MLX build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple Silicon, produced with [mtplx](https://github.com/philipjohnbasile/mtplx) forge. Unlike a standard `mlx_lm.convert` quantization, this build **preserves the model's native multi-token-prediction (MTP) head** instead of discarding it, so it runs with real speculative decoding on mtplx — not just a quantized weight dump.

## Why this build exists

Qwen3.8-27B is a hybrid architecture (Gated DeltaNet linear attention interleaved with Gated Attention, 16 groups of 3:1). Standard MLX conversion paths (`mlx_lm.convert`, and MLX inference engines built around dense-attention assumptions) silently strip the model's 15 MTP tensors during conversion — the model still loads and generates, but you lose the native speculative-decoding speedup entirely, often without any error or warning.

This build's recipe uses `mtp_policy: keep_bf16`, which retains the MTP block in full precision alongside the quantized body, and validates it with mtplx's `tune` verification suite (`long-code-uncapped`, 2048 max tokens) before shipping.

## Quantization recipe

| Parameter | Value |
|---|---|
| Body precision | 4-bit affine |
| Group size | 64 |
| MTP head | kept at bf16 (not quantized) |
| Source | `Qwen/Qwen3.8-27B` (bf16 native) |

Group size 64 was chosen over the tighter g32 default because the target hardware (Apple Silicon with 64GB+ unified memory) has headroom to trade a small amount of table overhead for faster dequantization, with no measurable quality cost at this bit depth.

## Requirements

- Apple Silicon Mac (M-series)
- [mtplx](https://github.com/philipjohnbasile/mtplx) — required to use the native MTP speculative-decode path. Standard `mlx-lm` can load this model's body weights but will not use the MTP head.

## Usage

```bash
mtplx quickstart --model johninthepool/Qwen3.8-27B-MTPLX-4bit --port 8020 \
  --reasoning off --paged-kv-quantization q8
```

`--reasoning off` is recommended for agentic/tool-calling workloads — leaving reasoning on causes this model to spend a large number of tokens per turn on hidden thinking before it produces usable output.

`--paged-kv-quantization q8` quantizes the KV cache to 8-bit, giving substantially larger usable context at negligible quality cost. Qwen3.8-27B natively supports up to 262,144 tokens of context (extendable to ~1M with YaRN).

## What you get vs. what you give up

**You get:** native speculative decoding via the preserved MTP head, full 4-bit body compression (~17GB on disk), long-context serving via paged KV quantization.

**You give up relative to the 8-bit sibling:** a small amount of body-weight precision. If output quality matters more than footprint/throughput for your use case, see [johninthepool/Qwen3.8-27B-MTPLX-8bit](https://huggingface.co/johninthepool/Qwen3.8-27B-MTPLX-8bit).

## Provenance

Built from the original `Qwen/Qwen3.8-27B` weights with no fine-tuning, distillation, or calibration pass applied — this is a direct affine (round-to-nearest) quantization of the release weights, with the MTP head kept unquantized. No behavioral changes beyond quantization noise are expected relative to the source model.
