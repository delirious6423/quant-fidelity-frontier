---
language:
- en
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- jang
- mxfp8
- quantized
- apple-silicon
- vision
- video
- reasoning
- agent
- tool-use
- qwen3_5
---

<p align="center"><a href="https://osaurus.ai"><img src="./osaurus-x-banner.png" alt="Osaurus AI"></a></p>

# OsaurusAI/Qwen3.8-27B-MXFP8

**Uniform 8-bit MXFP8 — structural, uncalibrated. Use it only if a runtime mishandles affine per-module maps.**

A JANG bundle of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
— Qwen's 27B dense hybrid (GatedDeltaNet + gated attention) vision-language
model with flexible thinking control — quantized for Apple Silicon / MLX and
runnable with stock `mlx_vlm`. Text, image and video paths are all present in
this exact bundle, with the model's native multi-token-prediction head preserved.

**Measured honestly: this is not the best tier.** At 26.4 GiB it scores median KL 0.0189 / 95.83% top-1, which JANG_4D beats at **0.0086 / 97.01% in 16.6 GiB** — 2.2x better KL in 37% less space. Uniform 8-bit with no Hessian allocation, no AWQ and no imatrix refit loses to a calibrated 4-bit-class allocation. MXFP8 is kept as a structural fallback, not as a quality ceiling.

## Why this quant

- **Uniform MXFP8** (gs32) across all 580 quantizable modules — no
  Hessian allocation, no AWQ, no imatrix refit. That is the whole point of this
  tier, and also why it loses to the calibrated ones.
- **fp16 where quantization would lie** — the 27 vision-block `linear_fc2`
  projections (in_features 4304) pass through in fp16.
- **The full serving contract is stamped, not documented-elsewhere** — sampling
  presets, reasoning-effort tiers, thinking defaults, EOS pair and context
  guidance are all machine-readable in the bundle.

## Measured quality

Scored against the **bf16 source** (not against another quant) on 24 held-out
prompts that are disjoint from the calibration corpus, teacher-forced on the
reference's greedy continuation.

| Metric | Value |
|---|---|
| Held-out KL vs bf16 (median) | **0.0189 nats** |
| Held-out top-1 agreement | **95.83 %** |
| MTP depth-1 draft acceptance | **81.0 %** |
| Decode speed (M5 Max, depth 1) | **27.9 tok/s** |
| Decode speed (best depth = 2) | **29.3 tok/s** |
| On disk | 26.40 GiB |
| Runs on | Apple Silicon with ≥ 32 GB unified memory |

Median KL is reported rather than mean: against a near-deterministic reference
continuation KL is unbounded, so a single low-entropy prompt dominates a mean.

## The lineup

[JANG_2D](https://huggingface.co/OsaurusAI/Qwen3.8-27B-JANG_2D) (10.6 GiB) · [JANG_4D](https://huggingface.co/OsaurusAI/Qwen3.8-27B-JANG_4D) (16.6 GiB) · [JANG_6D](https://huggingface.co/OsaurusAI/Qwen3.8-27B-JANG_6D) (23.6 GiB) · [MXFP8](https://huggingface.co/OsaurusAI/Qwen3.8-27B-MXFP8) (26.4 GiB)

## Model + bundle facts

| Field | Value |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (dense 27B VLM) |
| Layout | 64 layers — 48 GatedDeltaNet + 16 gated full-attention (partial RoPE dim 64) |
| Vision | native image + video tower (501 tensors, preserved) |
| MTP | native multi-token-prediction head **preserved** (31 tensors, own shard) |
| Context | 262,144 native, extensible to 1M |
| Quantization | 580x8-bit MXFP8 (gs32) |

## Serving contract (stamped in the bundle)

Read these from `generation_config.json` + `jang_config.json` rather than
re-deriving them:

- **Thinking ON by default** — `temperature=1.0, top_p=0.95, top_k=20`. This is
  the agentic preset and the correct preset for coding agents.
  Instruct / non-thinking preset: `temperature=0.7, top_p=0.80, top_k=20,
  presence_penalty=1.5`.
- **`reasoning_effort`: `low` / `medium` / `xhigh`** (default `xhigh`), carried
  as a chat-template kwarg.
- **`preserve_thinking` ON by default** — Qwen3.8 retains reasoning context
  across turns, and it is prefix-cache friendly.
- Reasoning OFF = prefilled closed `<think>\n\n</think>\n\n`, never plain
  omission. Reasoning parser `qwen3`; tool-call parser `qwen3_coder`
  (XML function dialect).
- Stop on **both** EOS ids `248046` and `248044`.
- Recommended output budget: up to 262,144 reasoning + 131,072 final tokens.

## Use it

```bash
pip install -U mlx-vlm
```

```python
from mlx_vlm import load, generate

model, processor = load("OsaurusAI/Qwen3.8-27B-MXFP8")

out = generate(model, processor, "Describe this image.", image=["photo.png"],
               max_tokens=512, temperature=1.0, top_p=0.95)
```

**Video note:** render video prompts through the bundle's own chat template
(`{"type": "video"}` → `<|vision_start|><|video_pad|><|vision_end|>`);
`mlx_vlm.prompt_utils.apply_chat_template` silently drops video items.

## MTP head

The native MTP head is preserved as its own shard (31 tensors). Depth-1 draft
acceptance was measured
on the model's own generated span — i.e. the span speculative decoding actually
drafts, not the prompt.

**Speculation depth.** Measured against a warm KV cache: best_depth is
**2** at **29.3 tok/s** (depth 1 = 27.9 tok/s). Deeper speculation stops
paying fast — each extra verified token adds ~20% to the target decode step,
so **depth 3 is a net loss on every tier**. The bundle stamps the measured
depth in `vmlx_mtp_tuning.json` with its baseline/best/speedup evidence.

Head width was measured to be irrelevant to acceptance between 4-bit gs64 and
8-bit gs128 (spread inside one standard error, and non-monotonic), so the head
is stored at the cheapest width that costs nothing.

Acceptance is strongly shape-dependent — it moves ~17 points across
reasoning/tool/turn-shape changes, and structured tool output drafts far better
than prose. Treat a single acceptance figure as one slice, not a guarantee.

## Credits

Quantized by Jinho Jang — eric@osaurus.ai
