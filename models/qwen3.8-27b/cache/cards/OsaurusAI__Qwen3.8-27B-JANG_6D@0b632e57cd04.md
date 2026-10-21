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
- jang-6d
- quantized
- apple-silicon
- vision
- video
- reasoning
- agent
- tool-use
- qwen3_5
- imatrix
- awq
- hessian-allocation
---

<p align="center"><a href="https://osaurus.ai"><img src="./osaurus-x-banner.png" alt="Osaurus AI"></a></p>

# OsaurusAI/Qwen3.8-27B-JANG_6D

**Near-lossless — 0.0008 nats and 98.8% top-1 against the bf16 reference.**

A JANG bundle of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
— Qwen's 27B dense hybrid (GatedDeltaNet + gated attention) vision-language
model with flexible thinking control — quantized for Apple Silicon / MLX and
runnable with stock `mlx_vlm`. Text, image and video paths are all present in
this exact bundle, with the model's native multi-token-prediction head preserved.



## Why this quant

- **Measured allocation, not name rules** — every quantized module got its
  bit width from a Hessian-trace sensitivity capture (tr(H)·‖W‖²_F) over a
  **164,105-token, 8-domain** calibration corpus, so bits go where this model
  actually needs them.
- **AWQ applied** (α=0.25), folded into the producing RMSNorm using this
  family's zero-centered (+1) convention. Measured with an identical bit map
  and pipeline, AWQ is worth **~20 % lower KL at identical bits**
  (0.01002 vs 0.01245).
- **imatrix refit on every module ≤ 8 bits** — activation-weighted least
  squares against the same capture, at zero size cost. The refit re-applies the
  AWQ scales, so it does not silently revert them.
- **fp16 where quantization would lie** — the 27 vision-block `linear_fc2`
  projections (in_features 4304, indivisible by any MLX quant group) pass
  through in fp16 rather than being force-fit.
- **The full serving contract is stamped, not documented-elsewhere** — sampling
  presets, reasoning-effort tiers, thinking defaults, EOS pair and context
  guidance are all machine-readable in the bundle.

## Measured quality

Scored against the **bf16 source** (not against another quant) on 24 held-out
prompts that are disjoint from the calibration corpus, teacher-forced on the
reference's greedy continuation.

| Metric | Value |
|---|---|
| Held-out KL vs bf16 (median) | **0.0008 nats** |
| Held-out top-1 agreement | **98.83 %** |
| MTP depth-1 draft acceptance | **84.7 %** |
| Decode speed (M5 Max, depth 1) | **28.3 tok/s** |
| Decode speed (best depth = 2) | **31.6 tok/s** |
| On disk | 23.64 GiB |
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
| Quantization | 199x6-bit / 381x8-bit |

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

model, processor = load("OsaurusAI/Qwen3.8-27B-JANG_6D")

out = generate(model, processor, "Describe this image.", image=["photo.png"],
               max_tokens=512, temperature=1.0, top_p=0.95)
```

**Video note:** render video prompts through the bundle's own chat template
(`{"type": "video"}` → `<|vision_start|><|video_pad|><|vision_end|>`);
`mlx_vlm.prompt_utils.apply_chat_template` silently drops video items.

## MTP head

The native MTP head is preserved as its own shard (31 tensors). Depth-1 draft
acceptance was measured at **84.7 %**
on the model's own generated span — i.e. the span speculative decoding actually
drafts, not the prompt.

**Speculation depth.** Measured against a warm KV cache: best_depth is
**2** at **31.6 tok/s** (depth 1 = 28.3 tok/s). Deeper speculation stops
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
