---
library_name: mlx
base_model: ornith-ai/Ornith-1.5-35B-A3B
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
tags:
- mlx
- mlx-vlm
- moe
- multimodal
- vision
- coding
- agentic
- swe
- reasoning
- basequant-xl
- ornith
pipeline_tag: image-text-to-text
---

# leonsarmiento/Ornith-1.5-35B-A3B-6bit-XL-mlx

[**Ornith-1.5-35B-A3B**](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) by [Ornith AI](https://huggingface.co/ornith-ai) — a 35B-A3B sparse MoE built through end-to-end self-improvement (jointly optimized task generation, scaffold construction, and solution rollouts via RL), quantized for Apple Silicon using the **BaseQuant_XL 6/8-bit** recipe.

Only ~3B parameters active per token — the decode speed of a small model with the capacity of a 35B one. Ornith-1.5 is an agentic-coding specialist: **79 SWE-bench Verified**, 59.6 SWE-bench Pro, 67.8/68.5 Terminal-Bench 2.1 — beating its Ornith-1.0 parent (+3.4 SWE-bench) and Qwen3.6-35B-A3B (+5.6) by wide margins.

This is a full multimodal build — the vision tower is preserved.

## About XL Quantization

**BaseQuant_XL** is a fully **data-agnostic**, static quantization. No calibration dataset, no sensitivity analysis, no importance matrix. Precision is allocated purely by architectural role — routing-critical layers get higher precision, bulk expert parameters get lower precision. The result is a transparent, faithful capture of the source model.

Data-dependent calibration quantizations (iMatrix, AWQ, GPTQ, oQ, oQ4e, etc.) use a calibration set to guide bit allocation. This can produce a **skewed representation** of the model: domains well-represented in the calibration data (English, popular topics, public or leaked benchmarks) are preserved better, while underrepresented domains (non-English languages, niche use cases, your own data) are preserved worse. XL avoids this trade-off entirely — it generalizes honestly because it is never fit to any particular data distribution.

## Quickstart

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model leonsarmiento/Ornith-1.5-35B-A3B-6bit-XL-mlx --max-tokens 256 --temperature 0.6 --top-p 0.95 --prompt "Implement an LRU cache in Python with O(1) get/put."
```

Works with LM Studio — vision mmproj included. Thinking mode is on by default (emits `<think>...</think>`).

## Quantization Strategy

BaseQuant_XL recipe — precision is allocated by layer importance, not applied uniformly:

| Layers | Bits | Rationale |
|--------|------|-----------|
| `mlp.gate` (router), `shared_expert_gate`, `lm_head`, `shared_expert` | bf16 | Routing decisions and output projection — any quantization noise here causes expert misrouting or output degradation |
| `embed_tokens`, `self_attn`, `linear_attn` | 8-bit | Every-token layers — near-lossless, attention quality preserved |
| `vision_tower`, `switch_mlp` (routed experts) | 6-bit | Bulk parameters — 256 experts with only 8 active per token; redundancy absorbs quantization noise. 6-bit is the sweet spot for routed experts (higher bits can cause overthinking) |

- **Bits per weight**: ~6.8 · **Total size**: ~28 GB · **Group size**: 64

**Notes specific to this build**:
- Source ships an MTP (multi-token prediction) layer — dropped in this build (engines that execute bundled MTP are the exception, not the rule; standard XL builds omit it).
- Unlike Ornith-1.0 (which stores experts individually), Ornith-1.5's source stores merged per-layer expert tensors — converts via stock `mlx_vlm` sanitize.

## Recommended Inference Parameters

Per the source model card — new settings for the Ornith-1.5 generation:

| Parameter | Value |
|-----------|-------|
| `temperature` | 0.6 (general tasks) · 1.0 (reproducing reported benchmarks) |
| `top_p` | 0.95 |
| `top_k` | 20 |
| `presence_penalty` | 1.1 |
| `reasoning_parser` | **qwen3** (changed from `deepseek_r1` in Ornith-1.0) |
| `tool_call_parser` | **qwen3_xml** |

Thinking is on by default (`<think>...</think>` before the answer); with a reasoning parser enabled the chain-of-thought is returned in a separate `reasoning_content` field.

## Model Overview

| Property | Value |
|----------|-------|
| **Architecture** | Qwen3.5-family MoE (35B-A3B) + native vision encoder |
| **Parameters** | 35.9B total / ~3B active per token |
| **Experts** | 256 (8 routed + 1 shared) |
| **Attention** | Hybrid — 30 linear_attn + 10 full attention (40 layers) |
| **Modalities** | text, image, video → text |
| **Context window** | 262,144 tokens native |
| **Thinking** | `<think>...</think>` |
| **Tool calling** | XML-style (`<tool_call><function=...><parameter=...>`) |
| **License** | MIT |

## Source Model Benchmarks (from Ornith AI)

| Benchmark | Ornith-1.5-35B-A3B | Ornith-1.0-35B-A3B | Qwen3.6-35B-A3B |
|-----------|:---:|:---:|:---:|
| SWE-bench Verified | **79.0** | 75.6 | 73.4 |
| SWE-bench Pro | **59.6** | 50.4 | 49.5 |
| Terminal-Bench 2.1 (Terminus-2) | **67.8** | 64.2 | 52.5 |
| Terminal-Bench 2.1 (Claude Code) | **68.5** | 62.8 | 49.2 |

## Source

[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — [Ornith 1.5 blog post](https://ornith.ai/ornith_1_5.html)
