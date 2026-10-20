---
library_name: mlx
base_model: google/gemma-4-26B-A4B-it
tags:
- mlx
- mlx-vlm
- moe
- multimodal
- vision
- gemma
- basequant-xl
pipeline_tag: image-text-to-text
---

# leonsarmiento/gemma-4-26B-A4B-it-3bit-XL-mlx

This model was converted to MLX format from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) using **BaseQuant_XL 3-bit mixed quantization** — the compact XL variant built for RAM-constrained Macs. The vision encoder is preserved (full multimodal). The unquantized BF16 base was used as the source.

**BaseQuant_XL** keeps the most routing-critical layers in full bf16 precision — the MoE `router.proj` (gate) and the `mlp` (shared/dense expert that processes all tokens) — while applying 8-bit quantization to embeddings and attention. Only the routed experts (`experts.switch_glu`), the bulk of the parameters, are pushed down to 3-bit. This is the same layer assignment as the [2-bit XL](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-2bit-XL-mlx) and [6-bit XL](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-6bit-XL-mlx) variants — only the routed-expert bit depth differs.

Note: `tie_word_embeddings=True` means there is no separate `lm_head` (tied to `embed_tokens`).

Gemma-4-26B-A4B is a 25.2B-parameter MoE (Mixture of Experts) model with 128 experts per layer (avg. 3.8B active parameters per token) and a vision encoder.

## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model leonsarmiento/gemma-4-26B-A4B-it-3bit-XL-mlx --max-tokens 256 --temperature 1.0 --top-p 1.0 --prompt "Hello"
```

## BaseQuant_XL Quantization Strategy

| Bit Depth | Layers | Rationale |
|-----------|--------|-----------|
| **bf16 (unquantized)** | `router.proj` (gate), `mlp` (shared/dense expert), vision tower | Routing decisions and the dense expert that processes every token — errors here cascade through all tokens |
| **8-bit** | `embed_tokens`, `self_attn` | Near-lossless for the attention path — the sensitive non-expert weights stay protected |
| **3-bit** | `experts.switch_glu` (routed experts) | The bulk of parameters (only ~4B of 26B active per token) — 3-bit keeps the footprint small |

### Quantization Details

| Layer | Bits | Group Size |
|-------|------|------------|
| `router.proj` (gate) | bf16 | — |
| `mlp` (shared/dense expert) | bf16 | — |
| Vision tower (full) | bf16 | — |
| `embed_tokens` | 8 | 64 |
| `self_attn` | 8 | 64 |
| `experts.switch_glu` (routed experts) | 3 | 64 |
| Default fallback | 8 | 64 |

- **Quantization type**: BaseQuant_XL mixed (multimodal, vision preserved)
- **Bits per weight**: 4.404
- **Total size**: ~13 GB (3 shards)
- **Group size**: 64
- **Method**: Custom `quant_predicate` via `mlx_vlm`
- **Note**: `tie_word_embeddings=True` — no separate lm_head

## Thinking Mode (LM Studio setup)

Thinking is **enabled by default** in this repo — the flags `enable_thinking = true` and `preserve_thinking = true` are baked into the chat template (both `chat_template.jinja` and `tokenizer_config.json`), so servers that don't pass template kwargs (LM Studio, oMLX) get thinking out of the box. Passing an explicit `enable_thinking=false` from a server that supports template kwargs still overrides the default.

For LM Studio to render the reasoning trace correctly, set the model's **Reasoning Parser** (in the model settings) to custom strings:

| Setting | Value |
|---------|-------|
| Reasoning parser **start string** | `<\|channel>thought` |
| Reasoning parser **end string** | `<channel\|>` |

The model opens each reasoning block with `<|channel>thought` and closes it with `<channel|>` before emitting the final answer — without these parser strings, LM Studio will show the raw channel markers inline with the response.

## Recommended Inference Parameters

| Parameter | Value |
|-----------|-------|
| `temperature` | 1.0 |
| `top_p` | 1.0 |
| `top_k` | 64 |
| `min_p` | 0.01 |
| `repeat_penalty` | 1.1 |

## Chat Template

The chat template is synced to the **canonical Google upstream** version from [`google/gemma-4-26B-A4B-it/chat_template.jinja`](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/chat_template.jinja). Do not rely on the HF cache copy — it may be stale.

Key markers in the latest template: `preserve_thinking`, `continues_into_next`, `prev_non_tool_role`, null handling in `format_argument`.

The template is injected into both `chat_template.jinja` (standalone file) and `tokenizer_config.json` (`chat_template` JSON field).

## Instruct Benchmarks

| Model | MMLU | MMLU_PRO | HUMANEVAL | MBPP | Avg |
|-------|------|----------|-----------|------|-----|
| Huihui-Gemma-4 26B 3bit-XL | 78.0% | **71.0%** | **98.0%** | 82.0% | **82.3%** |
| Gemma-4 26B 3bit-XL | 80.0% | 65.0% | **98.0%** | **86.0%** | **82.3%** |
| Qwen3.8-27B 3bit-mtp | 80.0% | 61.0% | 94.0% | 70.0% | 76.3% |
| Qwen3.6-35B 3bit-XL | 82.0% | 63.0% | 66.0% | 80.0% | 72.8% |
| Huihui-Qwen3.6-35B 3bit-XL | **83.0%** | 63.0% | 62.0% | 76.0% | 71.0% |

![Family comparison](poor_ram_family_benchmarks.png)

*Instruct mode (thinking off). Sampled: MMLU 100/14042, MMLU_PRO 100/12032, HUMANEVAL 50/164, MBPP 50/500. Wall times not directly comparable across architectures (dense vs MoE active params).*

**Highlights**: 98% HUMANEVAL + 86% MBPP — the family's coding leader, matching or beating its own 6-bit and 8-bit XL siblings on code at less than two-thirds the size. Avg ties the abliterated twin at 82.3%.
