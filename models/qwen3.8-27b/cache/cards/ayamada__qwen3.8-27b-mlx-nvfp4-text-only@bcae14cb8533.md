---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- nvfp4
- 4-bit
- text-generation
---

# qwen3.8-27b-mlx-nvfp4-text-only

A **text-only**, NVFP4-quantized MLX build of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).

The vision tower and the MTP head are intentionally removed, so this checkpoint runs on
`mlx-lm` alone and needs only ~15.1 GB of weights. It is **not** a vision-language model:
image and video inputs are not supported.

## Model details

| | |
| - | - |
| Base model | Qwen/Qwen3.8-27B (pre-trained + post-trained) |
| Parameters | 26.9B (language model only) |
| Layers | 64 — one gated full-attention layer every 4th layer, Gated DeltaNet (linear attention) in the other three |
| Full attention | 24 query / 4 KV heads, head dim 256, partial RoPE (dim 64, theta 1e7) |
| Linear attention | Gated DeltaNet, 16 QK / 48 V heads, head dim 128 |
| Hidden / FFN | 5120 / 17408 |
| Vocabulary | 248,320 |
| Context | 262,144 tokens native |
| Quantization | NVFP4 — 4 bits, group size 16, RTN, 4.501 bits per weight |
| Non-quantized weights | float16 (faster than bfloat16 on M1/M2-class GPUs; `A_log` kept in bfloat16) |
| Size | 15.13 GB in 3 safetensors shards |
| Format | MLX (`mlx-lm` 0.32.2) |

## What was changed from the base model

-   **Vision removed** — all 333 `model.visual.*` tensors (~0.92 GB), `vision_config` and the
    image/video/vision special-token IDs are gone. `config.json` sets `language_model_only: true`.
-   **MTP removed** — the multi-token-prediction head is not included.
-   **Reasoning effort capped at `medium`** — the bundled chat template defaults `reasoning_effort`
    to `medium` (upstream defaults to `xhigh`) and renders an explicit `xhigh` as `medium`. With the
    `xhigh` instruction this 4-bit checkpoint degenerates into long repetition loops on non-trivial
    prompts; `medium` and `low` are unaffected, and no weights were changed for this.
-   **Quantized** — every `Linear` and `Embedding` weight (including `lm_head` and
    `embed_tokens`) is NVFP4; norms, `conv1d` and `A_log` remain in full precision.

## Usage

```bash
pip install -U "mlx-lm>=0.32.2"
```

Generate:

```bash
python -m mlx_lm.generate \
    --model ayamada/qwen3.8-27b-mlx-nvfp4-text-only \
    --prompt "Write a Python quicksort in 20 lines." \
    --max-tokens 512 --temp 0
```

Interactive chat, or an OpenAI-compatible local server:

```bash
python -m mlx_lm.chat   --model ayamada/qwen3.8-27b-mlx-nvfp4-text-only
python -m mlx_lm.server --model ayamada/qwen3.8-27b-mlx-nvfp4-text-only --port 8080
```

Thinking mode is on by default, as in the base model. The bundled chat template accepts
`enable_thinking`, `reasoning_effort` (`medium` or `low`) and `preserve_thinking` to control it.

> **`reasoning_effort` is capped at `medium`.** This build defaults to `medium`, and an explicit
> `xhigh` is accepted but rendered as `medium`, so the xhigh system instruction is never emitted.
> Edit `chat_template.jinja` if you really need `xhigh` (not recommended for this 4-bit build).

## Quality

Perplexity against the text part of [`mlx-community/Qwen3.8-27B-nvfp4`](https://huggingface.co/mlx-community/Qwen3.8-27B-nvfp4)
(~5.1k tokens of English and Japanese prose, non-overlapping 512-token windows):

| Model | Perplexity ↓ | Relative |
| - | - | - |
| mlx-community/Qwen3.8-27B-nvfp4 | 6.9239 | — |
| this model | 6.9519 | +0.4% |

The build was also checked for self-consistency: re-quantizing the upstream checkpoint with
the same pipeline reproduces the stored tensors byte-for-byte, and the exported index contains
zero vision or MTP tensors.

## Limitations

-   **Text only.** There is no vision tower, so images and videos cannot be processed.
-   **mlx-lm only.** Tools that always build a vision tower (`mlx-vlm`, oMLX, …) cannot load
    this checkpoint. Use `mlx-lm` or a runtime that honors `language_model_only`.
-   **No MTP head**, so MTP-based speculative decoding is unavailable. Use an external drafter
    (e.g. mlx-dspark / DFlash) if you need that speedup.
-   **Reasoning effort is capped at `medium`.** The upstream template defaults to `xhigh`, which
    sends this checkpoint into long repetition loops (endless name/enumeration lists) on non-trivial
    prompts. `xhigh` requests are downgraded to `medium` rather than rejected, so agent frameworks
    that always ask for the highest level keep working.
-   Quantization is lossy: expect small degradations versus the BF16 base model, more visible
    on long reasoning chains.

## How it was built

```bash
mlx_lm.convert \
    --hf-path Qwen/Qwen3.8-27B \
    --mlx-path qwen3.8-27b-mlx-nvfp4-text-only \
    --quantize --q-mode nvfp4 --dtype float16
```

`mlx-lm`'s `sanitize()` drops the vision and MTP tensors, shifts the zero-centered RMSNorm
weights and transposes `conv1d` on its own; post-processing was a small `config.json` cleanup
(leftover vision token IDs, `language_model_only`, `text_config.dtype`) plus the reasoning-effort
cap in `chat_template.jinja`. Built on an Apple M1 Max (64 GB) on 2026-10-18; the effort cap was
added on 2026-10-21.

## License

Apache-2.0, inherited from the base model. All credit for the underlying model goes to the
Qwen team.
