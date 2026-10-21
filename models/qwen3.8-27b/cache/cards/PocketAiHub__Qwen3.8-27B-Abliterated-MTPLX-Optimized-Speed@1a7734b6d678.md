---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- macos
- mtplx
- speculative-decoding
- multi-token-prediction
- qwen
- qwen3.8
- multimodal
- abliterated
---

# Qwen3.8-27B Abliterated — MTPLX Optimized Speed

**Qwen3.8 was created by the Qwen team. This is a derivative conversion, not a
PocketAI-created base model.** PocketAiHub performed a refusal-direction orthogonal projection on 80 language residual-output tensors, then converted and validated the result.

Built with **[MTPLX 2.7.1](https://github.com/youssofal/MTPLX)** from
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The mixed-precision layout follows
[`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed): bulk 4-bit/group
32, embeddings + LM head + GDN output projections + final eight MLP blocks at
8-bit/group 64, sensitive state/norm tensors and the native MTP head in BF16.

## Abliteration notice

This research derivative suppresses learned refusal behavior. It is not safer,
does not guarantee compliance, and may produce incorrect, harmful, or
unpredictable content. Apply appropriate safeguards for your use case.

## Use with MTPLX

```bash
pip install "mtplx==2.7.1"
mtplx serve --model PocketAiHub/Qwen3.8-27B-Abliterated-MTPLX-Optimized-Speed --profile turbo
```

The OpenAI-compatible API is then available at `http://127.0.0.1:8000/v1`.
The shipped runtime contract selects native MTP depth 3. Official Qwen3.8
target sampling is temperature 1.0, top-p 0.95, top-k 20.

> **Recommended context: 32K tokens or less.** The checkpoint retains Qwen3.8's
> 262,144-token configured context window, but the MTPLX speed path is currently
> recommended only through 32K. For longer contexts, use the regular 4-bit MLX
> release instead.

## Measured performance

Apple M5 Max (40-core GPU, 128 GB unified memory), macOS 26.4, single stream,
Apple automatic fan control. These are measurements on this exact artifact,
not universal hardware guarantees.

| Mode | Decode tok/s | End-to-end tok/s | Speedup vs AR | Acceptance by depth |
|---|---:|---:|---:|---|
| AR | 24.74 | 24.28 | 1.00× | — |
| D1 | 41.55 | 40.53 | 1.68× | 95.6% |
| D2 | 51.55 | 49.70 | 2.08× | 97.9%, 91.6% |
| D3 | 58.05 | 56.01 | **2.35×** | 96.3%, 88.8%, 80.6% |

The controlled tune used official Qwen3.8 sampling, thinking disabled, up to
1,024 generated tokens, and selected D3. Download payload: 21.31 GB.

### 4K context

The prompt contained 4,099 formatted tokens with `COBALT-7319` at 59.9% depth.
AR and D3 both returned exactly `COBALT-7319`; greedy outputs were identical.

| Mode | Prefill tok/s | Decode tok/s | Prompt eval | Peak process RSS | Peak footprint |
|---|---:|---:|---:|---:|---:|
| AR | 602.4 | 26.4 | 6.80s | 20.9 GB | 24.9 GB |
| D3 | 615.2 | 58.4 | 6.66s | 20.9 GB | 29.0 GB |

The 4K response contained only nine generated tokens, so decode tok/s is more
informative than its end-to-end generation rate.

## Feature validation

- 12/12 capability checks
- 8/8 structured tool-selection checks
- reasoning on and reasoning off
- OpenAI `image_url` input: solid-red image identified as `red`
- verified-native MTP contract with all 15 BF16 MTP tensors
- 262,144-token configured context window; exact 4K retrieval tested; MTPLX
  usage recommended at 32K or less

The 8/8 tool-selection suite follows the repository's established scorer. In
the no-tool control, MTPLX 2.7.1 correctly emitted no tool call but returned an
empty body while the tool schema was attached; ordinary text exactness passed
separately. PNG/JPEG/WebP image input is live-tested. Video processor metadata
is included for checkpoint completeness, but live video-server input was not
claimed or tested.

## Explicit-refusal screen

| Suite | Explicit refusals | Final-answer text | Finish reason |
|---|---:|---:|---:|
| JailbreakBench safety-evaluation set | 0/100 | 100/100 | 100/100 length |
| JBB benign | 0/100 | 100/100 | 100/100 length |

This was a deterministic, batch-size-1, thinking-off, 128-token **early
explicit-refusal screen** using a transparent phrase scorer. It does not prove
universal compliance, answer correctness, or safety. Raw aggregate methodology
is in `evaluation/results.json`.

These results characterize refusal behavior; a lower refusal rate is not
inherently better and is not presented as a quality score, safety improvement,
or endorsement of unsafe use.

## KL divergence

PocketAiHub measured **mean forward KL 0.30906 nats** for regular MTPLX → abliterated MTPLX at matched quantization (median 0.01142, p95 1.61390, top-1 agreement 85.94%). The capability-prompt mean was 0.05640; the safety-evaluation subset mean was 0.56171. This isolates additional abliteration drift by comparing checkpoints with the same MTPLX quantization layout. The official 0.0220 value measures a different quantization comparison and must not be reused as the abliteration KL.

Method: 24 prompts (12 capability + 12 pinned JailbreakBench safety-evaluation
prompts), 16
BF16-greedy teacher-forced assistant positions per prompt, 384 positions total,
all 248,320 vocabulary logits, float32 capture and float64 probability math.
Direction is `D_KL(P_reference || P_candidate)` in nats. Full per-suite and
per-case aggregates are in `evaluation/kl-summary.json`.

## Provenance and license

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen
- MTPLX recipe/runtime: [MTPLX](https://github.com/youssofal/MTPLX) by Youssofal
- Conversion, validation, packaging, and refusal-direction projection: PocketAiHub
- License: Apache-2.0; see `LICENSE` and `NOTICE`

Exact build and test summaries are in `build-recipe.json` and
`evaluation/results.json`.
