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
---

# Qwen3.8-27B — MTPLX Optimized Speed

**Qwen3.8 was created by the Qwen team. This is a derivative conversion, not a
PocketAI-created base model.** PocketAiHub reproduced, validated, and packaged the published MTPLX Optimized Speed recipe.

Built with **[MTPLX 2.7.1](https://github.com/youssofal/MTPLX)** from
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The mixed-precision layout follows
[`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed): bulk 4-bit/group
32, embeddings + LM head + GDN output projections + final eight MLP blocks at
8-bit/group 64, sensitive state/norm tensors and the native MTP head in BF16.

## Use with MTPLX

```bash
pip install "mtplx==2.7.1"
mtplx serve --model PocketAiHub/Qwen3.8-27B-MTPLX-Optimized-Speed --profile turbo
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
| AR | 24.62 | 24.11 | 1.00× | — |
| D1 | 41.69 | 40.51 | 1.69× | 96.8% |
| D2 | 52.85 | 51.18 | 2.15× | 95.4%, 87.3% |
| D3 | 58.02 | 56.01 | **2.36×** | 93.3%, 87.4%, 79.9% |

The controlled tune used official Qwen3.8 sampling, thinking disabled, up to
1,024 generated tokens, and selected D3. Download payload: 21.31 GB.

### 4K context

The prompt contained 4,099 formatted tokens with `COBALT-7319` at 59.9% depth.
AR and D3 both returned exactly `COBALT-7319`; greedy outputs were identical.

| Mode | Prefill tok/s | Decode tok/s | Prompt eval | Peak process RSS | Peak footprint |
|---|---:|---:|---:|---:|---:|
| AR | 907.8 | 26.4 | 4.52s | 20.9 GB | 24.9 GB |
| D3 | 810.5 | 70.5 | 5.06s | 20.9 GB | 29.0 GB |

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

## KL divergence

PocketAiHub measured **mean forward KL 0.05850 nats** for BF16 → regular MTPLX on the disclosed token-level suite below (median 0.00203, p95 0.25609, top-1 agreement 94.27%). This repository's six weight payloads are also byte-for-byte identical to the [official MTPLX reference](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed), whose card reports **0.0220** on a different coding battery. The two values are not directly comparable because their prompt suites and token positions differ.

Method: 24 prompts (12 capability + 12 pinned JailbreakBench safety-evaluation
prompts), 16
BF16-greedy teacher-forced assistant positions per prompt, 384 positions total,
all 248,320 vocabulary logits, float32 capture and float64 probability math.
Direction is `D_KL(P_reference || P_candidate)` in nats. Full per-suite and
per-case aggregates are in `evaluation/kl-summary.json`.

The safety-evaluation subset is included to characterize model behavior, not to
endorse unsafe use or present reduced refusal as a quality or safety improvement.

## Provenance and license

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) by Qwen
- MTPLX recipe/runtime: [MTPLX](https://github.com/youssofal/MTPLX) by Youssofal
- Conversion, validation, packaging: PocketAiHub
- License: Apache-2.0; see `LICENSE` and `NOTICE`

Exact build and test summaries are in `build-recipe.json` and
`evaluation/results.json`.
