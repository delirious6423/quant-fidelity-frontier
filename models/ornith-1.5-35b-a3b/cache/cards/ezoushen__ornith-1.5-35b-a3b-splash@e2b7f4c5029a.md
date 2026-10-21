---
license: apache-2.0
language:
  - en
library_name: splash
pipeline_tag: image-text-to-text
inference: false
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
tags:
  - splash
  - apple-silicon
  - mac
  - macos
  - metal
  - on-device
  - local-inference
  - ornith
  - qwen3.6
  - moe
  - mixture-of-experts
  - vision
  - multimodal
  - 4-bit
  - quantized
  - dflash2
  - speculative-decoding
---

# Ornith 1.5 35B-A3B for Mac: 4-bit Splash package with vision

Run [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) on an Apple silicon Mac
with [Splash](https://github.com/incoai/splash): a community 4-bit conversion with vision, 1.47× faster
than the same weights on MLX with MTP for short code in a single stream, and 3.46× with four concurrent
requests.
Independent and unendorsed by Ornith AI or Inco AI.

```bash
brew install incoai/tap/splash
splash serve --model ezoushen/ornith-1.5-35b-a3b-splash
```

Requires Apple M3 or newer, macOS 26.4+, and at least 36 GB unified memory. About 20 GB of
Splash-specific packed binaries — not Transformers, GGUF or MLX.

## At a glance

Results from an M5 Max (40-core GPU, 128 GB); qualifications below.

| | This package | Comparison or notes |
| --- | --- | --- |
| Size | 20.9 GB, 56 files, vision included | Ornith's MLX 4-bit repo has no vision, and Splash 1.1.0 stops on its config ([reported](https://github.com/incoai/splash/issues/38#issuecomment-5865447638)) |
| Short code, single stream | **1.47× faster** | the same weights on MLX with MTP |
| Four concurrent requests | **3.46× faster** | the same weights on MLX with MTP |
| Draft acceptance | 2.63 tokens per step | 4.45 for Inco's own Qwen3.6 package |
| Identical greedy output to Ornith on MLX | 136 of 200 prompts | code 40/40 · vision 40/40 · long context 35/40 · math 21/40 · chat 0/40 |
| GSM8K (32-item subset) | 30 / 32 | 28 / 32 on the MLX production lane |
| JevBench public decisions (231, `/v1/systemone`) | 83.1% | — |
| Splash versions | 1.0.2, with vision | also loads on 1.1.0 (checked text-only) |

## Draft, tokenizer and output fidelity

**The draft was trained on Qwen3.6-35B-A3B, not Ornith.** Inco's DFlash 2 draft is reused
unmodified. Splash decodes only through the draft, with no autoregressive fallback. Acceptance
is **2.631 tokens per verify step** versus **4.447** for Inco's Qwen3.6 package on the same
harness: **59.2%**. An Ornith-conditioned draft would be faster; training one remains unfinished.

**Combining marks may tokenise differently than on other Ornith engines.** The packed tokenizer's
`pre_tokenizer` Split regex omits `\p{M}` from two character classes where official Ornith's and
Inco's shipped tokenizers include it. Vocabulary (248,044 entries) and merges match Ornith's;
two pre-tokenisation fields do not. The variant's origin is unknown. Prompt rendering matched
on 12 of 12 test conversations, but none contained combining marks, so this does not generalise.

**Output is not token-identical to the production MLX lane:** 136/200 exact matches, with the
breakdown in the table. Speculative decoding was ruled out as the cause (50/50 exact against a
zero-draft control), as was prompt rendering (0 differing artifacts of 12). The 32-item GSM8K
scores are 93.75% versus the MLX lane's 87.50% — a subset result, not a quality verdict.

## What is in it

| component | origin | licence | notes |
|---|---|---|---|
| language-model body, 625 sections | `ornith-ai/Ornith-1.5-35B-A3B` | MIT | repacked bit-exactly; `max_abs=0` against the source; 555 direct, 40 fused, 30 derived, no dequantize-requantize |
| vision tower, 333 bf16 sections | `ornith-ai/Ornith-1.5-35B-A3B` | MIT | byte-identical to Inco's Qwen3.6 tower, because the two models ship the same vision weights — verified by sampled value comparison against official Ornith |
| tokenizer | `ornith-ai/Ornith-1.5-35B-A3B` | MIT | vocabulary and merges verified identical; see the `\p{M}` caveat above |
| DFlash 2 draft, 7 files | `incoai/Qwen3.6-35B-A3B-Splash` | Apache-2.0 | reused unmodified; see below |
| container layout, manifest, packing scheme | Inco's schema 4 | Apache-2.0 | their format, our writer |

Integrity: `verify-package.py --full` passes over **56 artifacts, 20,949,446,234 bytes**, SHA-256
against the manifest.

### On redistributing the draft

Inco's `incoai/Qwen3.6-35B-A3B-Splash` card states that Apache-2.0 covers its DFlash 2 draft,
vision encoder and tokenizer. Its perpetual, irrevocable §2 grants permit redistribution;
this package complies with §4 by carrying the licence, retaining attribution and stating changes.

The draft's source repository, `incoai/Qwen3.6-35B-A3B-DFlash2`, is gated and returns HTTP 401;
the copy in the Apache-2.0 package is not. The gate signals an intent the licence does not encode.
If Inco prefers that this package not carry its draft, it will be removed on request.

## Speed

Same weights, same host, one session, arms alternated — Splash against an MLX lane with MTP:

| shape | speedup | band | pairs |
| --- | --- | --- | --- |
| single stream, short code | **1.47×** | 1.36–1.50 | 4 |
| four concurrent requests | **3.46×** | 3.38–3.47 | 3 |

Absolute tok/s are omitted because the same benchmark cell varied by 1.43× between runs;
ratios within each pair cancel that drift. Bands include only cells passing a CPU-quiescence
gate that was harsher on MLX, slightly favouring Splash. Excluded cells read 1.34× and 3.18–3.62×.

## Limitations

Natural photographs are untested; vision coverage is generated shapes and colours.
No safety evaluation has been performed. See above for draft, tokenizer and output differences.
Since 2026-09-30 the chat template renders a `system` message after the first as a system turn,
the one-line change incoai's Qwen3.6 Splash package makes to the upstream template. Earlier downloads
carry the upstream Ornith template, which raises on it: Splash 1.0.2 and runtimes that use the template
as shipped answer `messages could not be rendered`, so re-download
([discussion](https://huggingface.co/ezoushen/ornith-1.5-35b-a3b-splash/discussions/1)).

## Related

- [Umpire 35B-A3B](https://huggingface.co/ezoushen/umpire-35b-a3b-splash): a calibrated decision fine-tune of Ornith 1.5, also packaged for Splash.
- [Umpire MLX 4-bit](https://huggingface.co/ezoushen/umpire-35b-a3b-mlx-4bit): its MLX checkpoint, for reproducing Umpire served beside Ornith's MLX checkpoint without a second copy of the shared weights, with the [shared-expert-weights](https://github.com/ezoushen/splash/tree/shared-expert-weights) Splash fork.

## Credits and licence

Apache-2.0 for the draft, vision encoder, tokenizer container and packing format, from
[Inco AI](https://huggingface.co/incoai/Qwen3.6-35B-A3B-Splash). MIT for the Ornith weights, from
[Ornith AI](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B). See `LICENSE`.
