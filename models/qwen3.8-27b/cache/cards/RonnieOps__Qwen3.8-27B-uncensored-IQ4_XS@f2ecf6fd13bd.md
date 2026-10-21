---
license: apache-2.0
base_model:
  - orcarouter/Qwen3.8-27B-Uncensored-GGUF
  - Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.8
  - qwen3.5
  - quantized
  - iq4-xs
  - mtp
  - multi-token-prediction
  - llama-cpp
  - uncensored
  - abliterated
  - domain-calibrated
  - vocab-pruned
  - ascii
  - finance
  - sec-filings
  - trading
pipeline_tag: text-generation
language:
  - en
---

> [!IMPORTANT]
> **Purpose-built, not benchmark-chasing.** This quantization exists for one reason: to squeeze the maximum possible performance out of one specific machine — the author's RTX 5060 Ti 16 GB (eGPU) + Ryzen 5 9600X workstation. It is not an attempt to outdo other quants, base models, or fine-tunes, and it makes no such claim. Every number below is a measurement on that single machine, published as-is.

> [!WARNING]
> This model has had its alignment-based refusals removed (abliterated upstream base). It will respond to prompts the stock model refuses, including harmful ones. **You are the safety layer.** Do not expose it to untrusted input sources, and do not connect it to autonomous tooling without understanding what that means.

# Qwen3.8-27B-uncensored-IQ4_XS — uncensored, domain-calibrated, ASCII-pruned

> [!Note]
> An **uncensored, domain-calibrated, vocabulary-pruned GGUF of Qwen3.8-27B** (27.8B, hybrid Gated-DeltaNet + gated attention): the abliterated base produced by [orcarouter](https://huggingface.co/orcarouter), re-quantized through a measured domain-calibration pipeline (~1M-token SEC-filings/finance importance matrix blended with a general corpus), then pruned to **English + accented Latin + math/typography symbols** (141,141 of 248,320 tokens). Quality gates run against the BF16 reference on the same engine and data.

> [!Tip]
> Requires a recent **llama.cpp build with Qwen3.5-family (hybrid Gated DeltaNet) and MTP support** (main after 2026-08-27; the PRs are linked below). On Blackwell GeForce cards, build with `CMAKE_CUDA_ARCHITECTURES=120` and verify the smoke test — see [Requirements](#requirements).

> [!WARNING]
> **English and accented-Latin text only.** Scripts outside the kept set (CJK, Arabic, Cyrillic, Thai, …) fall back to one token per UTF-8 byte and the model reads them as garbled text, because it was never trained on those byte sequences. If you need multilingual text, use a full-vocabulary quant instead. See [Limitations](#limitations).

## What makes this quant different

Three things, all measured rather than claimed:

**1. The abliteration is gentle.** The upstream base was ablated with directional-ablation methods while preserving quality — verified before building on it: **0/3 refusal probes answered with refusals** (all produced real content), and generic-text fidelity of the full-vocab sibling against the BF16 reference measured RMS Δp **4.33%** / top-p agreement **93.9%** (E4 revision: token embedding at q4_K — parity with the best community calibration on both axes, 2026-09-19).

**2. The importance matrix is domain-calibrated.** It blends a general corpus with **~1M tokens of SEC-contract extraction and financial-analysis tasks**, chat-template-formatted to match real serving traffic. On a 59-task domain benchmark, this quant scores **93.2% (55/59)** vs **89.8%** for the best community calibrations of the same model at the same bit rate — a +3.4-point controlled A/B, isolated to the imatrix.

**3. The vocabulary is pruned with verification.** 141,141 tokens survive (ASCII + Latin-1/Latin-Extended-A accents + math/typography symbols + specials + byte-fallback). Rows are gathered in quantized space — **no dequantize/requantize, every kept row is bit-identical to the source**. All 864 non-vocab tensors (including the MTP head) verified byte-identical. The prune buys **−0.65 GiB**, which on a 16 GB card is the difference between 64K context fitting and not fitting.

| Benchmark                           | **This quant (this repo)**                          | Full-vocab build (same recipe) | Unsloth UD-IQ4_XS     |
| ----------------------------------- | --------------------------------------------------- | ------------------------------ | --------------------- |
| 59-task domain harness (temp 0)     | **55/59 (93.2%)**                                   | 55/59 (93.2%)                  | 53–54/59 (89.8–91.5%) |
| 115-task extended suite @32K MTP    | **92/115**                                          | 91–92/115                      | 91/115                |
| 127-task suite v4 @32K (2026-09-20) | **100/127** == predecessor with **zero task flips** | —                              | —                     |
| same, thinking @high effort         | **105/127** (+5 vs thinking-off)                    | —                              | —                     |
| KL RMS Δp / top-p vs BF16           | see full-vocab build                                | **4.33% / 93.9%** (E4)         | 4.35% / 93.9%         |
| Refusal probes                      | **0 refused**                                       | —                              | refused               |

_(Duplicate harness runs reproduced these scores exactly.)_

## Variants in this family

This card documents the **served daily — the E4-ASCII revision (2026-09-20)**. Both variants ship in this repository:

| Variant                                        | File                                              | Vocabulary                                                                                                           | Size (on disk) | Suite gate                                                                       | Notes                                                                                                                |
| ---------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Served daily** (this card)                   | `Qwen3.8-27B-uncensored-IQ4_XS.gguf`              | 141,141 — ASCII + Latin-1/Latin-Extended-A accents + math/typography symbols (same keep-set as the censored sibling) | 13.58 GiB      | **100/127** @32K MTP, zero task flips vs predecessor; **105/127** at high effort | E4 revision: q4_K embedding; the predecessor E3-ASCII build remains retrievable in this repo's revision history      |
| **Full-vocabulary anchor** (also in this repo) | `Qwen3.8-27B-uncensored-IQ4_XS-fullvocab-E4.gguf` | 248,320 — multilingual (en/zh) intact                                                                                | 14.32 GiB      | 91/115 @32K                                                                      | the fidelity reference: KL 4.33% RMS / 93.9% top-p vs BF16 — parity with the best community calibration on both axes |

KL divergence against saved BF16 logits is structurally undefined for the pruned variant (logit dimensions differ — see [Fidelity](#fidelity)); the 4.33% / 93.9% figures belong to the full-vocab anchor and bound the shared quantized body. The stock-alignment sibling of this family is published separately.

## Fidelity

- **Kept-vocabulary rows are bit-identical** to the source quant — the prune cannot change behavior on text the kept tokens can represent.
- Standalone perplexity (full wiki validation corpus, identical settings): **6.4248 ± 0.04**. External full-file anchors for this architecture: 6.744 (BF16) and ≈6.77 (community IQ4*XS). A vocab prune slightly \_raises* PPL mechanically (the softmax renormalizes over fewer candidates) — the sibling measurement bounds this effect at ≈+0.2%.
- KL divergence against saved BF16 logits is structurally undefined for a pruned-vocab model (logit dimensions differ) and was not faked.

## Quantization recipe

Source: **orcarouter F16 (abliterated)** — then vocabulary pruned (rows gathered in quantized space). Per-tensor layout over a flat **IQ4_XS** body:

| Tensor class                      | Type                                                                          | Rationale                                                                                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Body (attn, FFN, GDN projections) | `IQ4_XS`                                                                      | bandwidth-optimal on this GPU (448 GB/s class)                                                                                            |
| `token_embd`                      | `q4_K` — **the E4 revision (2026-09-20)**                                     | KL-guided search found the embedding tier closes the residual fidelity gap (full-vocab E4: 4.33% RMS / 93.9% top-p = parity on both axes) |
| `ffn_down`                        | `IQ4_XS` — **kept high**                                                      | measured KL driver at this size point; demoting it cost 0.8–0.9 top-p in A/B                                                              |
| MTP / nextn (blk.64, 15 tensors)  | embedded, quantized with the body                                             | enables native speculative decoding                                                                                                       |
| Importance matrix                 | bartowski general + **~1M-token SEC-finance corpus**, importance donor IQ4_XS | see the calibration study below                                                                                                           |

- **Size:** 13.58 GiB (~1.1B parameters removed from `token_embd` + `output`; +0.09 GiB vs the E3 revision buys the q4_K embedding — the measured fidelity upgrade)

- **MTP:** the 15 `blk.64` tensors are embedded and verified byte-identical — speculative decoding works out of the box
- **Tokenizer:** specials, all 256 byte-fallback tokens, and partial-UTF-8 fragments retained; special ids remapped; merges verified to reference only kept tokens in original priority order

## Serving (measured config)

```bash
llama-server -m Qwen3.8-27B-uncensored-IQ4_XS.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 99 -c 32768 -np 1 -fa on -ctk q4_0 -ctv q4_0 \
  -b 2048 -ub 512 -t 6 --jinja --cache-reuse 256 \
  --host 127.0.0.1 --port 8080 --metrics --slots
```

Measured on the development hardware (RTX 5060 Ti 16 GB, eGPU, custom SM120/Zen 5 build):

| Metric                       | Value                                                                 |
| ---------------------------- | --------------------------------------------------------------------- |
| Generation (batch 1, MTP d3) | **59.5 t/s** decode at ~28K position; 0.87–0.89 mean draft acceptance |
| Generation (plain)           | ~26 t/s short context; **21.0 t/s at 64K native**                     |
| Prompt processing            | ~830–875 t/s at depth                                                 |
| VRAM                         | 15.4 GiB (32K + MTP); **15.2 GiB (64K, no MTP)**                      |

- **Context:** 64K fits natively at q4 KV — on this card the unpruned sibling of the same recipe OOMs at 64K. 96K+ requires KV-cache streaming (out of scope here).
- **Vision:** the stock Qwen `mmproj` projector works with this quant (CPU-encode recommended on 16 GB). With the projector loaded, disable MTP — the two do not fit together reliably at 16 GB. Quantized projector variants (Q8_0 / Q4_0 / a merger-upgraded Q4_0 mix) ship in our mmproj repository, all measured at parity with F16 on objective and real-image batteries — and the 2026-09-20/21 factorial sweep proved vision **exactly** quality-neutral on the 127-task suite (every vision cell identical to its text twin, cell for cell).
- **Recommended chat template**: [`froggeric/Qwen-Fixed-Chat-Templates`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) v22.5 — A/B measured on this family: identical harness pass rate, **−27% reasoning tokens at xhigh effort**, tool-calling unaffected, and it renders mid-conversation system messages that the stock template rejects.
- **Sampling (per the Qwen3.8 model card):** thinking mode — temp 1.0, top_p 0.95, top_k 20; non-thinking — temp 0.7, top_p 0.80, top_k 20, presence_penalty 1.5. Thinking toggles per request via `chat_template_kwargs: {"enable_thinking": true|false}`; for short answers disable thinking and keep `max_tokens ≥ 256`.
- **OpenAI-compatible**: `/v1/chat/completions` and `/v1/completions` work as expected; tool calling is clean (parallel calls included).

## Context behavior (measured on this architecture)

The hybrid Gated-DeltaNet architecture has a tiny KV cache (16 of 64 layers carry attention KV), and the family has a measured decode-throughput decay above ~80K of context position (upstream issue #27623): 32K ≈ 21–23 t/s plain, 64K ≈ 20 t/s, 128K ≈ 7 t/s, 232K ≈ 5 t/s. **Recommendation:** 32K in the MTP config for daily work; retrieval/summaries beat ever-growing context past ~80K of position. **KV-cache streaming runtimes** lift this further: with a community adaptive-KV-streaming fork this exact file serves 128K contexts (~14.8 t/s at 105K position) and its 115-task scores stay flat out to 252K — measured on our 16 GB card; ships with those forks, not stock llama.cpp.

**Best measured configuration by context window** (this file, 16 GB card):

| Context  | Best config                                                                     | Engine         | Decode                                 |
| -------- | ------------------------------------------------------------------------------- | -------------- | -------------------------------------- |
| 8–12K    | vision + MTP d3 (Q4_0 projector) — the only window where both co-exist          | stock          | ~50–70 t/s                             |
| 16–32K   | text + MTP d3 (daily) — or vision-first (Q8_0 projector; MTP auto-off)          | stock          | ~67–70 t/s text · ~27 vision           |
| 48–64K   | text-only, q4_0 KV, no MTP (the MTP draft mirrors the window and OOMs past 32K) | stock          | ~21–27 t/s                             |
| 96K      | text-only, kvarn4/4 KV (variance-normalized)                                    | beellama fork  | ~20 t/s @77K position                  |
| 128–252K | text-only, KV-stream arena 1536 MiB, q8_0 K / q4_0 V                            | kv-stream fork | ~14.8 t/s @105K · quality flat to 252K |

**Using the >64K engines** — both are llama.cpp forks; pick by context window:

**kv-stream fork** ([RaymondHuang210129/llama.cpp-adaptive-kv-streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming), branch `feature/kv-stream-phase-arena`) — for 128K–252K. Build with the fork's pre-rename FA flag (`-DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=120`), then serve 128K+ from a ~15 GiB footprint:

```bash
llama-server -m Qwen3.8-27B-<variant>.gguf -ngl 99 -c 131072 -np 1 -fa on \
  -ctk q8_0 -ctv q4_0 -b 512 -ub 512 --kv-stream-arena-mib 1536
```

Text-only, **no MTP** (the shared arena rejects speculative batches), no `--cache-reuse`. Non-resident KV pages live pinned in host RAM — budget ~6 GiB of system memory at 252K.

**beellama fork** ([Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp)) — for 96K native, no streaming machinery, with its variance-normalized KV quant:

```bash
llama-server -m Qwen3.8-27B-<variant>.gguf -ngl 99 -c 98304 -np 1 -fa on \
  -ctk kvarn4 -ctv kvarn4 -b 2048 -ub 512
```

Keep speculative decoding **off** on hybrid-GDN models with this fork (its prompt-cache rollback path is measured-unsafe with spec until their fix ships), and treat 96K as the ceiling — deeper contexts fit at load but OOM on real long prompts.

## Requirements

- llama.cpp with **Qwen3.5-family hybrid architecture support** (merged upstream 2026-08-27 or later) and **MTP speculative decoding** ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673))
- ~15.4 GiB VRAM for the 32K MTP config; 16K ctx runs comfortably alongside other GPU apps

## Limitations

- **Non-Latin scripts break** (see the warning above). Accented Latin (é, ü, ñ) is kept — European names in financial documents survive. _(Correction, 2026-09-17: an earlier 1×1-pixel result suggested a projector-quantization effect; follow-up testing disproved it — all projector levels fail degenerate images identically. It is a model-family artifact, not a projector effect.)_
- The domain benchmark is a **custom suite built for this deployment** — not a public academic benchmark; other domains will see different (likely smaller) gains.
- The upstream abliteration method is unpublished (gated repository); its quality was validated by our gates rather than by method disclosure.
- The MTP draft head was not itself ablated; draft acceptance may dip on sensitive completions (greedy verification keeps output lossless).
- Decode throughput decays past ~80K of context position (upstream issue); this is an architecture/runtime limitation, not specific to this quant.
- Not a general-purpose uncensored model for public deployment — alignment removal is total, and the security implications of that are yours.

## Calibration study summary

Three imatrix variants of the identical recipe were built and gated (duplicate runs reproduced every score):

| Variant                                 | Domain corpus   | Domain harness (59 tasks) | wiki KL RMS / top-p                                    |
| --------------------------------------- | --------------- | ------------------------- | ------------------------------------------------------ |
| General only                            | none            | 52/59 (88.1%)             | —                                                      |
| General + 377K finance                  | SEC 377K        | 54/59 (91.5%)             | 4.93 / 91.7                                            |
| **General + ~1M finance (this family)** | **SEC ~1M**     | **55/59 (93.2%)**         | 5.10 / 91.8 (V-C layout) → **4.33 / 93.9** (E4 layout) |
| Unsloth UD-IQ4_XS reference             | Unsloth general | 53/59 (89.8%)             | 4.35 / 93.9                                            |

Domain calibration scales monotonically with corpus size. A Q8-donor importance matrix was tested as a control and showed no measurable difference (negative result, documented).

## Acknowledgements

- [Qwen team](https://huggingface.co/Qwen) — the base model (Apache-2.0) and architecture.
- [orcarouter](https://huggingface.co/orcarouter) — the abliterated full-precision base of the uncensored lineage.
- [bartowski](https://huggingface.co/bartowski) — the general-tier calibration corpus and the per-tensor layout methodology this family's sensitivity work builds on.
- [Unsloth](https://huggingface.co/unsloth) — the UD-IQ4_XS reference quant used as the comparison baseline, and the MTP-bearing BF16 repack used as a quantize source.
- [bsaleh03](https://github.com/bsaleh03/ASCII-Condensed-prune-tools) — the ASCII-Condensed vocabulary-pruning toolchain (audit → prune → verify with policy replay).
- [froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — the Qwen-Fixed-Chat-Templates jinja template (v22.5), the measured A/B winner this family serves with.
- [ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp) — the runtime, the hybrid-architecture support, and MTP speculative decoding.

## License

Apache-2.0, inherited from the base model. The SEC-contract calibration corpus is derived from public SEC filings (EDGAR); users are responsible for compliance with their own use case, and — this being an alignment-removed model — for every output it produces.
