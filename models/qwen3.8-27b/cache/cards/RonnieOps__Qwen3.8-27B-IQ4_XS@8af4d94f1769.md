---
license: apache-2.0
base_model:
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
  - domain-calibrated
  - finance
  - sec-filings
  - trading
pipeline_tag: text-generation
language:
  - en
---

> [!IMPORTANT]
> **Purpose-built, not benchmark-chasing.** This quantization exists for one reason: to squeeze the maximum possible performance out of one specific machine — the author's RTX 5060 Ti 16 GB (eGPU) + Ryzen 5 9600X workstation. It is not an attempt to outdo other quants, base models, or fine-tunes, and it makes no such claim. Every number below is a measurement on that single machine, published as-is.

# Qwen3.8-27B-IQ4_XS — domain-calibrated quant (stock alignment)

> [!Note]
> A **domain-calibrated GGUF of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** for quantitative-finance, SEC-filings analysis, trading workflows, and agentic tool use: a flat IQ4_XS body with a measured mixed-precision layout and embedded MTP speculative decoding. Quality gates run against the BF16 reference on the same engine and data.

> [!Tip]
> Requires a recent **llama.cpp build with Qwen3.5-family (hybrid Gated DeltaNet) and MTP support** (main after 2026-08-27; the PRs are linked below). On Blackwell GeForce cards, build with `CMAKE_CUDA_ARCHITECTURES=120` and verify the smoke test — see [Requirements](#requirements).

## What makes this quant different

Two things, all measured rather than claimed:

**1. The importance matrix is domain-calibrated.** It blends a general-purpose corpus (bartowski's standard multi-source mix) with **~1M tokens of SEC-contract extraction and financial-analysis tasks**, chat-template-formatted to match real serving traffic.

**2. The layout is measured, not default.** `ffn_down` is kept at IQ4_XS while `token_embd` drops to q3_K — sensitivity measurement found demoting `ffn_down` to q3_K was the single largest KL driver at this size point.

On a 59-task domain benchmark (finance reasoning, covenant/extraction tool calls, JSON structured output, debugging, sysadmin), this quant scores **93.2% (55/59)** vs **89.8–91.5%** for the best-known community calibrations of the same model at the same bit rate, run identically on the same engine and GPU.

| Benchmark (59 tasks, temp 0)  | **This quant (this repo)** | Unsloth UD-IQ4_XS | orcarouter Unc IQ4_XS |
| ----------------------------- | -------------------------- | ----------------- | --------------------- |
| **Overall pass rate**         | **93.2% (55/59)**          | 89.8% (53/59)     | 89.8% (53/59)         |
| Tool calling (JSON args)      | 5/5                        | 5/5               | 5/5                   |
| Structured output             | 6/6                        | 6/6               | 6/6                   |
| Debugging (execution-graded)  | 5/5                        | 5/5               | 5/5                   |
| Sysadmin (command generation) | 7/7                        | 7/7               | 7/7                   |
| Long-context recall           | 3/3                        | 3/3               | 3/3                   |
| Codegen (execution-graded)    | 9/10                       | 8/10              | 8/10                  |
| Reasoning (numeric finance)   | 8/10                       | 7/10              | 7/10                  |
| Architecture/design           | 5/6                        | 5/6               | 5/6                   |

_(Duplicate harness runs reproduced these scores exactly.)_

On a larger 115-task extended suite (adds execution-graded multi-language codegen/debugging, multi-turn agentic tool sessions, and a 60K-token regression task), this quant scores **91/115** vs 91/115 for UD-IQ4_XS — the domain gap concentrates in the finance/agent reasoning categories. On the 127-task v4 instrument (2026-09-20/21): the pruned daily scores **101/127** text+MTP thinking-off, rising to **105/127 at high reasoning effort** (effort pays; xhigh adds nothing over high); MTP and the vision projector are each exactly quality-neutral (every vision cell identical to its text twin). The full-vocabulary build ships in this repo as `Qwen3.8-27B-IQ4_XS-fullvocab-E3.gguf` (same `-fullvocab-<revision>` naming as the uncensored sibling's E4 archive).

## Variants in this family

This card documents the **full-vocabulary build**. The published file in this repository is its **vocabulary-pruned variant** — same recipe, pruned with the quantization toolchain's level-2 policy ("P2": ASCII + accented Latin + math/typography symbols), the same keep-set as the uncensored sibling:

| Variant                                                | File                                   | Vocabulary                                                                                                             | Size (on disk, measured 2026-09-17) | 115-task gate       | Notes                                                                                                                                                                                                     |
| ------------------------------------------------------ | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Full-vocab build** (this card, in this repo)         | `Qwen3.8-27B-IQ4_XS-fullvocab-E3.gguf` | 248,320 — multilingual (en/zh) intact                                                                                  | 14.14 GiB                           | 91/115 @24K ctx     | the build every number in this card was measured on, incl. KL 4.14% / 93.4%                                                                                                                               |
| **Vocabulary-pruned variant** (published in this repo) | `Qwen3.8-27B-IQ4_XS.gguf`              | 141,141 — ASCII + Latin-1/Latin-Extended-A accents + math/typography symbols (same keep-set as the uncensored sibling) | 13.49 GiB                           | **92/115 @32K ctx** | −0.65 GiB vs full vocab → 64K context fits natively; non-Latin scripts garble exactly as on the ASCII sibling; prune is fidelity-neutral (kept embedding rows gathered in quantized space, bit-identical) |

KL divergence against the saved BF16 logits is structurally undefined for the pruned variant (logit dimensions differ — see the uncensored sibling's fidelity note); the 4.14% / 93.4% fidelity figures above belong to the full-vocab build and bound the shared quantized body.

## Fidelity

Measured on the full wiki validation corpus against saved BF16 logits, identical settings — this isolates generic-text fidelity:

| Metric                | **This quant (this repo)** | Unsloth UD-IQ4_XS |
| --------------------- | -------------------------- | ----------------- |
| KL divergence, RMS Δp | **4.14% ± 0.30%**          | 4.35%             |
| Same top-p agreement  | 93.40% ± 0.55%             | 93.9%             |

The best RMS fidelity of any Qwen3.8-27B quant measured on this machine, community quants included. Perplexity (full wiki corpus): **6.4248 ± 0.04** for the ASCII-pruned sibling of this family; bartowski's published full-file anchors are 6.744 (BF16) and ≈6.77 (IQ4_XS).

## Quantization recipe

Source: **Qwen/Qwen3.8-27B** BF16 (stock, censored; unsloth BF16 repack used as the quantize source because it carries the MTP/nextn tensors). Per-tensor layout over a flat **IQ4_XS** body:

| Tensor class                      | Type                                                                          | Rationale                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Body (attn, FFN, GDN projections) | `IQ4_XS`                                                                      | bandwidth-optimal on this GPU (448 GB/s class)                               |
| `token_embd` (248,320-vocab)      | `q3_K`                                                                        | large vocab tensor; sensitivity-verified                                     |
| `ffn_down`                        | `IQ4_XS` — **kept high**                                                      | measured KL driver at this size point; demoting it cost 0.8–0.9 top-p in A/B |
| MTP / nextn (blk.64, 15 tensors)  | embedded, quantized with the body                                             | enables native speculative decoding                                          |
| Importance matrix                 | bartowski general + **~1M-token SEC-finance corpus**, importance donor IQ4_XS | see fidelity note                                                            |

- **Size:** 14.14 GiB (measured on disk)
- **MTP:** the 15 `blk.64` multi-token-prediction tensors are **embedded** — speculative decoding works out of the box, no separate draft file.

## Serving (measured config)

```bash
llama-server -m Qwen3.8-27B-IQ4_XS.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -ngl 99 -c 32768 -np 1 -fa on -ctk q4_0 -ctv q4_0 \
  -b 2048 -ub 512 -t 6 --jinja --cache-reuse 256 \
  --host 127.0.0.1 --port 8080 --metrics --slots
```

Measured on the development hardware (RTX 5060 Ti 16 GB, eGPU, custom SM120/Zen 5 build):

| Metric                   | Value                                  |
| ------------------------ | -------------------------------------- |
| Generation (batch 1)     | **~50 t/s** with MTP d3; ~26 t/s plain |
| Prompt processing        | ~810–1,030 t/s                         |
| TTFT (median / p90)      | 0.23 / 0.26 s (harness)                |
| MTP draft acceptance     | 0.49–0.89 by prompt                    |
| VRAM (24–32K ctx, q4 KV) | 15.3–15.6 GiB                          |

- **Context:** 64K fits natively at q4 KV with the pruned variant (the full-vocabulary build of this family OOMs at 64K on 16 GB); 96K+ requires KV-cache streaming (see below).
- **Vision:** the stock Qwen `mmproj` projector works with this quant (CPU-encode recommended on 16 GB). With the projector loaded, disable MTP — the two do not fit together reliably at 16 GB. Quantized projector variants (Q8_0 / Q4_0 / a merger-upgraded Q4_0 mix) ship in our mmproj repository, all measured at parity with F16 on objective and real-image batteries.
- **Recommended chat template**: [`froggeric/Qwen-Fixed-Chat-Templates`](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) v22.5 (`chat_template.jinja`) — A/B measured on this family: identical harness pass rate, **−27% reasoning tokens at xhigh effort**, tool-calling unaffected; it also renders mid-conversation system messages that the stock template rejects. Serve with `--chat-template-file <file> --reasoning-format deepseek`.
- **Sampling (per the Qwen3.8 model card):** thinking mode — temp 1.0, top_p 0.95, top_k 20; non-thinking — temp 0.7, top_p 0.80, top_k 20, presence_penalty 1.5. Thinking toggles per request via `chat_template_kwargs: {"enable_thinking": true|false}`; for short answers disable thinking and keep `max_tokens ≥ 256`.
- **OpenAI-compatible**: `/v1/chat/completions` and `/v1/completions` work as expected; tool calling is clean (parallel calls included).

## Context behavior (measured on this architecture)

The hybrid Gated-DeltaNet architecture has a tiny KV cache (16 of 64 layers carry attention KV). Two things are measured on this machine:

1. decode-throughput decays above ~80K of context position (upstream issue #27623): 32K ≈ 21–23 t/s plain, 64K ≈ 20 t/s, 128K ≈ 7 t/s, 232K ≈ 5 t/s;
2. **KV precision is speed-free** (q8_0 vs q4_0 identical within noise) — use q4 KV for headroom.

**Recommendation:** 24K–32K is the daily envelope in the MTP config. For longer documents, retrieval/summaries beat ever-growing context past ~80K of position. **KV-cache streaming runtimes** (community forks, e.g. adaptive-KV-streaming) lift both limits: on our 16 GB card the same architecture serves 128K–252K contexts at a flat ~26 t/s decode with measured quality unchanged (115-task scores flat to 252K) — that capability ships with those forks, not with stock llama.cpp.

**Best measured configuration by context window** (16 GB card; rows ≥96K were measured on the uncensored sibling's file — identical size class and fit envelope, same recipe family):

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

- The domain benchmark is a **custom suite built for this deployment** (finance reasoning, SEC extraction, tool calls, sysadmin, execution-graded codegen) — it is not a public academic benchmark; other domains will see different (likely smaller) gains.
- Alignment is **stock** — this model refuses like the original. See the uncensored sibling from the same lineage, published separately.
- Full 248,320-token vocabulary: multilingual capability is intact (en/zh and scripts beyond). If you only serve English/code, a vocab-pruned sibling of this family trades multilingual text for ~0.65 GiB and native long-context fit.
- Vision was functional but not the model-side optimization target; the projector side is covered — quantized projector variants (measured at parity with F16) ship in our mmproj repository.

## Calibration study summary

Three imatrix variants of the identical recipe were built and gated (duplicate runs reproduced every score):

| Variant                                 | Domain corpus   | Domain harness (59 tasks) | wiki KL RMS / top-p                     |
| --------------------------------------- | --------------- | ------------------------- | --------------------------------------- |
| General only                            | none            | 52/59 (88.1%)             | —                                       |
| General + 377K finance                  | SEC 377K        | 54/59 (91.5%)             | 4.93 / 91.7                             |
| **General + ~1M finance (this family)** | **SEC ~1M**     | **55/59 (93.2%)**         | **4.14 / 93.4** (with the ffn_down fix) |
| Unsloth UD-IQ4_XS reference             | Unsloth general | 53/59 (89.8%)             | 4.35 / 93.9                             |

Domain calibration scales monotonically with corpus size and beats the community calibration on the target workload. A Q8-donor importance matrix was tested as a control and showed no measurable difference (negative result, documented).

## Acknowledgements

- [Qwen team](https://huggingface.co/Qwen) — the base model (Apache-2.0) and architecture.
- [orcarouter](https://huggingface.co/orcarouter) — the abliterated full-precision base of the uncensored lineage.
- [bartowski](https://huggingface.co/bartowski) — the general-tier calibration corpus and the per-tensor layout methodology this family's sensitivity work builds on.
- [Unsloth](https://huggingface.co/unsloth) — the UD-IQ4_XS reference quant used as the comparison baseline, and the MTP-bearing BF16 repack used as a quantize source.
- [bsaleh03](https://github.com/bsaleh03/ASCII-Condensed-prune-tools) — the ASCII-Condensed vocabulary-pruning toolchain (audit → prune → verify with policy replay).
- [froggeric](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — the Qwen-Fixed-Chat-Templates jinja template (v22.5), the measured A/B winner this family serves with.
- [ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp) — the runtime, the hybrid-architecture support, and MTP speculative decoding.

## License

Apache-2.0, inherited from the base model. The SEC-contract calibration corpus is derived from public SEC filings (EDGAR); users are responsible for compliance with their own use case.
