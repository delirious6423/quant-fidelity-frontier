---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - ukisai/Swift-1.5-Qwen3.8-27b
  - Qwen/Qwen3.8-27B
  - z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
  - ninfer
  - ninfer-v3
  - qwen3.8
  - qwen3_5
  - swift
  - swift-1.5
  - nvfp4
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
  - dflash2
  - speculative-decoding
---

# Swift-1.5 Qwen3.8-27B nvfp4full + DFlash2 for NInfer — **v3 container**

An **all-NVFP4** artifact of [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)
for the [NInfer](https://github.com/Neroued/ninfer) engine, with z-lab's DFlash2
speculative drafter and the indexed proposal head embedded. Native **v3** container.

It replaces [kaushikvira/Qwen3.8-27B-swift-abliterated-nvfp4full-dflash2-NInfer-v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift-abliterated-nvfp4full-dflash2-NInfer-v3)
(Swift-1.0-based) as our production profile after a same-session A/B — see below.

## Credits & provenance (full chain)

**All credit for the model itself goes to the authors below** — this artifact is a
format conversion + quantization of their work, with no fine-tuning of our own:

| Role | Model | Author | SHA / commit |
|---|---|---|---|
| **Source weights (BF16)** | [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) | **ukisai (UkisAI)** | 18-shard BF16 safetensors, published 2026-09-24 |
| **Upstream base** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | **Qwen team** | Swift-1.5 = RL + OPD post-training derivative of Qwen3.8-27B |
| **DFlash2 drafter** | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | **z-lab** | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| **Engine** | [Neroued/ninfer](https://github.com/Neroued/ninfer) | **Neroued** | v3 container, ≥ `98dada0e` |
| Quantization-runbook inspiration | [Barding-Defense/Qwen3.8-27B-huihui-abliterated-NVFP4-NInfer](https://huggingface.co/Barding-Defense/Qwen3.8-27B-huihui-abliterated-NVFP4-NInfer) | Barding-Defense | community ninferization walkthrough (credited wherever its steps shaped our pipeline) |
| Prior serving lineage | [ukisai/Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b) / [d0xin/Swift-Qwen3.8-27B-Uncensored-BF16](https://huggingface.co/d0xin/Swift-Qwen3.8-27B-Uncensored-BF16) | ukisai / d0xin | our previous profile |

Swift 1.5 is UkisAI's reasoning-efficient derivative of Qwen3.8-27B: scaled-up
RL + OPD post-training focused on long-horizon agentic and coding tasks
(LiveCodeBench, Terminal-Bench improvements), with pathological overthinking
penalized — "compressed" token usage while maintaining accuracy.

## What we did (the technique)

Identical pipeline to our Swift-1.0 build (see the sibling model card):

1. **All-NVFP4 quantization** with `llm-compressor` (512 Ultrachat calibration
   samples, seq 2048, sequential pipeline): NVFP4 W4A4 group-16 on every text
   projection (MLP gate/up/down all 64 layers, attention Q/K/V/O, GDN
   in_proj_qkv/z/out_proj), W8G32 token embedding + output head, official q6/q8
   vision allocation, MTP + DFlash2 in BF16.
2. **Global-divisor normalization** (our tool): llm-compressor emits per-module
   `weight_global_scale`; the engine's native A4 route requires one contiguous
   fused parent per attention group (GDN `qkvz` 16,384 rows, attn `qkgv` 14,336
   rows). We unify each packing group's divisor to `D = min(dᵢ)` and rescale the
   E4M3 block scales by `D/dᵢ` (RNE, shrink-only). For Swift-1.5 the rescale was
   **exactly representable — 0.0000% re-encode error across all 272 groups**.
3. **v3 conversion** with `tools/convert` (`--components text,vision,mtp,dflash2 --proposal`).

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_swift15_nvfp4full-dflash2.ninfer` |
| Size | 19,782,447,364 bytes (18.42 GiB) |
| **SHA-256** | `16f313c043c06a19f7c27d74f8259ed9d86c8cb5beed631efbb96f392b373d96` |
| Container version | **3** (`NINFER\0\x03`) |
| NInfer model ID | `qwen3.8-27b` (serve as `Qwen3.8-27B`) |
| Stored objects | 1,590 (Text + Vision + MTP + DFlash2 + indexed proposal head) |
| Formats | nvfp4 ×256 (fused text parents), q8_g32 ×30, q4/q5/q6 (vision), bf16 remainder |
| Hash manifest | `SHA256SUMS`; full conversion contract in `.conversion.json` |

## Serving (RTX 5090 32 GB, single GPU)

```
ninfer-serve qwen3_8_27b_swift15_nvfp4full-dflash2.ninfer \
  --model-id Qwen3.8-27B --max-context 262144 --kv-capacity auto --kv-dtype k8v4 \
  --max-concurrency 4 --default-max-tokens 32768 --prefill-chunk 4096 \
  --temperature 0.9 --min-p 0.05 --spec dflash2 --draft-tokens 7 --lm-head-draft \
  --host-kv-mib 49152 --host-state-slots 16 --vision \
  --default-thinking-budget 16384 --preserve-thinking
```

> **Stock-engine note:** `--image-token-budget` and the `dflash2` speculative
> backend exist only in our forked engine build. On the stock NInfer engine,
> drop both (`--image-token-budget 1280` and `--spec dflash2 --draft-tokens 7
> --lm-head-draft`): vision input is capped at 32,768 tokens by a compile-time
> constant, and the stock `dflash` backend cannot be combined with `--vision`.
> The model runs fine without speculative decoding. The image-token-budget
> flag is upstream PR [Neroued/ninfer#61](https://github.com/Neroued/ninfer/pull/61)
> (open) — it lands in the stock engine when that PR merges; the fork build
> carries it in the meantime.

Live capacity on one 5090: **weights 18.0 GiB, device KV pool 308,736 tokens
(auto), 48 GiB pinned host-KV arena, full 262,144-token context at concurrency 4.**

## Benchmarks — same-session A/B vs our Swift-1.0-based production

Both sides served by the same engine build, fresh generations, temp 0,
no cache reuse, single RTX 5090.

| Benchmark | Swift-1.0-abliterated (prev. prod) | **Swift-1.5 (this artifact)** | Δ |
|---|---|---|---|
| IFBench prompt-strict (n=300) | 66.3 | **69.0** | +2.7 |
| IFBench prompt-loose | 70.3 | **72.7** | +2.4 |
| IFBench instr-strict | 67.7 | **70.4** | +3.2* |
| IFBench instr-loose | 71.2 | **73.6** | +4.3 |
| GSM8K-200 accuracy | 95.5% (191/200) | 95.0% (190/200) | −0.5 pp (n=200 noise) |
| Gate decode (256-tok samples) | 148.4–159.0 tok/s | **158.6 tok/s** | faster |
| Perf decode (mean of 3) | 148.7 tok/s | **160.8 tok/s** | +8.1% |
| Prefill @ 200k ctx | 3,260 tok/s | 3,269 tok/s | = |
| Weights in VRAM | 18.0 GiB | 18.0 GiB | = |
| Artifact size | 18.42 GiB | 18.42 GiB | = |
| Device KV pool (auto) | 308,736 tok | 308,736 tok | = |
| Long-context recall (needle) | 250,031 tok EXACT ×3 depths | 250,031 tok EXACT ×3 depths (23/24 overall, one transient empty response @135k) | = |
| Thinking-token efficiency | — | fewer (Swift-1.5 post-training penalizes overthinking) | faster agentic runs |

\* per-mode deltas: prompt-strict +2.7, prompt-loose +2.4, instr-strict +3.2, instr-loose +4.3 (in absolute points).

## Quality-oriented sibling: SmoothQuant + NVFP4-MLP (measured A/B)

A second artifact of the **same base model and drafter**, quantized with
**SmoothQuant applied before the 4-bit step** (plus a finer Q8 head/embed grid and a
lossless BF16 vision tower):
[kaushikvira/Qwen3.8-27B-swift15-smoothquant-nvfp4mlp-dflash2-NInfer-v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift15-smoothquant-nvfp4mlp-dflash2-NInfer-v3).

Both were served through the same engine build, on the same GPU, with the same KV format,
harness and served model id — only the artifact file differed.

| Benchmark | **this artifact** (18.42 GiB, all-NVFP4) | SmoothQuant sibling (22.80 GiB) | Δ |
|---|---|---|---|
| **GPQA-Diamond** (200 samples) | **87.00 % ± 4.2** | **91.00 % ± 3.4** | +4.00 pp for the sibling |
| **AIME 2026** (30 questions) | **83.33 % (25/30)** | **90.00 % (27/30)** | +6.67 pp (+2 q) for the sibling |
| Time per GPQA question | **10.96 s** | 13.99 s | this artifact is **27.6 % faster** |
| Aggregate tokens/s (GPQA) | **597** | 483 | +19.1 % for this artifact |
| Weights on disk | **18.42 GiB** | 22.80 GiB | this artifact is 4.38 GiB smaller |

The deltas sit within roughly 1σ at n=200 (combined SE ±5.4 pp), but they move the same way
on both suites and match the expected mechanism: W4A4 losses come from **activation
outliers**, and the sibling removes them (SmoothQuant re-scales activations into the
weights) before the 4-bit grid is fitted.

**Guidance:** this artifact is the throughput/VRAM-efficient choice and remains the default
for volume decoding; the SmoothQuant sibling is the choice when a wrong answer is more
expensive than ~4 GiB and ~28 % latency on long-reasoning questions.

## Notes

- This build is **not abliterated** — Swift-1.5 carries the base Qwen3.8 refusal
  behavior. Our Swift-1.0-based production was uncensored (huihui-style via
  d0xin). If you need uncensored, use our Swift-1.0 artifact, or apply a
  refusal-direction edit to Swift-1.5 (no community abliterated Swift-1.5 exists
  yet at publication time).
- Same-session methodology, response caches keyed per-model to avoid reuse.

## Measurement conditions

All numbers on this card were measured on a single RTX 5090 32 GB (driver 595.71.05):

- **GPU power cap 450 W** (stock 575 W) and **SM clock pinned 2280 MHz** via the
  `nv-power-limit.service` systemd unit — a thermal-efficiency mod (load power
  ~450 W → ~330 W) with no measurable tok/s loss vs stock.
- Telemetry correlated with the runs below (70 s thermal-log samples plus a
  2 s-sampled instrumented run): peak board draw **375–433 W** (cap 450 W),
  SM clock **2248–2272 MHz** (pin 2280), GPU temp **36–69 °C**, util 100 % under load.
- llama-benchy 0.4.0 against the engine's **native OpenAI API** (not llama.cpp),
  greedy, mean ± std of 3 runs; tokenizer `ukisai/Swift-1.5-Qwen3.8-27b`.
- `pp8192 @ dN` = throughput of 8,192 **new** tokens processed on top of N cached
  context tokens — the agentic-coding profile (long, growing context; small
  incremental prompts). `e2e ttft` is end-to-end time to first response chunk.

## llama-benchy (single RTX 5090, engine-native API)

| test | t/s (mean ± std of 3) |
|---|---|
| pp2048 | 38,919 ± 13,188 |
| tg256 | 214 ± 4 |
| pp2048 @ 16k cached ctx | 384,586 ± 1,394 |
| tg256 @ 16k cached ctx | 184 ± 5 |
| pp8192 @ 32k cached ctx | 746,013 ± 2,464 |
| tg512 @ 32k cached ctx | 179 ± 21 |
| pp8192 @ 131k cached ctx | 1,418,153 ± 12,502 |
| tg512 @ 131k cached ctx | 153 ± 2 |
| pp8192 @ 200k cached ctx | 1,718,319 ± 17,247 |
| tg512 @ 200k cached ctx | 158 ± 16 |
## kaushikvira Qwen3.8-27B artifact family (single-RTX-5090 serving)

All artifacts are NInfer containers of Qwen3.8-27B derivatives, served by the same
engine on the same box. Per-card A/B tables were measured pairwise in the same
session; cross-card rows are from different sessions — treat small deltas as indicative.

| artifact | base | uncensored | IFBench prompt-strict | GSM8K-200 | decode tok/s | tg512 @ 131k ctx | status |
|---|---|---|---|---|---|---|---|
| [nvfp4full-dflash2 (v2)](https://huggingface.co/kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer) | Qwen3.8-27B | no | — | — | 146.4* | — | superseded by v3 (same weights, v2 container) |
| [nvfp4full-dflash2.v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer-v3) | Qwen3.8-27B | no | 65.0* | 96.5%* | 146.4* | 162.9 | available |
| [swift-abliterated v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift-abliterated-nvfp4full-dflash2-NInfer-v3) | d0xin Swift-1.0 (huihui-abliterated) | **yes** | 66.3 | 95.5% | 148.7 | 173.7 | available |
| [swift15 v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3) | ukisai Swift-1.5 | no | 69.0 | 95.0% | 160.8 | 153.4 | rollback profile |
| [thinkingcap v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-thinkingcap-nvfp4full-dflash2-NInfer-v3) | BottleCapAI ThinkingCap | no | 68.0 | 95.5% | 167.6 | 177.3 | available (PolyForm — non-commercial) |
| [swift15-uncensored-ajgazin v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift15-uncensored-ajgazin-nvfp4full-dflash2-NInfer-v3) | ukisai Swift-1.5 + ajgazin/orcarouter ablation | **yes** | **70.3** | 94.5% | 165.7 | **187.9** | **current production** |

\* values for the nvfp4full-dflash2.v3 row come from its same-session A/B against
swift-abliterated (66.3-vs-65.0 etc. are paired measurements, not independent runs).


---

## Acceptable Use Policy & Legal Notice

**By downloading, possessing, or using this artifact you agree to the terms below.**

### Permitted use
Research, personal/local experimentation, red-teaming, safety research, and
evaluating alignment and quantization techniques — subject to all applicable
laws and regulations.

### Prohibited use
You must **not** use this model, alone or in any pipeline, to:
- carry out, plan, or facilitate any activity that is illegal in your jurisdiction;
- produce content that harms, endangers, defrauds, harasses, or exploits others —
  including but not limited to weapons, malware, exploitation of minors,
  targeted harassment, or disinformation presented as fact;
- provide medical, legal, or financial advice presented as professionally qualified;
- violate the rights of any person or entity, including intellectual property
  and privacy rights.

### Your responsibility
Model outputs are a **capability, not a judgment** — never guaranteed
trustworthy, factual, or legal. **You are solely and fully responsible** for
every prompt you send, every output you generate, and every use you make of
them. The publisher of this artifact (`kaushikvira`) does not monitor, endorse,
or take any part in downstream use.

### No warranty / limitation of liability
The artifact is provided **"AS IS", WITHOUT WARRANTY OF ANY KIND**, express or
implied, including merchantability, fitness for a particular purpose, and
non-infringement. **To the maximum extent permitted by applicable law, the
publisher shall not be liable for any claim, damages, or other liability,
whether in contract, tort or otherwise, arising from, out of, or in connection
with this artifact or its use.** Nothing in this card limits liability where
limitation is not permitted by law.

### Reporting
If you become aware of misuse of this model, report it to the platform where
the misuse occurs and to the relevant authorities.

## HF-format variant (vLLM / SGLang — JSON-schema structured output)

NInfer does not support JSON-schema output; if you need structured output or the
standard vLLM/SGLang stack, use the HF-format NVFP4 weights:

- **[ukisai/Swift-1.5-Qwen3.8-27b-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4)** (official, by the model author).

If you benchmark an HF-format variant on your hardware, please share numbers in
the repo's Community tab — we collect them on the variant's card.

## License

Apache-2.0 (inherited via the artifact chain); Swift-1.5 weights under
[swift-open-license-1.0](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/LICENSE).
**All model credit to ukisai (UkisAI)** for Swift-1.5, **Qwen team** for the base,
**z-lab** for DFlash2, **Neroued** for the engine.
