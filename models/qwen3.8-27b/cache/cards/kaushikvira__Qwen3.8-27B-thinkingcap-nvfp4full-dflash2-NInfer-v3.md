---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: polyform-small-business-1.0.0
license_link: LICENSE
base_model:
  - bottlecapai/ThinkingCap-Qwen3.8-27B
  - Qwen/Qwen3.8-27B
  - z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
  - ninfer
  - ninfer-v3
  - qwen3.8
  - qwen3_5
  - thinkingcap
  - bottlecapai
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

# ThinkingCap Qwen3.8-27B nvfp4full + DFlash2 for NInfer — **v3 container**

An **all-NVFP4** artifact of
[bottlecapai/ThinkingCap-Qwen3.8-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B)
for the [NInfer](https://github.com/Neroued/ninfer) engine, with z-lab's DFlash2
speculative drafter and the indexed proposal head embedded. Native **v3** container.

ThinkingCap (BottleCap AI — Osusky, Lindauer, Jirkovsky, Mihal, Platek, Herel,
Ihnatchenko, Bartek, Jirak, Kubista, Krus & **Tomas Mikolov**) is a
thinking-efficiency derivative of Qwen3.8-27B: it cuts reasoning tokens by
**37% on average** (11–66% depending on benchmark) at 85.8% macro accuracy vs
the base model's 86.6%, shining on long-context retrieval (−39% thinking,
+2.3pp). This artifact brings that checkpoint to Blackwell consumer GPUs via
the NInfer stack.

It is the sibling of our production artifact
[Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3)
— same engine, same quantization pipeline, same DFlash2 drafter. Our same-session
A/B (below) kept Swift-1.5 as production and uses this artifact as the
token-efficient alternate profile.

## Credits & provenance (full chain)

**All credit for the model itself goes to the authors below** — this artifact is
a format conversion + quantization of their work, with no fine-tuning of our own:

| Role | Model | Author | SHA / commit |
|---|---|---|---|
| **Source weights (BF16)** | [bottlecapai/ThinkingCap-Qwen3.8-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B) | **BottleCap AI** | repo commit `4bd4e11054e4ceb0dcecfff2f0d2ffa906f37232`; all 18 BF16 shards + tokenizer sha256-verified against HF LFS metadata before quantization (full manifest in `SOURCE_HASHES.txt`) |
| **Upstream base** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | **Qwen team** | ThinkingCap is a thinking-efficiency finetune of Qwen3.8-27B (Apache-2.0 upstream materials, see `NOTICE`) |
| **DFlash2 drafter** | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | **z-lab** | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (same drafter as our Swift artifacts) |
| **Engine** | [Neroued/ninfer](https://github.com/Neroued/ninfer) | **Neroued** | v3 container, ≥ `98dada0e`; built from fork `f76e19c0` + carried commits (`PATCHES.md`) |
| Sibling artifact (our prod) | [kaushikvira/Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3) | kaushikvira | ukisai Swift-1.5, sha256 `16f313c0…` |
| Quantization-runbook inspiration | [Barding-Defense/Qwen3.8-27B-huihui-abliterated-NVFP4-NInfer](https://huggingface.co/Barding-Defense/Qwen3.8-27B-huihui-abliterated-NVFP4-NInfer) | Barding-Defense | community ninferization walkthrough |

## What we did (the technique)

Identical pipeline to our Swift-1.5 / Swift-1.0 builds (see the sibling cards):

1. **All-NVFP4 quantization** with `llm-compressor` (512 Ultrachat calibration
   samples, seq 2048, sequential pipeline): NVFP4 W4A4 group-16 on every text
   projection (MLP gate/up/down all 64 layers, attention Q/K/V/O, GDN
   in_proj_qkv/z/out_proj), W8G32 token embedding + output head, official q6/q8
   vision allocation, MTP + DFlash2 in BF16. Architecture verified identical to
   Swift-1.5 pre-build: 64L/5120/24×256/4kv, vocab 248320, 262,144 ctx,
   16 full-attention + 48 linear-attention (GDN) layers.
2. **Global-divisor normalization** (our tool): llm-compressor emits per-module
   `weight_global_scale`; the engine's native A4 route requires one contiguous
   fused parent per attention group (GDN `qkvz` 16,384 rows, attn `qkgv` 14,336
   rows). We unify each packing group's divisor to `D = min(dᵢ)` and rescale the
   E4M3 block scales by `D/dᵢ` (RNE, shrink-only). 128 groups / 272 member
   modules; true worst scale re-encode error **6.19% ≤ the 6.25% E4M3 RNE
   bound** (Swift-1.5 was 0.0000% — its within-group divisors were nearly
   equal; ThinkingCap's differ up to 2.1×, so RNE rounding shows. Decode
   behaviour is unaffected — verified by the needle + gate results below).
3. **v3 conversion** with `tools/convert` (`--components text,vision,mtp,dflash2 --proposal`).

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_thinkingcap_nvfp4full-dflash2.ninfer` |
| Size | 19,782,447,364 bytes (18.42 GiB) |
| **SHA-256** | `fd977d3b1721e45231eb4ede9aed3c3a0ef781a7066d0c98a158685cf65994fb` |
| Container version | **3** (`NINFER\0\x03`) |
| NInfer model ID | `qwen3.8-27b` (serve as `Qwen3.8-27B`) |
| Stored objects | 1,590 (Text + Vision + MTP + DFlash2 + indexed proposal head) |
| Formats | nvfp4 ×256 (fused text parents), q8_g32 ×30, q4/q5/q6 (vision), bf16 remainder |
| Hash manifests | `SHA256SUMS` (artifact + conversion contract), `SOURCE_HASHES.txt` (verified source weights); full conversion contract in `.conversion.json` |

## Serving (RTX 5090 32 GB, single GPU)

```
ninfer-serve qwen3_8_27b_thinkingcap_nvfp4full-dflash2.ninfer \
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
(auto), 48 GiB pinned host-KV arena, full 262,144-token context at concurrency
4.** The chat template is ThinkingCap's own (carried from the source repo,
including its `reasoning_effort` knob; `xhigh` is their recommended default).

## Benchmarks — same-session A/B vs our Swift-1.5 production artifact

Both sides served by the same engine build, fresh generations, temp 0, no cache
reuse, single RTX 5090 (450 W cap, SM clock pinned 2280 MHz).

| Benchmark | Swift-1.5 (our prod) | **ThinkingCap (this artifact)** | Δ |
|---|---|---|---|
| Gate (all probes incl. tool calls) | PASS | **PASS** (decode 159.8 tok/s) | = |
| Perf decode (mean of 3) | 160.8 tok/s | **167.6 tok/s** | +4.2% |
| Prefill @ 200k ctx | 3,269 tok/s | 3,261 tok/s | = |
| GSM8K-200 accuracy | 95.0% (190/200) | **95.5% (191/200)** | +0.5 pp (n=200 noise) |
| IFBench prompt-strict (n=300) | **69.0** | 68.0 | −1.0 |
| IFBench prompt-loose | **72.7** | 71.0 | −1.7 |
| IFBench instr-strict | **70.4** | 67.7 | −2.7 |
| IFBench instr-loose | **73.6** | 70.3 | −3.3 |
| IFBench mean completion tokens (thinking on) | 4,805 | **3,988** | **−17.0%** |
| Long-context recall (needle → 1M chars) | 250,031 tok EXACT ×3 | **250,031 tok EXACT ×3 (24/24 overall)** | = |
| Weights in VRAM / artifact / KV pool | 18.0 GiB / 18.42 GiB / 308,736 tok | same | = |

**Reading:** ThinkingCap delivers its advertised efficiency on our stack —
−17% thinking tokens vs Swift-1.5 with math, speed and long-context retrieval
intact (and the fastest decode of any artifact we've built) — but gives back
1.0–3.3pp IFBench. Against the unmodified base Qwen3.8-27B, BottleCap's own
H200/vLLM evaluation reports −37% thinking at −0.8pp macro accuracy, so the
trade is exactly as advertised by the upstream card. Pick per workload:
instruction-following-critical → Swift-1.5; token-cost-dominated long thinking
episodes → this artifact. (Both are lossless-swap profiles on the same engine.)

## Notes

- Source license is **PolyForm Small Business 1.0.0 + BottleCap personal-use
  grant** (see License below) — distributed in full compliance: license terms,
  personal-use grant and notices ship with the artifact. It is **not**
  Apache-2.0 like our Swift builds.
- ThinkingCap carries the base Qwen3.8 refusal behavior (not abliterated).
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
| pp2048 | 49,063 ± 321 |
| tg256 | 212 ± 7 |
| pp2048 @ 16k cached ctx | 389,124 ± 1,353 |
| tg256 @ 16k cached ctx | 190 ± 13 |
| pp8192 @ 32k cached ctx | 746,204 ± 5,838 |
| tg512 @ 32k cached ctx | 186 ± 45 |
| pp8192 @ 131k cached ctx | 1,419,363 ± 8,115 |
| tg512 @ 131k cached ctx | 177 ± 26 |
| pp8192 @ 200k cached ctx | 1,804,137 ± 18,660 |
| tg512 @ 200k cached ctx | 153 ± 7 |
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

- **[Qwen3.8-27B-thinkingcap-NVFP4-HF](https://huggingface.co/kaushikvira/Qwen3.8-27B-thinkingcap-NVFP4-HF)** (ours) — the exact pre-conversion source of this container (17.1 GiB compressed-tensors NVFP4 + MTP head); README has vLLM serve + guided-JSON snippets.

If you benchmark an HF-format variant on your hardware, please share numbers in
the repo's Community tab — we collect them on the variant's card.

## License

This artifact inherits **PolyForm Small Business License 1.0.0** from BottleCap
AI's ThinkingCap, plus BottleCap AI's additional personal-use permission. Full
texts ship in this repo and apply to every copy distributed from here:

- [`LICENSE`](./blob/main/LICENSE) — BottleCap AI license statement + personal-use grant
- [`LICENSE-PolyForm-Small-Business-1.0.0.txt`](./blob/main/LICENSE-PolyForm-Small-Business-1.0.0.txt) — full license terms
- [`NOTICE`](./blob/main/NOTICE) — attribution + upstream Apache-2.0 Qwen materials
- [`LICENSE-Apache-2.0-Qwen.txt`](./blob/main/LICENSE-Apache-2.0-Qwen.txt) — upstream Qwen license text

Per the PolyForm terms, **we make no sublicense** — recipients of this artifact
are licensed directly by the original licensor (BottleCap AI) under the same
included terms. Anyone who receives a copy gets these terms and the required
notice:

> Required Notice: Copyright 2026 BottleCap AI (https://bottlecapai.com)

The PolyForm SB copyright + distribution + new-works grants cover individuals
and organizations meeting the Small Business condition (< 100 employees and
contractors, < $1M revenue, CPI-adjusted); BottleCap's personal-use grant
additionally covers individual non-commercial use free of charge. Commercial
use outside these conditions requires a license from
[BottleCap AI](https://www.bottlecapai.com/) (enterprise@bottlecapai.com).
Upstream Qwen materials remain Apache-2.0 (see `NOTICE`).

**All model credit to BottleCap AI** for ThinkingCap, **Qwen team** for the
base, **z-lab** for DFlash2, **Neroued** for the engine. If you use this
artifact, please also cite the upstream model:

```bibtex
@misc{ThinkingCap-Qwen3.8-27B,
  title     = {bottlecapai/ThinkingCap-Qwen3.8-27B},
  author    = {Osusky, Adam and Lindauer, Jan and Jirkovsky, Adam and Mihal, Filip and Platek, Ondrej and Herel, David and Ihnatchenko, Luka and Bartek, Vojtech and Jirak, Jiri and Kubista, Daniel and Krus, Frantisek and Mikolov, Tomas},
  year      = {2026},
}
```
