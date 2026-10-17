---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: swift-open-license-1.0
license_link: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/LICENSE
base_model:
  - ukisai/Swift-1.5-Qwen3.8-27b
  - Qwen/Qwen3.8-27B
  - z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
tags:
  - ninfer
  - ninfer-v3
  - qwen3.8
  - swift
  - swift-1.5
  - abliterated
  - uncensored
  - nvfp4
  - w4a4
  - blackwell
  - multimodal
  - rtx-5090
  - dflash2
  - speculative-decoding
---

# Swift-1.5 Qwen3.8-27B Uncensored (ajgazin-direction) nvfp4full + DFlash2 for NInfer — v3 container

An **all-NVFP4 abliterated** artifact of [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b)
for the [NInfer](https://github.com/Neroued/ninfer) engine (v3 container), with
z-lab's DFlash2 speculative drafter and indexed proposal head embedded.
Uncensored sibling of [Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-swift15-nvfp4full-dflash2-NInfer-v3).

## Credits & provenance (full chain)

All credit for the underlying model belongs to the original authors. This
artifact is a format conversion + quantization + refusal-direction ablation —
**no fine-tuning of our own**.

| Role | Model | Author | SHA / commit |
|---|---|---|---|
| Source weights (BF16) | [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) | **ukisai (UkisAI)** — RL+OPD post-training of Qwen3.8-27B | 18-shard BF16 safetensors, published 2026-09-24; shards sha256-verified against HF LFS on download |
| Upstream base | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | **Qwen team** (Alibaba Cloud), Apache-2.0 | — |
| Abliteration | [ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored-MTP](https://huggingface.co/ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored-MTP) — single refusal direction recovered from [orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) | ajgazin / orcarouter | recovered direction `r` sha256 `f99121055e9090a5a85d39002fe3e83f2d77119b1943f494497cf262c5a6f4ad` (`abliteration.json` in this repo); 18 shards LFS-verified on download |
| DFlash2 drafter | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | **z-lab** | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Engine | [Neroued/ninfer](https://github.com/Neroued/ninfer) | **Neroued** | v3 container, ≥ `98dada0e` |
| Quantization-runbook inspiration | [Barding-Defense/Qwen3.8-27B-huihui-abliterated-NVFP4-NInfer](https://huggingface.co/Barding-Defense/Qwen3.8-27B-huihui-abliterated-NVFP4-NInfer) | Barding-Defense | community ninferization walkthrough |

## What we did

1. **Abliteration** — single-direction Arditi-style ablation: orcarouter's refusal direction r, recovered from their Qwen3.8-27B-Uncensored weights, projected out of 131 residual-writing tensors of Swift-1.5 (o_proj, GDN out_proj, mlp.down_proj, embed_tokens) in float32, exactly as ajgazin's release.
2. **All-NVFP4 quantization** with `llm-compressor` (512 Ultrachat calibration
   samples, seq 2048, sequential CPU pipeline): NVFP4 W4A4 group-16 on every
   text projection, W8G32 token embedding + output head, official q6/q8 vision
   allocation, MTP + DFlash2 BF16.
3. **Global-divisor normalization** across each fused packing group
   (worst re-encode rel err 79.25%), then v3 conversion with the fork
   converter (`--components text,vision,mtp,dflash2 --proposal`).

## Measured results (same-session A/B on one RTX 5090, temp 0)

| metric | Swift-1.5 (prod, censored) | **this artifact** |
|---|---|---|
| Refusal rate (harmful_behaviors test[:100], greedy) | 59/100 | **8/100** (harmless sanity 45/50) |
| Harmless sanity pass rate (n=50) | 43/50 | **45/50** |
| IFBench prompt-strict (n=300, fresh) | 69.0 | **70.3** (loose 74.3, instr 70.6/74.1) |
| GSM8K-200 (fresh) | 95.0% | 94.5% (−0.5 pp, n=200 noise) |
| Gate decode tok/s | 145.5 | **163.2** (GATE PASS) |
| Perf decode (mean of 3) | 167.4 | 165.7 (parity) |
| Prefill @ 200k | 3,259 tok/s | 3,259 tok/s (parity) |
| llama-benchy tg @ depth 0/16k | 202.6 / 189.2 | 198.9 / 211.9 (full grid below) |
| Needle full ladder (haystack 1M chars / 250,031 tok, 3 depths) | EXACT | **24/24 EXACT, 0 corrupted** |

## Artifact

| Field | Value |
|---|---|
| Filename | qwen3_8_27b_swift15abl_nvfp4full-dflash2.ninfer |
| Size | 19,782,447,364 bytes (18.42 GiB) |
| **SHA-256** | `65f9d2ab1161e95d65ecdb20858b42ae937bb40c92d3dabf969fd47bfd03c0cb` |
| Container version | 3 |
| Formats | nvfp4 ×256 (fused text parents), q8_g32 ×30, q4/q5/q6 (vision), bf16 remainder |

## Serving (RTX 5090 32 GB, single GPU)

```
ninfer-serve qwen3_8_27b_swift15abl_nvfp4full-dflash2.ninfer \
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

---

## Acceptable Use Policy & Legal Notice

**By downloading, possessing, or using this artifact you agree to the terms below.**

### Permitted use
Research, personal/local experimentation, red-teaming, safety research, and
evaluating alignment-removal techniques — subject to all applicable laws and
regulations.

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
This model is a **capability, not a judgment**. Abliteration removes refusal
behavior; it does not make outputs trustworthy, factual, or legal. **You are
solely and fully responsible** for every prompt you send, every output you
generate, and every use you make of them. The publisher of this artifact
(`kaushikvira`) does not monitor, endorse, or take any part in downstream use.

### No warranty / limitation of liability
The artifact is provided **"AS IS", WITHOUT WARRANTY OF ANY KIND**, express or
implied, including merchantability, fitness for a particular purpose, and
non-infringement. **To the maximum extent permitted by applicable law, the
publisher shall not be liable for any claim, damages, or other liability,
whether in contract, tort or otherwise, arising from, out of, or in connection
with this artifact or its use.** Nothing in this card limits liability where
limitation is not permitted by law.

### Measurement conditions

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
| pp2048 | 48,625 ± 35 |
| tg256 | 199 ± 18 |
| pp2048 @ 16k cached ctx | 387,707 ± 2,966 |
| tg256 @ 16k cached ctx | 212 ± 17 |
| pp8192 @ 32k cached ctx | 746,777 ± 2,595 |
| tg512 @ 32k cached ctx | 184 ± 18 |
| pp8192 @ 131k cached ctx | 1,428,495 ± 13,022 |
| tg512 @ 131k cached ctx | 188 ± 26 |
| pp8192 @ 200k cached ctx | 1,729,533 ± 11,547 |
| tg512 @ 200k cached ctx | 167 ± 9 |
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


## HF-format variant (vLLM / SGLang — JSON-schema structured output)

NInfer does not support JSON-schema output; if you need structured output or the
standard vLLM/SGLang stack, use the HF-format NVFP4 weights:

- **[ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored-NVFP4)** (official, by the abliteration author).

If you benchmark an HF-format variant on your hardware, please share numbers in
the repo's Community tab — we collect them on the variant's card.

## License terms carried by this artifact
- **Swift Contribution** (ukisai): [Swift Open License v1.0](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/LICENSE) — a full copy is in `LICENSE`. Note its **Commercial Use Limitation (§5)**: commercial use by entities over US$1M gross annual revenue requires a separate enterprise license from UkisAI. This limitation applies to the Swift Contribution contained in this derivative.
- **Base Model** (Qwen3.8-27B): Apache License 2.0 — copy in `LICENSE-APACHE-2.0`.
- **Modified files notice (Swift Open License §4b):** the weights in this repository are **modified** relative to the upstream Work — refusal-direction ablation (131 tensors, listed in `abliteration.json`) plus NVFP4/q8 requantization and NInfer v3 containerization.
- **NOTICE:** attribution notices preserved in `NOTICE`.

### Reporting
If you become aware of misuse of this model, report it to the platform where
the misuse occurs and to the relevant authorities. Do not contact the publisher
for takedowns of third-party actions — the publisher has no control over
downstream deployments.

