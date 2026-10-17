---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
  - kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer
base_model_relation: quantized
tags:
  - ninfer
  - ninfer-v3
  - qwen3.8
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

# Qwen3.8-27B nvfp4full + DFlash2 for NInfer — **v3 container**

The same bytes you know from
[kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer](https://huggingface.co/kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer)
(v2), migrated to the **NInfer v3 artifact container** (upstream `f76e19c0`,
model/weight decoupling). Weight tensors are **bit-preserved** — this is a
packaging upgrade, not a re-quantization.

> ## 🔑 No engine patch needed anymore
>
> The v2 artifact required a *forked* NInfer engine: the `qwen3.8-27b/nvfp4full`
> weights profile had to be registered in C++ (4 files: profile enum, resolver,
> binder, workspace cases). **NInfer v3's loader is data-driven** — every weight
> binds from the artifact's own logical bindings and per-tensor formats — so
> this artifact needs **no model-support patch of any kind**. Any NInfer engine
> at upstream v3 (`Neroued/ninfer` master ≥ `f76e19c0`, 2026-09-13) loads and
> serves it. Our previous fork's C++ registration commits were deleted during
> the v3 rebase and the model still passes the full quality gate (see below),
> which is the proof: the engine path in our build *is* upstream's.
>
> The only fork-local piece left is one allow-list line in the *offline upgrade
> tool* (to produce v3 files like this one) — not in the engine, not needed to
> serve.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4full-dflash2.v3.ninfer` |
| Size | 20,551,161,328 bytes (19.14 GiB) |
| **SHA-256** | `74d2fc17742537defa6897583caaabcc9adfc99bdfbea402ec5e8facd0d5fdfc` |
| Container version | **3** (`NINFER\0\x03` entry magic) |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `nvfp4full` |
| Stored objects | 1,325 (1,259 nvfp4full + 66 DFlash2) — same logical set as v2 |
| Chat template | upstream-maintained `qwen3_8.jinja` (installed by the upgrade; replaces cometkim's embedded copy) |
| Upgrade report | [`v3-upgrade.json`](v3-upgrade.json) (included) |

Verify after download:

```bash
printf '%s  %s\n' \
  '74d2fc17742537defa6897583caaabcc9adfc99bdfbea402ec5e8facd0d5fdfc' \
  'qwen3_8_27b_nvfp4full-dflash2.v3.ninfer' | sha256sum --check
```

## Credits

| Role | Model / repo | Author | SHA / commit |
|---|---|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | **Qwen team** (Alibaba Cloud) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0) |
| Quantized source | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | **unsloth** | `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` |
| Original nvfp4full NInfer conversion | [cometkim/Qwen3.8-27B-nvfp4full-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer) | **cometkim** | v1 artifact `2f59cc27d67cb7acba0ba8a0e0881ac89c1db2b267a60119a696fefa12faf4e7` (byte-preserved through our graft/upgrade) |
| DFlash2 speculative drafter | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | **z-lab** | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| Engine | [Neroued/ninfer](https://github.com/Neroued/ninfer) | **Neroued** | v3 container, ≥ `98dada0e` |
| Quantization-runbook inspiration | [Barding-Defense/Qwen3.8-27B-huihui-abliterated-NVFP4-NInfer](https://huggingface.co/Barding-Defense/Qwen3.8-27B-huihui-abliterated-NVFP4-NInfer) | Barding-Defense | community ninferization walkthrough (credited wherever its steps shaped our pipeline) |
## Lineage / provenance (complete chain)

```
Qwen/Qwen3.8-27B  (base model)
  └─ unsloth/Qwen3.8-27B-NVFP4  (quantized source, per cometkim's conversion.json)
      └─ cometkim/Qwen3.8-27B-nvfp4full-NInfer  (v1, v2-container, 1,259 objects)
          └─ graft: z-lab/Qwen3.8-27B-DFlash2  (+66 W8G32/BF16 objects → 1,325)
              = kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer  (v2)
                  └─ tools/upgrade_ninfer_v2_to_v3.py  (bytes-preserved container upgrade)
                      = this repository  (v3)
```

| Stage | Artifact | Bytes | SHA-256 |
|---|---|---|---|
| L0 base model | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | — | (upstream) |
| L0 quantized source | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | — | (upstream) |
| L1 cometkim nvfp4full v1 | `qwen3_8_27b_nvfp4full.ninfer` | 18,324,059,648 | `2f59cc27d67cb7acba0ba8a0e0881ac89c1db2b267a60119a696fefa12faf4e7` **(re-hashed on disk 2026-09-18 — matches)** |
| L2 + DFlash2 graft (v2) | `qwen3_8_27b_nvfp4full-dflash2.ninfer` | 20,550,864,896 | `373dab08733f96555d964e02c6cebe40b23f753c0b2c4592fe130d641f937de1` **(re-hashed on disk 2026-09-18 — matches)** |
| **L3 this file (v3)** | `qwen3_8_27b_nvfp4full-dflash2.v3.ninfer` | 20,551,161,328 (**+289,432 B**) | `74d2fc17742537defa6897583caaabcc9adfc99bdfbea402ec5e8facd0d5fdfc` |

- L1→L2: base tensors copied **byte-for-byte**; only the 66 DFlash2 objects
  (21 × `W8G32_F16S` matrices, 45 × BF16 norms/conv bases/codebooks) produced
  from `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
  via `tools/artifact/graft_dflash2_w8.py`
  (recipe `qwen3_8_27b_nvfp4full-dflash2-w8`, report
  [`graft.json`](graft.json) included; MTP module retained, validate-only).
- L2→L3: upstream's `tools/upgrade_ninfer_v2_to_v3.py`, run 2026-09-18 on the
  exact L2 file above. Per the tool's contract: *"Stored weight values and
  formats are preserved"* — the +289 KB delta is the v3 directory (JSON logical
  bindings), format table entries, framing padding, and the maintained
  `qwen3_8.jinja` chat template. Weights, formats and object set are unchanged.
- The tool allow-lists the seven official NInfer v2 inputs; our `nvfp4full`
  identity required adding one `KNOWN_COUNTS` entry
  (`("qwen3.8-27b","nvfp4full"): (1259, 1325)`) in
  [kaushikvira/ninfer@`3369b385`](https://github.com/kaushikvira/ninfer).
  Everything else validated unchanged — every format/layout we carry already
  exists upstream.

## Reproduce the L2→L3 step yourself

```bash
git clone https://github.com/kaushikvira/ninfer && cd ninfer
# tool from our main (allow-list extended; engine not required for the tool)
python3 tools/upgrade_ninfer_v2_to_v3.py \
  /path/to/qwen3_8_27b_nvfp4full-dflash2.ninfer \
  /path/to/qwen3_8_27b_nvfp4full-dflash2.v3.ninfer
# run from the repo root (it installs chat_templates/qwen3_8.jinja from there)
```

Input must match SHA-256 `373dab08…937de1`; output must match `74d2fc17742537defa6897583caaabcc9adfc99bdfbea402ec5e8facd0d5fdfc`.

## Requirements

- **NInfer v3** — upstream [`Neroued/ninfer`](https://github.com/Neroued/ninfer)
  master at/after the v3 loader switch (`4cde7ad0`, 2026-09-14; any current
  master qualifies). **No forks, no registrations, no build flags.**
- ⚠️ v3 is one-way in this respect: a v3 engine **rejects v2 artifacts**
  (explicit error), and pre-v3 engines reject v3. This file only runs on v3.
- 64-bit Linux; NVIDIA RTX 5090 (`sm_120a`); CUDA Toolkit 13.1+.
  (Upstream v3 targets the Qwen3.5/3.6/3.8 family on Blackwell; other
  devices are upstream's concern, our validation is on a 5090.)

## Download and run

```bash
hf download kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer-v3 \
  qwen3_8_27b_nvfp4full-dflash2.v3.ninfer --local-dir models

# CLI smoke
apps/ninfer models/qwen3_8_27b_nvfp4full-dflash2.v3.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 --max-new 8192 --kv-dtype k8v4 \
  --spec dflash2 --draft-tokens 7

# server
apps/ninfer-serve models/qwen3_8_27b_nvfp4full-dflash2.v3.ninfer \
  --host 127.0.0.1 --port 8080 \
  --max-context 262144 --kv-capacity auto --max-concurrency 4 \
  --kv-dtype k8v4 --spec dflash2 --draft-tokens 7 --preserve-thinking
```

Sanitized full serving profile (same flags, no secrets):
[`examples/ninfer-nvfp4full-grafted-dflash2.cfg.example`](https://github.com/kaushikvira/ninfer/blob/main/examples/ninfer-nvfp4full-grafted-dflash2.cfg.example).

## Validation (gate run on the cutover box, 2026-09-18)

Engine: upstream `f76e19c0` + serving conveniences only (no engine-model code).
One RTX 5090, 32 GB, 450 W SM-capped, k8v4, 262,144 context, conc 4, DFlash2
K=7, vision on. Probes: the repo's own `gate.sh` (needle 12 / probes 4 / tool
10 / streaming) + `perf.py`:

| Metric | v3 (this file) | v2 (same weights) |
|---|---|---|
| Weights on device | 18.7 GiB | 18.7 GiB |
| KV pool | 278,912 tok (k8v4) | ~284k tok |
| Needle long-context retrieval | **12/12 exact, 0 corrupted** | 12/12 |
| Tool calls / streaming | 10/10, 10/10 | 10/10 |
| Decode C=1 warm | **162.5 tok/s** | 159.8 tok/s |
| Prefill 12.6k / 50k / 100k / 200k | 10.9k / 7.6k / 5.2k / 3.2k tok/s | 10.2k / 7.5k / 5.4k / 3.5k |
| DFlash2 acceptance | 35–80% (live mix) | — |

v3's ops reorg (partial-M-tile TMA route `5f5fccab`, activation-scale tiling
`1d8587bc`) is on top — the +1.7% decode and small prefill shifts are our
single-rig numbers, not vendor benchmarks.

## Notes / limits

- Chat template changed at L3: the upgrade installs upstream's maintained
  `qwen3_8.jinja` in place of cometkim's embedded copy. All quality probes
  pass; if a client is picky about thinking-tag formatting, diff it against
  the v2 artifact's behavior.
- Same limits as v2: single device, startup-bounded concurrency 1–8, NInfer
  does not execute generated tool calls.
- License: Apache-2.0; base, quantized source, and DFlash2 drafter likewise Apache-2.0.

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
| pp2048 | 40,772 ± 4,739 |
| tg256 | 207 ± 9 |
| pp2048 @ 16k cached ctx | 354,872 ± 30,667 |
| tg256 @ 16k cached ctx | 174 ± 5 |
| pp8192 @ 32k cached ctx | 744,829 ± 3,981 |
| tg512 @ 32k cached ctx | 163 ± 23 |
| pp8192 @ 131k cached ctx | 1,423,081 ± 17,745 |
| tg512 @ 131k cached ctx | 163 ± 14 |
| pp8192 @ 200k cached ctx | 1,760,942 ± 65,915 |
| tg512 @ 200k cached ctx | 165 ± 9 |
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

- **[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)** (upstream) — the HF-format NVFP4 this artifact's lineage is built on; we deliberately did not re-publish a duplicate.

If you benchmark an HF-format variant on your hardware, please share numbers in
the repo's Community tab — we collect them on the variant's card.

