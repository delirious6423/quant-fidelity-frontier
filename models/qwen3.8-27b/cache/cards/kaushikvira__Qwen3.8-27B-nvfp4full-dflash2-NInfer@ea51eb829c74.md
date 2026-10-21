---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - ninfer
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

> **📦 v3 container available.** For NInfer engines at upstream v3 (master >= `f76e19c0`, 2026-09-13): the bytes-preserved v3 build of this artifact is at **[kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer-v3](https://huggingface.co/kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer-v3)** (`qwen3_8_27b_nvfp4full-dflash2.v3.ninfer`, sha256 `74d2fc17…d5fdfc`) and needs **no engine patch** — the C++ `nvfp4full` profile registration is only required by this v2 container. Full lineage chain: `v3-upgrade.json` in the v3 repo.

# Qwen3.8-27B nvfp4full + DFlash2 for NInfer

This repository contains the **nvfp4full** weight profile of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format, **with the
z-lab DFlash2 speculative-decoding drafter module added in the upstream W8G32/BF16
format**.

It is the `qwen3.8-27b` / `nvfp4full` identity (same as
[cometkim/Qwen3.8-27B-nvfp4full-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer))
with the registered DFlash2 module appended — the base tensors are byte-for-byte
identical to cometkim v1, and the MTP module is retained (unused and
validate-only under `--spec dflash2`).

> ⚠️ **Format note.** Unlike
> [gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer)
> (NVFP4 weight-only drafter), the DFlash2 module here uses the **upstream
> W8G32_F16S / BF16** encoding, which is the format our
> [fork](https://github.com/kaushikvira/ninfer) (upstream master + 1 commit)
> binds. The two artifacts are **not interchangeable**.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4full-dflash2.ninfer` |
| Size | 20,550,864,896 bytes (19.14 GiB) |
| SHA-256 | `373dab08733f96555d964e02c6cebe40b23f753c0b2c4592fe130d641f937de1` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `nvfp4full` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,325 (1,259 inherited from cometkim v1 + 66 DFlash2) |
| DFlash2 module objects | 66 (21 × W8G32_F16S matrices, 45 × BF16 norms/conv bases/codebooks) |
| MTP module | retained (validate-only with `--spec dflash2`) |

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '373dab08733f96555d964e02c6cebe40b23f753c0b2c4592fe130d641f937de1' \
  'qwen3_8_27b_nvfp4full-dflash2.ninfer' | sha256sum --check
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
## Lineage / provenance

| Field | Value |
|---|---|
| Base repository | `cometkim/Qwen3.8-27B-nvfp4full-NInfer` (v1, module-less) |
| Base recipe | `qwen3_8_27b_nvfp4full-v1` (cometkim fork, `feat/qwen3.8-nvfp4full`) |
| Base artifact SHA-256 | `2f59cc27d67cb7acba0ba8a0e0881ac89c1db2b267a60119a696fefa12faf4e7` |
| Base model | `Qwen/Qwen3.8-27B` (revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`) |
| Quantized source | `unsloth/Qwen3.8-27B-NVFP4` (revision `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`, per cometkim's conversion.json) |
| DFlash2 drafter | `z-lab/Qwen3.8-27B-DFlash2` (revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`) |
| DFlash2 encoding | `W8G32_F16S` (matrices) + `BF16` (norms / conv bases / selector codebooks) |
| Graft recipe | `qwen3_8_27b_nvfp4full-dflash2-w8` via `tools/artifact/graft_dflash2_w8.py` |
| Graft report | `qwen3_8_27b_nvfp4full-dflash2.ninfer.graft.json` (included) |

The base tensors are copied from cometkim v1 **byte-for-byte** (no
re-encoding); only the 66 DFlash2 module objects are produced by the graft
tool from the z-lab checkpoint. Verified: source identity
`qwen3.8-27b/nvfp4full`, 1,259 source objects, no pre-existing `dflash2/*`
objects, output = 1,325.

## Requirements

- [NInfer](https://github.com/kaushikvira/ninfer) `main` (= upstream
  `487f8977` + commit `2eb59dbc`), or upstream master at/after DFlash2
  binding (`385b30ce`) **plus** the `qwen3.8-27b/nvfp4full` weights
  registration — upstream `Neroued/ninfer` alone does **not** register the
  `nvfp4full` profile yet;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

## Download and run a CLI example

```bash
hf download kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer \
  qwen3_8_27b_nvfp4full-dflash2.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_8_27b_nvfp4full-dflash2.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype k8v4 \
  --spec dflash2 --draft-tokens 7
```

## Start a local server

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4full-dflash2.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 262144 \
  --kv-capacity auto \
  --max-concurrency 4 \
  --kv-dtype k8v4 \
  --spec dflash2 --draft-tokens 7 \
  --preserve-thinking
```

`--spec dflash2 --draft-tokens 7` selects the DFlash2 drafter (K=7,
block width 8; K=1..15 supported for the z-lab module, 7 recommended).
A full sanitized serving profile (same flags, no secrets) is at
[`examples/ninfer-nvfp4full-grafted-dflash2.cfg.example`](https://github.com/kaushikvira/ninfer/blob/main/examples/ninfer-nvfp4full-grafted-dflash2.cfg.example)
in the engine fork. Leave `--lm-head-draft` **off** (the full-vocabulary proposal head is the
default and DFlash2 candidates span the whole vocabulary). `--spec mtp`
still works — the MTP module is retained.

## Supported use

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages (`--vision`);
- DFlash2 speculative decoding, draft windows 1–15 (recommended 7);
- MTP speculative decoding (`--spec mtp`, windows 1–5);
- BF16, INT8, FP8, NVFP4, and K8V4 KV cache (`--kv-dtype`);
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving (1–8 active requests);
- the NInfer CLI and OpenAI/Anthropic-compatible HTTP serving.

## Performance (measured on our rig, 2026-09-06)

One NVIDIA GeForce RTX 5090 (32 GB, 450 W cap), CUDA 13.1, DFlash2 K=7,
`--kv-dtype k8v4`, `--max-context 262144`, `--max-concurrency 4`, vision on.
Probes: `bench/tests/perf.py`, `bench/tests/conc_perf.py`,
`bench/tests/needle.py` (same probes used for the engine A/B).

| Metric | Value |
|---|---|
| Weights on device | 18.7 GiB |
| Device KV pool | 283,840 tokens (k8v4) |
| Free VRAM after startup | ~2.75 GiB |
| Decode (C=1, warm) | **175.2 tok/s** |
| Prefill 12,582 tok | 1.27 s TTFT (9,901 tok/s) |
| Prefill 50,082 tok | 6.83 s TTFT (7,331 tok/s) |
| Prefill 100,082 tok | 18.66 s TTFT (5,363 tok/s) |
| Prefill 200,082 tok | 58.20 s TTFT (3,438 tok/s) |
| Aggregate decode C=4 | **527.1 tok/s** (140.8–152.1 per stream) |
| Needle (long-context retrieval, 12 cases) | **12/12 exact, 0 corrupted** |
| DFlash2 acceptance (short smoke) | ~24.5% |

Same rig, same probes — the reference profile (cometkim nvfp4full + MTP4 +
k8v4, our previous base) decodes at 157.9 tok/s with needle 12/12; this
artifact is +11% decode at equal quality on the same KV codec.

These are our own numbers (single sample per probe, deterministic where
possible) — not vendor benchmarks; reproduction commands in the repository.

## Limits

- Accepted only by a NInfer build that registers `qwen3.8-27b/nvfp4full`
  (our fork, or upstream after the registration lands);
- one RTX 5090 / one CUDA device, startup-fixed capacity 1–8 requests;
- no multi-GPU, offload, preemptive batching, or distributed serving;
- context allocation is subject to GPU memory and the KV-cache type —
  on a 32 GB card with k8v4 the auto KV pool lands at ~284k tokens, still
  ≥ 1 full 262,144-token context; host-RAM continuation reuse is a
  server flag, not this artifact;
- NInfer does not execute generated tool calls.


> **Superseded:** the v2 container is retained for history only; the v3 engine rejects v2 artifacts by design. Use [the v3 artifact](https://huggingface.co/kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer-v3).

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


## llama-benchy (single RTX 5090, engine-native API)

Measured on the **v3 twin of this exact artifact** (`Qwen3.8-27B-nvfp4full-dflash2-NInfer-v3`
— byte-identical weights, container-format upgrade only; the v2 container itself
is not servable by the current v3 engine).

| test | t/s (mean ± std of 3) |
|---|---|
| pp2048 | 40,772 ± 4,739 |
| tg256 | 206.9 ± 9.1 |
| pp2048 @ 16k cached ctx | 354,872 ± 30,667 |
| tg256 @ 16k cached ctx | 173.5 ± 4.7 |
| pp8192 @ 32k cached ctx | 744,829 ± 3,981 |
| tg512 @ 32k cached ctx | 162.6 ± 23.3 |
| pp8192 @ 131k cached ctx | 1,423,081 ± 17,745 |
| tg512 @ 131k cached ctx | 162.9 ± 13.7 |
| pp8192 @ 200k cached ctx | 1,760,942 ± 65,915 |
| tg512 @ 200k cached ctx | 164.9 ± 8.9 |

See the [v3 card](https://huggingface.co/kaushikvira/Qwen3.8-27B-nvfp4full-dflash2-NInfer-v3)
for the full measurement-conditions block (450 W cap, SM clock pin, telemetry).

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

## License

Apache-2.0. Base repositories, quantized source, and the DFlash2 drafter
are also Apache-2.0.
