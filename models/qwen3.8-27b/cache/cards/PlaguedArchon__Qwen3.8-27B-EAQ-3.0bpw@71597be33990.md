---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - eaq
  - exl3
  - trellis
  - intel-xpu
language:
  - en
---

# Qwen3.8-27B-EAQ-3.0bpw

A 3.0 bit-per-weight quantization of Qwen3.8-27B in the **EAQ container** format
(EXL3 trellis), built with TensorFold's XPU quantizer on an Intel Arc Pro B70.
The pack is text-only; the vision tower is not included.

## Format

- `config.json` — the base config plus `quantization_config` (`quant_method: eaq`,
  `manifest: eaq_manifest.json`, `manifest_sha256: 35ece13d…`).
- `eaq_manifest.json` — per-part bits/halves, shard map, tensor names, hashes.
- `model-0000N-of-00008.safetensors` — 8 shards, ~11.4 GB total.
- `EAQ_COMPLETE.json` — the commit marker, written last; carries the manifest sha256.
- `quant_report.json` — the per-part encoder report (401 parts, 212,226 B).

Weights:

| Part | Bits | Halves | Tensors |
|---|---|---|---|
| Body (64 layers) | 3 | 6 | 400 trellis |
| `lm_head` | 4 | 8 | 1 trellis |
| Dense (norms, embedding, gates, …) | source dtype | — | 450 |

Table census: 851 entries (401 trellis + 450 dense), 496 projections, head and
embedding present.

## Method

- Encoder: EXL3 Viterbi, contract `exl3-enc/v1: q12 u32-rebased inf=0xFFFFFFFF
  clamp=4.0`, run on the Intel XPU (`xpu-kernel`).
- Calibration: 204,800 token ids (100 rows x 2048; corpus sha `c4c8b035…`).
- LDLQ: lambda 0.025, seed 224.
- Source: Qwen3.8-27B bf16; all 64 layers plus the head were re-encoded.
- Built on an Intel Arc Pro B70 (32,656 MiB physical, ~30.3 GiB usable); no NVIDIA hardware was used.

## Certification

- The reader (`openEaq` + `loadExl3` admission) opens the pack: Table complete,
  496 projections, head and embedding present.
- Tamper refusal: a byte-flipped shard refuses with `ShardHashMismatch` naming the
  shard; a symlinked shard refuses with `ShardOutside`; the manifest hash is bound
  by the commit marker.
## Quality

Measured as forward KL(reference || pack) in nats, fp64 log-sum-exp, on 64 rows x 2048
positions: 16 log-domain rows (the original eval), 16 general-text rows (the docs under
docs/eaq-review), 16 code rows (this tree's Zig and OpenCL sources) and 16 random rows.
"top-1" is argmax agreement with the reference. The reference is the device's own bf16
forward of the dense model through the same Intel XPU serving path (fp16 between kernels,
fp32 inside); it is not an fp64 forward. Its own canary: 48 natural rows top-1 0.4333-0.9306
(aggregate 0.6587), mode 198 at 4.0%, 5906 distinct argmax, top-5 varying across rows; the
three watch rows carry the torch-oracle ladder endorsement (rows 1/8 clean at 0.9946/0.9976
agreement; row 0's exception is disclosed in the sidecar: sample agreement 0.8906 and the
worse-device check pass, its near-tie share is 0.1429). The reference scored 0 against itself
on all 131072 positions.

|                         | logs   | text   | code   | random | all-natural | top-1 |
|-------------------------|--------|--------|--------|--------|-------------|-------|
| EAQ 3.0 bpw (this pack) | 0.2414 | 0.0449 | 0.0331 | 0.0312 | 0.1065      | 0.921 |
| Stock EXL3 3.5 bpw      | 0.2560 | 0.0380 | 0.0280 | 0.0104 | 0.1074      | 0.922 |
| Stock EXL3 2.0 bpw      | 0.6595 | 0.2096 | 0.1821 | 0.1181 | 0.3504      | 0.826 |
| GPTQ 4-bit (g128)       | 0.3911 | 0.0844 | 0.0589 | 0.0373 | 0.1782      | 0.883 |

Mean KL per domain; the EAQ's row-level se is 0.0089 (logs), 0.0030 (text), 0.0025 (code) and
0.0030 for the 48-row natural mean. Paired on the 48 natural rows: EAQ 3.0 vs stock 3.5 is
indistinguishable (mean difference -0.0009 +- 0.0025 nats, 95% CI [-0.0058, +0.0040]; top-1
McNemar chi2 3.74); EAQ is better on the log rows (-0.015) and slightly worse on the text
(+0.007) and code (+0.005) rows. Against the GPTQ 4-bit pack EAQ is better by 0.0717 nats
(23 se, CI [-0.078, -0.066]; McNemar 1252.7) on every domain; against stock 2.0 by 0.244
nats. Including the random rows, the stock 3.5 leads by 0.0045 nats (2.4 se): EAQ's random-row
KL is 0.0312 against its 0.0104 - the one place it is clearly worse.

Caveats. The reference is the device's own bf16 forward, not an fp64 oracle; its watch-row
exception (row 0) is ladder-endorsed and disclosed in the sidecar. The boilerplate log rows
(0/4/6) carry a 3x-headline KL in every pack; the per-domain means include them. The GPTQ
pack's calibration differs (ExLlamaV3's), its body is 4-bit group 128 while its lm_head is
affine 4/64, and the size gap partly reflects what each pack leaves dense. Next-token
accuracy on the natural rows is 65.66% for this pack, 65.77% for the stock 3.5, 64.86% for
stock 2.0 and 64.95% for the GPTQ, against the reference's 65.87% - all in band.
Teacher-forced delta-NLL per token is not reported. The row file is eval64.i32 (sha
c0be24db...), the reference ref_logits_v2.f32 (sha 4bd284fd...), the comparator build
ecbdc8ec... The pre-registered quality gate (mean <= 0.05, top-1 >= 95%, p99 <= 1.0) was not
met by any pack on the 16-row log eval; it was re-registered after the fact (see the
certification section) and the numbers above are the measured values, not a pass claim.

## Usage

TensorFold's XPU runner opens the container directly:

```
xpu-qwen27 <PACK> <ids> <N>
```

The device tests build as `tf-xpu-qwen27-*` (`zig build xpu-qwen27-tests`).

The pack is read through the EAQ container path (`exl3_eaq.zig`) and decoded
through the EXL3 projection path (`forward_exl3.zig`).

## Attributions

This pack's format and method build on the following work:

- **Qwen3.8-27B** — the Qwen Team, Apache License 2.0. The base model; this pack is a quantization of it.
- **EXL3 / ExLlamaV3** — Turboderp, MIT License (Copyright © 2025 Turboderp). The trellis layout and bitstream, the `mul1`/`mcg`/`3inst` codebooks, the half-integer bit widths and the 128-point Hadamard rotations. TensorFold's reader, decoder and XPU kernels are separate implementations, checked bit for bit against ExLlamaV3's dequantization; none of its code ships in this pack.
- **QTIP** — Cornell RelaxML, "QTIP: Quantization with Trellises and Incoherence Processing" (NeurIPS 2024; arXiv 2406.11235; its code is GPLv3). The tail-biting Viterbi trellis search and the block LDL / LDLQ formulation, as ExLlamaV3 carries it. TensorFold's encoder, LDL and LDLQ are its own implementations; no QTIP code is included or linked.
- **Intel** — the Level Zero loader and the Intel compute runtime are loaded at run time and are not bundled; `ocloc` compiles the kernels at build time and is not distributed. The XPU kernels are TensorFold's own.
- **Hugging Face** — the safetensors format (Apache License 2.0) and the Hub.

## License

The base model Qwen3.8-27B is under the Apache License 2.0, and this pack inherits it. The attributions above are reproduced in full in `THIRD_PARTY_NOTICES.md`, with license texts under `LICENSES/`.
