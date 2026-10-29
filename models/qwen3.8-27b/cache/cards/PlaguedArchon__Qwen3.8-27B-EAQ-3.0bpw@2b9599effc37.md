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

Measured as forward KL(reference || pack) in nats, fp64 log-sum-exp, on 16 rows x 2048
positions of log-domain text plus 4 rows x 2048 positions of random tokens. The reference is
the dense bf16 model run through the same Intel XPU serving path (fp16 between kernels, fp32
inside); it is not an fp64 forward. The reference scored 0 against itself on all 40960
positions.

|                         | mean   | median | top-1 | >=1 nat | >=10 nat | random-row mean |
|-------------------------|--------|--------|-------|---------|----------|-----------------|
| EAQ 3.0 bpw (this pack) | 0.241  | 0.0066 | 91.4% | 3.08%   | 0.79%    | 0.029           |
| Stock EXL3 3.5 bpw      | 0.256  | 0.0078 | 90.6% | 3.30%   | 0.80%    | 0.009           |
| Stock EXL3 2.0 bpw      | 0.660  | 0.0507 | 81.3% | 7.82%   | 2.15%    | 0.107           |
| GPTQ 4-bit (g128)       | 0.391  | 0.0210 | 86.4% | 4.80%   | 1.16%    | 0.035           |

Paired against stock 3.5 on the same positions, EAQ 3.0 is not statistically distinguishable
(mean difference -0.015 nats, row-level t = 1.6 over 16 rows; better on 11 of 16 rows).
Against stock 2.0 it is better on all 16 rows (mean difference -0.418 nats). On random-token
input EAQ 3.0 is about 3x the stock 3.5 KL and sits between the 2.0 and 3.5 packs. Against a GPTQ 4-bit (g128, 19 GB) pack measured on
the same rows, EAQ 3.0 is better on all 16 rows (paired mean -0.15 nats, row-level t = 2.5),
with a lower median (0.0066 vs 0.0210), higher top-1 (91.4% vs 86.4%) and smaller tails, at
12 GB vs 19 GB; the engine's GPTQ path decodes about 9x faster (33.9 vs 3.7 tok/s on the same
box).

Caveats. The text rows are a single domain (logs); no general-text eval has been run. One row
contributes 44-46% of the total KL across the packs (46% for EAQ and both EXL3 packs, 44% for
the GPTQ); excluding it, the means are 0.140 (EAQ), 0.148 (stock 3.5), 0.436 (stock 2.0),
0.235 (GPTQ). Over half of each mean comes from the 3% of positions above 1
nat. Next-token accuracy on the eval rows is 65.3% for this pack and 65.25% for the reference;
KL measures fidelity to the reference, not task quality. The calibration set is 204 model
README files (204,800 token ids), disjoint from the eval's log rows. The pre-registered
quality gate (mean <= 0.05, top-1 >= 95%, p99 <= 1.0) was not met by this pack or by either
stock pack, including stock 3.5; the gate was set for general text at higher bit widths and
is reported here as failed.

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
