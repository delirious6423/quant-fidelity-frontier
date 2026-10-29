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
- Quality: **pending** — the KL gate against a dense bf16 reference (forward KL, fp64
  log-sum-exp, 16 natural rows) is not yet issued; the comparator that produces it is being
  fixed. The certification above is integrity, not a quality number. See `quant_report.json`
  for the per-part encoder statistics.

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
