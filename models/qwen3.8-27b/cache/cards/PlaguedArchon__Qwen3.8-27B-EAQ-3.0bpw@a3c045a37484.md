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
- Calibration: 204,800 tiles (cols 2048, rows 100; corpus sha `c4c8b035…`).
- LDLQ: lambda 0.025, seed 224.
- Source: Qwen3.8-27B bf16; all 64 layers plus the head were re-encoded.
- Built on an Intel Arc Pro B70 (96 GB XPU); no NVIDIA hardware was used.

## Certification

- The reader (`openEaq` + `loadExl3` admission) opens the pack: Table complete,
  496 projections, head and embedding present.
- Tamper refusal: a byte-flipped shard refuses with `ShardHashMismatch` naming the
  shard; a symlinked shard refuses with `ShardOutside`; the manifest hash is bound
  by the commit marker.
- Quality: a KL gate against a dense bf16 reference is measured with TensorFold's
  comparator (forward KL, fp64 log-sum-exp, 16 natural rows); see
  `quant_report.json` for the per-part encoder statistics.

## Usage

TensorFold's XPU runner opens the container directly:

```
tf-xpu-qwen27 <PACK> <ids> <N>
```

The pack is read through the EAQ container path (`exl3_eaq.zig`) and decoded
through the EXL3 projection path (`forward_exl3.zig`).

## Provenance

- Base model: Qwen3.8-27B, Apache License 2.0.
- Quantized 2026-10-28 with TensorFold's EAQ quantizer (`xpu-eaq-quant`).
