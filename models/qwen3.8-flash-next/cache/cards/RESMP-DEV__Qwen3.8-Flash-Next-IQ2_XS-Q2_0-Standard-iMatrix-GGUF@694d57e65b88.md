---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- llama-cpp
- quantization
- imatrix
- iq2_xs
- q2_0
- qwen3.8
- mixture-of-experts
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
---

# Qwen3.8 Flash-Next · IQ2_XS / Q2_0 standard-iMatrix GGUF

This repository contains a **research static requantization** of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
It is not a runtime-dynamic quantization and it is not a general-purpose
release recommendation. The routed expert weights are fixed at IQ2_XS or Q2_0
when written to the GGUF; routing at inference remains dynamic as in the
original mixture-of-experts model.

> **Status:** quantized, integrity-checked, packed, and loaded by Strata on a
> real RTX 5070 Laptop host. A one-token greedy gate and a same-day public-iMatrix
> control both produced token 157. No perplexity, task-quality, long-generation,
> or broad throughput evaluation has been run.

## Important scope and disclaimer

- The source was an **already quantized IQ4_XS GGUF**, not the original BF16
  checkpoint. This artifact is therefore a compounded research requantization;
  it must not be interpreted as a quality-equivalent low-bit conversion from
  BF16.
- 96 routed gate/up tensors use **IQ2_XS** and 48 routed down tensors use
  **Q2_0**. The other 1,080 tensors retain the intermediate GGUF's existing
  types and shapes.
- Calibration used a private RESMP-DEV exact-token 256-window by 2,048-token
  standard iMatrix view. The calibration corpus and importance matrix are not
  published with this model.
- The GGUF retains F32 router tensors. Strata's native pack path converts those
  routers exactly to BF16 while leaving expert and PLE bytes unchanged.
- This is not a default Strata model choice and has not been promoted beyond
  research.

## At a glance

| Property | Value |
| --- | --- |
| Architecture | Qwen3.8 Flash Next / qwen4exp mixture-of-experts |
| Runtime tested | Strata engine 0.1.41 |
| Tensor count | 1,224 across two llama.cpp GGUF shards |
| Routed gate/up experts | 48 IQ2_XS tensors |
| Routed down experts | 48 Q2_0 tensors |
| Non-target tensors | 1,080 preserved source types and shapes |
| Unsharded artifact size | 66,649,935,712 bytes |
| Reported quantized size | 63,551.82 MiB / 3.01 bits per weight |
| Calibration view | 256 disjoint windows of 2,048 token IDs |
| Deterministic IQ2_XS path | Enabled for the described research build |

## Files

These are a matching llama.cpp GGUF split set; neither shard is an independent
model. Sizes below are decimal GB and describe files on disk, not runtime
memory.

| File | Size | SHA-256 |
| --- | ---: | --- |
| [Qwen3.8-Flash-Next-IQ2_XS-Q2_0-Standard-iMatrix-00001-of-00002.gguf](./Qwen3.8-Flash-Next-IQ2_XS-Q2_0-Standard-iMatrix-00001-of-00002.gguf) | 39.969 GB | `3c69f42a305c15209893f0e1c8b70ffcbc9dd8346cdd7c781e7edefbe42b2915` |
| [Qwen3.8-Flash-Next-IQ2_XS-Q2_0-Standard-iMatrix-00002-of-00002.gguf](./Qwen3.8-Flash-Next-IQ2_XS-Q2_0-Standard-iMatrix-00002-of-00002.gguf) | 26.681 GB | `adf8e9e382276a7ff20150c0f3701adb847c4a140895a693367c4f7400123036` |

The unsharded source artifact has SHA-256
`f88ddbabfe3a7c307c4685ae524efddcceada6e5b427d0d5aa574a53a767cc08`.
Splitting and merging reconstruct GGUF split metadata and alignment, so the
merged file need not have the same whole-file hash. Tensor-level verification
is authoritative: all 1,224 names, shapes, types, byte counts, and payload
hashes matched before upload.

## Usage

Download both shards to the same directory. Keep the complete filenames
together:

```bash
hf download RESMP-DEV/Qwen3.8-Flash-Next-IQ2_XS-Q2_0-Standard-iMatrix-GGUF \
  --local-dir ./Qwen3.8-Flash-Next-IQ2_XS-Q2_0-Standard-iMatrix-GGUF
```

A current llama.cpp `llama-gguf-split --merge` can reconstruct a single GGUF
from shard 1. Use the exact binary and commands for your llama.cpp version;
merge behavior was verified, but llama.cpp text generation was not part of this
artifact's validation. Strata validation used the unsharded GGUF and Strata's
native pack path.

## Calibration and conversion

The input was
[mradermacher/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-GGUF),
file `Qwen3.8-Flash-Next.IQ4_XS.gguf`, pinned at revision
`68d514cd9f411c9d6b62f8a422102a202d57c18c` and SHA-256
`ea9b7537389918d69bcf3ea1fb5693d427ae82278dd67f415fa9383858b51efb`.

The RESMP-DEV calibration view contains 524,288 token IDs in 256 disjoint
2,048-token windows. Its resulting importance matrix has SHA-256
`92c6a742027b569907a2ce3cd2bf12b3ae5ea037c86105ef7ba4ae244a43fc89` and 902
entries. The matrix itself is retained locally and is not required for
inference.

Quantization used llama.cpp revision
`3cf03257f219afbe7334045ff7c6a06ac68c627d` with two opt-in research paths:
double-precision deterministic IQ2_XS derived-weight arithmetic and a complete
anchored per-tensor COPY recipe. The full run used 24 threads, took 27:48.85,
and peaked at 29.33 GiB RSS.

## Validation

The artifact's 1,224 tensor names and shapes match the source. Exactly 144
intended routed-expert conversions occurred: 96 gate/up tensors became IQ2_XS
and 48 down tensors became Q2_0. No non-target tensor type changed.

Strata loaded 32.23 GiB of native experts at 6.93 GiB/s, selected 2,233
expert-cache slots on an RTX 5070 Laptop GPU, and generated greedy token 157
for prompt tokens `[100,101,102]`. Measured rates for that one-token gate were
26.33 tokens/s decode and 6.83 tokens/s prefill. A same-boot control using the
prior public-matrix artifact, the same engine, and the same prompt also produced
token 157 at 24.96 tokens/s decode and 5.45 tokens/s prefill.

These checks prove artifact integrity and execution, not quality. The speed
comparison is a same-day observation, not an interleaved confidence interval.
No perplexity, coding, reasoning, agentic, long-context, vision, or general
throughput benchmark has been run.

## Reproducibility and integrity

- Source GGUF: 98,421,788,160 bytes; SHA-256 `ea9b7537…b51efb`.
- Importance matrix: 579,789,056 bytes; SHA-256 `92c6a742…a43fc89`.
- Unsharded artifact: 66,649,935,712 bytes; SHA-256 `f88ddbab…67cc08`.
- Split/merge tensor verification: 1,224 matched payloads, 0 mismatches.
- Strata source revision: `e19b1d73a08ff4205925b0bc5697c70ec1e6bded`.

See [SHA256SUMS](./SHA256SUMS) and [provenance.json](./provenance.json) for
structured integrity data.

## Credits

Original model: Qwen. Intermediate GGUF: mradermacher. Quantization runtime:
ggml-org/llama.cpp. Native engine validation: Strata. Standard exact-token
iMatrix view, deterministic research quantization, split verification, and
publication: RESMP-DEV.
