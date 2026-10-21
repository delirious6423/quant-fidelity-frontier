---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - qwen3.8
  - quantized
  - mtp
  - lobo
---

# Lobo Qwen3.8-27B GSQ-RCO IQ3_S + MTP GGUF

This repository contains the single-file deployment model for [Lobo](https://github.com/alectodescent/lobo-qwen38-16gb), an SM120-specialized Qwen3.8-27B inference appliance for one RTX 5070 Ti 16 GB.

## Artifact

| Field | Value |
| --- | --- |
| Filename | `Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf` |
| Exact bytes | `11,975,960,640` |
| Decimal GB | `11.975960640` |
| GiB | `11.153482497` |
| SHA-256 | `4eb8482539194ed9bc1555c88613f39f2e65db37b16d9ab173f908e78d454512` |
| Intended runtime | [alectodescent/lobo-qwen38-16gb](https://github.com/alectodescent/lobo-qwen38-16gb) |

This is one complete GGUF at runtime. Users do not need the source GGUFs below unless they want to reproduce the assembly.

## Provenance

The artifact is a deterministic payload-preserving assembly of public Apache-2.0 inputs:

- Base model: `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Base quantization: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, revision `d562806dbafae37109975e970aae91b43e73b440`, file `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`.
- MTP donor: `unsloth/Qwen3.8-27B-GGUF`, revision `f975863083b62f54a5e6fac11671c750c2bbc59c`, file `Qwen3.8-27B-UD-IQ3_XXS.gguf`.

All 851 base tensor payloads are byte-identical to the pinned GSQ source. The 15 `blk.64.*` MTP tensor payloads are byte-identical to the pinned donor. No further weight training or approximation was applied during assembly.

Large MTP tensors use IQ4_XS/IQ3_S; small normalization tensors remain F32. The added MTP payload is 204,412,928 bytes.

The exact assembler, input hashes, runtime source, Windows release package, launchers, and behavioural qualification are in the [Lobo public repository](https://github.com/alectodescent/lobo-qwen38-16gb).

## Use

Download this GGUF into the Lobo release's `models` directory, verify it, and start the Balanced profile:

```powershell
.\tools\verify-model.ps1 -Profile Balanced
.\launchers\run-balanced-230k.ps1
```

For an effectively headless RTX 5070 Ti with near-zero desktop GPU residency:

```powershell
.\launchers\run-mtp-headless-262k.ps1
```

The qualified target is Windows 11, CUDA 13.3, and NVIDIA SM120. Other platforms are not claimed.

## License and attribution

The source model repositories identify their weights as Apache-2.0. This repository preserves upstream attribution and links to the exact source revisions. Lobo runtime code is distributed separately under MIT-compatible terms; see its `THIRD_PARTY_NOTICES.md`.
