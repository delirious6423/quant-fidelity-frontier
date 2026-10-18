---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - deepseek-v4
  - exl3
  - exllamav3
  - quantized
  - mixture-of-experts
  - code
license: mit
---

# DeepSeek-V4-Flash-0731 EXL3 3.0 bpw

This repository contains a **3.0 bpw EXL3 quantization** of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

## Status

| Gate | Status |
|---|---|
| Quantization and artifact assembly | Complete |
| Structural verification and per-file SHA-256 manifest | Passed |
| H200 model load and CUDA kernel compilation | Reached |
| End-to-end generation on H200 | **Not yet passed** |

The weights are complete and structurally verified, but this release is **not being represented as runtime-validated**. The pinned experimental H200/vLLM integration loaded the model and compiled the native SM90 CUDA path, then failed during TileLang host-DSO export with `Target triple should not be empty`. No successful generation result is claimed.

## Artifact

- Target routed-weight rate: **3.0 bpw**
- Achieved routed-weight rate: **3.0 bpw**
- Quantization: EXL3, `mcg` codebook
- Source representation: packed E2M1 FP4 with UE8M0 scales
- Layout: rank-sliced TP4 expert weights
- Weight files: **177**
- Weight bytes: **124,867,114,600** (about 116.29 GiB)
- Tensor records: **534,653**

`EXL3_MANIFEST.json` records the complete artifact inventory and per-file SHA-256 checksums. `CALIBRATION_COVERAGE.json` records the calibration composition and coverage evidence.

## Calibration

The quantization used 1,146,665 calibration tokens spanning four axes:

- general language
- legal text
- code and agentic tasks
- reasoning and termination behavior

Routing was observed naturally; experts were not artificially forced active. The preserved capture contained 688 files totaling 104,303,431,496 bytes, with digest `e31f665d1ffc24b9f80e90ff31e804cf4336efa25f30a033c065afbcf3df5968`.

## Compatibility

These are experimental rank-sliced DeepSeek V4 EXL3 weights. A compatible ExLlamaV3/EXL3 implementation that understands this layout is required. Mainstream drop-in compatibility is **not claimed**, and users should validate generation and structured output behavior in their own target runtime before production use.

For a related runnable deployment of the original DeepSeek V4 Flash 0731 FP8 checkpoint on two DGX Sparks, see [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark). That repository is **not** a drop-in runtime for this EXL3 checkpoint.

## Provenance and acknowledgements

- Base model: [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) by DeepSeek
- EXL3 format and runtime work: [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by TurboDerp and contributors
- EXL3 quantization reference: [brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw) by brandonmusic
- H200 compute for this campaign: RunPod
- Thanks to Inkling and JarvisLabs for support and prior infrastructure used across the broader quantization campaign

The upstream model's license, acceptable-use terms, and limitations continue to apply. This repository changes the weight representation; it does not change the underlying model license.
