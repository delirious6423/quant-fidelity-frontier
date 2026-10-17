---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/f5d08274bafd880402bd16f5e3e6c514136ec06c/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - gguf
  - qwen4exp
  - qwen3.8-flash-next
  - dgx-spark
  - ds4
---

# Qwen3.8-Flash-Next GGUF

## Support my work

I work on making large language models practical on hardware they were never really designed to fit on — through mixed quantization, inference optimization, custom kernels, and serving experiments.

While much of the development happens on local hardware, calibration, profiling, and large-scale validation often require expensive on-demand GPUs.

Contributions help pay for that compute, storage, and testing infrastructure, so I can keep experimenting and publishing the results openly.

<a href="https://www.buymeacoffee.com/baekpica" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" style="height: 60px !important;width: 217px !important;"></a> <a href="https://github.com/sponsors/Baekpica" target="_blank"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor Baekpica on GitHub" style="height: 60px !important;width: 217px !important;"></a>

GGUF conversions of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), pinned to source revision [`f5d08274bafd880402bd16f5e3e6c514136ec06c`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/f5d08274bafd880402bd16f5e3e6c514136ec06c).

## Compatibility status

These files use a custom `qwen4exp` GGUF architecture prepared for the [`dfm` branch of Baekpica/ds4](https://github.com/Baekpica/ds4/tree/dfm), with DGX Spark/GB10 as the deployment target. Runtime support is still under implementation and validation. Do **not** assume compatibility with upstream llama.cpp or other GGUF runtimes unless they explicitly support this architecture and tensor schema.

## Available variants

| Variant | Status | Notes |
|---|---:|---|
| `BF16` | verified / available | Lossless reference conversion. Source payload bits are preserved after semantic tensor splits. |
| `Q8_0` | verified / available | Most weight matrices use Q8_0; numerically sensitive or unsupported tensors remain BF16/F32/I64. |

A separately tuned mixed-quant release will be published only after calibration, H200 quality checks, and ds4 runtime validation.

## BF16 verification

The BF16 release contains 1,756 GGUF tensors in 12 shards, totaling 360,011,029,056 bytes. It was checked against all 1,658 tensors from the pinned source revision:

- all 359,999,963,128 source payload bytes compared exactly;
- zero payload mismatches;
- fused expert gate/up tensors were split semantically without numerical conversion;
- routed expert down projections were split losslessly into the main 512-column region and the 128-column tail;
- GGUF shard metadata, tensor offsets, shapes, types, and Qwen Community License metadata were validated;
- per-shard SHA-256 checksums are included alongside the files.

The source snapshot itself was also checksum-verified with `hf cache verify` before conversion.

## Q8_0 verification

The Q8_0 release contains 1,756 GGUF tensors in 7 shards, totaling 192,201,208,384 bytes. Its audited tensor distribution is 806 Q8_0, 363 BF16, 584 F32, and 3 I64 tensors. GGUF metadata, shard numbering, tensor names, shapes, offsets, declared payload ends, and types were checked against the same exhaustive pinned source map with zero structural errors.

Per-shard SHA-256 checksums are included in `Q8_0/SHA256SUMS`. All seven public Hugging Face LFS object hashes and remote byte sizes were also compared with the local artifacts after upload and matched exactly.

## Architecture notes

This conversion retains the complete multimodal and speculative-decoding topology: 48 text layers, 36 gated-delta layers, 12 full-attention layers, 512 routed experts with top-10 routing, shared experts, four hyper-connection streams, a 51.2B-parameter PLE n-gram table, the vision tower, and the MTP layer.

## License

Use of these converted weights is governed by the original **Qwen Community License 1.0**. The exact upstream license file is included in this repository. No Apache-2.0 license is claimed for these weights.

## Reproducibility

Conversion, verification, calibration, mixed-quant recipe, and ds4 runtime materials are being prepared for publication with the validated mixed-quant handoff. Until runtime validation is complete, the BF16 and Q8_0 files should be treated as conversion artifacts rather than a ready-to-run general-purpose release.
