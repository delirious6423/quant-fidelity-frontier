---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
  - extensor
  - gguf
  - mixture-of-experts
  - quantized
  - qwen
  - rocm
  - rocmfp4
---

# Qwen3.8-Flash-Next EXTENSOR

An EXTENSOR runtime image derived from
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
It contains the text model only; vision and MTP weights are not included.

## Artifact

- File: `Qwen3.8-Flash-Next-ROCmFP4.extensor.gguf`
- Size: 176,688,558,080 bytes
- SHA-256: `89686029e1eacf35bbf7d928f0e546c682a161a379a03d4c09d67c51615dee3e`
- Quantization: ROCmFP4 experts; F16/F32 residents; BF16 n-gram table

Requires EXTENSOR 3.0.0. This file is not compatible with llama.cpp.

## Source

`Qwen/Qwen3.8-Flash-Next` at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`.

## License

Qwen Community License 1.0. See [LICENSE](LICENSE).
