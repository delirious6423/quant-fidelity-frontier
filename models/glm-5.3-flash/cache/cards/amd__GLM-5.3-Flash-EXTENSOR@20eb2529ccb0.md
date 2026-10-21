---
license: mit
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - extensor
  - gguf
  - mixture-of-experts
  - quantized
  - glm
  - rocm
  - rocmfp4
---

# GLM-5.3-Flash EXTENSOR

An EXTENSOR runtime image derived from
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
It contains the text model only; vision and MTP weights are not included.

## Artifact

- File: `GLM-5.3-Flash-ROCmFP4.extensor.gguf`
- Size: 173,462,265,856 bytes
- SHA-256: `4a682bbab963efdb58764d64863c4cc34d6cb71893cc8aee55c17342239c5e8e`
- Quantization: ROCmFP4 experts and selected resident projections; F16/F32 remaining tensors

Requires EXTENSOR 3.0.0. This file is not compatible with llama.cpp.

## Source

`zai-org/GLM-5.3-Flash` at revision
`eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.

## License

MIT License. See [LICENSE](LICENSE).
