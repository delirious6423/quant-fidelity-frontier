---
base_model: "deepseek-ai/DeepSeek-V4.1-Flash-1017"
base_model_relation: quantized
license: other
license_name: lil-license-1.0
license_link: LICENSE
tags: [mxfp4, lossless-compression, csf, lil-license-1.0]
lil_license:
  spdx: LicenseRef-LIL-1.0
  name: Local Inference Lab License, Version 1.0
  licensor: Local Inference Lab, Inc.
  work_name: "DeepSeek-V4.1-Flash-1017-lossless-CSF"
  original_location: "https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF"
  canary: "Ні пуху, ні пера"
  canary_id: "LIL-CANARY-3272-AD41-4E28-4DD8"
---

# DeepSeek-V4.1-Flash-1017-lossless-CSF

This model is based on DeepSeek-V4.1-Flash-1017-lossless-CSF by Local Inference Lab, Inc., a non-profit organization, available at https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF. DeepSeek-V4.1-Flash-1017-lossless-CSF is licensed under the Local Inference Lab License, Version 1.0.

Lossless MXFP4-CSF checkpoint of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

Original location: https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF.

Selected MXFP4 block-scale planes use integer byte compression without requantization. Every other tensor payload and the source files in `metadata/` retain their original bytes.

Container schema: `lil-mxfp4-csf-checkpoint/1`. Codec: `row-base-offset1-u24-exceptions/1`.
Compressed scale matrices: 46,080. Scale bytes: 16,986,931,200 source, 2,316,453,540 encoded.
Shard bytes: 510,296,708,312 source, 495,632,175,852 encoded.

Recorded storage verification passed: `true`. Qualification: "storage and CPU reconstruction". Storage verification does not qualify model serving.

`tensors/` contains the source shard basenames, with selected scale tensors replaced by `.mxfp4_csf_fixed` (U8) and `.mxfp4_csf_exceptions` (U32). `receipts/` preserves the original shard headers and hashes. `build-contract.json` records the original tensor inventory. Reconstruct original shards using the exact receipted header bytes and original payload offsets; validate their SHA-256 against the receipts.

Consumers read `metadata/` for the original config, tokenizer, and source index. The root `config.json` is an unchanged copy; there is no root safetensors index. A CSF-aware checkpoint loader is required.

`lil-manifest.json` and `SHA256SUMS` inventory all package files except those two inventory files themselves. Shard hashes are the conversion results recorded in `manifest.json`.

The container is licensed under the Local Inference Lab License, Version 1.0; upstream deepseek-ai/DeepSeek-V4.1-Flash-1017 material remains under MIT. See `LICENSE`, `NOTICE`, `LICENSES/`, and `metadata/LICENSE`.

Identifying marks: Ні пуху, ні пера; `LIL-CANARY-3272-AD41-4E28-4DD8`.
