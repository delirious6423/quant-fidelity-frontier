---
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
license: other
pipeline_tag: image-text-to-text
tags:
- mxfp4
- fp8
- multimodal
- lossless-compression
- csf
- lil-license-1.0
license_name: lil-license-1.0
license_link: LICENSE
lil_license:
  spdx: LicenseRef-LIL-1.0
  name: Local Inference Lab License, Version 1.0
  licensor: Local Inference Lab, Inc.
  work_name: DeepSeek-V4.1-Flash-lossless-CSF
  original_location: https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF
  canary: Ні пуху, ні пера
  canary_id: LIL-CANARY-4A08-5BF0-5CDE-25E4
---

# DeepSeek-V4.1-Flash-lossless-CSF

A lossless MXFP4-CSF container of [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) at revision `dba1be0a40aa45a94ad051997016db3960a90277`.

CSF stores the routed-expert block-scale planes in a compressed form. Nothing else changes: FP4 weight nibbles, every other tensor, the tensor names, dtypes, shapes and the source metadata are all byte-identical. Decoding is integer arithmetic on bytes, with no requantization or fitting. The original safetensors shards can be restored bit-exactly.

```
Routed experts (40 layers x 384) - MXFP4 (E2M1, UE8M0 scale per 32)
Engram tables, attention, shared experts - FP8 E4M3 with UE8M0 scales
Vision encoder, aligner, MTP (including its routed experts) - source format

Routed-expert block scales - lossless MXFP4-CSF (row-base-offset1-u24-exceptions/1)
```

## Sizes

- Weight files: 510.30 GB in the source, 495.63 GB here (14.67 GB saved).
- Compressed scales: 46,080 matrices (UE8M0 block scales of the 40 x 384 x 3 main-layer routed-expert projections): 16.99 GB -> 2.31 GB (13.6%).

## Provenance

- Source: `deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a40aa45a94ad051997016db3960a90277`, 48 index-referenced safetensors shards. During export, each source shard's SHA-256 was checked against its Hugging Face LFS hash.
- Built with trellis-quant `trellis_quant.lossless_scale_checkpoint` (commit `60feca330087`, family `deepseek_v41`).
- `verification.json`: every original shard was rebuilt from this container, and both the rebuilt and the stored files matched their SHA-256 (46,080 scale matrices, 48 shards, passed).
- Hub `main` (2cba9e42) adds `chat_template.jinja` and is otherwise byte-identical to dba1be0a. Every tensor and config file matches.
- It replaces the earlier `DeepSeek-V4.1-Flash-MXFP4-CSF` repository, which has been removed: `tensors/` is byte-identical (all 48 compressed-shard SHA-256 values), and this build adds the source LICENSE to `metadata/` and records the source revision.

## Layout

`lil-mxfp4-csf-checkpoint/1`, codec `row-base-offset1-u24-exceptions/1`:

- `tensors/` - the source shard names; each routed-expert scale `<name>` is stored as `<name>.mxfp4_csf_fixed` (uint8) plus `<name>.mxfp4_csf_exceptions` (uint32)
- `metadata/` - byte copies of the source's config, tokenizer, index, README and LICENSE
- `config.json` - a copy of `metadata/config.json` at the top of the repository, where the Hub counts downloads; runtimes read `metadata/`
- `manifest.json`, `build-contract.json`, `receipts/` (per-shard source headers and hashes), `verification.json`
- `LICENSE`, `NOTICE`, `LICENSES/` (upstream license texts), `CITATION.cff`, `REUSE.toml`, and the SHA-256 of every file in `SHA256SUMS` and `lil-manifest.json`

There is no top-level `model.safetensors.index.json`, so a plain safetensors loader will not open this directory by mistake.

## Serving

Use vLLM with the MXFP4-CSF reader: `--quantization mxfp4_csf --load-format mxfp4_csf`. Point vLLM at a serving directory that holds the files of `metadata/`, with `config.json`'s `quantization_config` replaced by:

```json
{
  "...": "every key of metadata/config.json quantization_config",
  "quant_method": "mxfp4_csf",
  "format_version": 1,
  "checkpoint_root": "/path/to/this/checkpoint"
}
```

The weights are read from `checkpoint_root`; the serving directory holds only metadata.

## Verify or restore

```bash
PYTHONPATH=/path/to/trellis-quant python3 -m trellis_quant.lossless_scale_checkpoint verify \
  --checkpoint /path/to/this/checkpoint --workers 16
PYTHONPATH=/path/to/trellis-quant python3 -m trellis_quant.lossless_scale_checkpoint restore \
  --checkpoint /path/to/this/checkpoint --destination /path/to/fresh/dir --workers 16
```

`verify` rebuilds every original shard and compares SHA-256 values. `restore` writes the original shards and metadata files into a fresh directory, checking every hash before a file is published. The result is an ordinary copy of the source checkpoint.

## Upstream license

The source model, [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), is licensed by DeepSeek under the MIT License. Its license text is in `LICENSES/MIT.txt`, `metadata/LICENSE` keeps the source copy unchanged, and `NOTICE` names the upstream materials. This container is licensed under the Local Inference Lab License, Version 1.0 (`LICENSE`); see [License and attribution](#license-and-attribution) below.

<!-- lil-license-section:begin (managed by lilkit) -->
## License and attribution

DeepSeek-V4.1-Flash-lossless-CSF is licensed under the [Local Inference Lab License, Version 1.0](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF/blob/main/LICENSE), which reproduces the terms and conditions of the Apache License, Version 2.0, and adds conditions that restrict them. It is not the Apache License, and these files are not open source. Copyright (c) 2026 Local Inference Lab, Inc., a non-profit organization, 311 West Main Street, Grayson, Kentucky 41143. Revisions published up to and including revision c5c41fe301c4c1d24fc09376883d9168e521dc66 were released under the MIT License, and copies of those revisions remain under it. The Local Inference Lab License, Version 1.0 applies to later revisions.

- **No reuploads.** Do not upload, mirror, or redistribute these files, or any Substantially Similar Copy of them (including renamed, re-sharded, re-packaged, metadata-stripped, converted, or dequantized copies). Link to <https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF> instead.
- **Attribution at the top.** Every README, model card, or other landing page for a project, model, dataset, application, service, or distribution that contains, is derived from, or runs DeepSeek-V4.1-Flash-lossless-CSF must begin with this Attribution Notice as its first paragraph (a single title line may come before it):

> This model is based on [DeepSeek-V4.1-Flash-lossless-CSF](https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF) by Local Inference Lab, Inc., a non-profit organization, available at <https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF>. DeepSeek-V4.1-Flash-lossless-CSF is licensed under the Local Inference Lab License, Version 1.0.

  Plain-text form:

```text
This model is based on DeepSeek-V4.1-Flash-lossless-CSF by Local Inference Lab, Inc., a non-profit organization, available at https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF. DeepSeek-V4.1-Flash-lossless-CSF is licensed under the Local Inference Lab License, Version 1.0.
```

- **Keep the marks.** Do not remove the license metadata, the Identifying Marks ("Ні пуху, ні пера", LIL-CANARY-4A08-5BF0-5CDE-25E4), or the copyright notices embedded in these files.
- **Breach.** A breach of the no-reupload or attribution terms ends the license immediately (Section 5.1), and Local Inference Lab, Inc. may ask hosting services to remove the material (Section 5.2).

File integrity: `SHA256SUMS` and `lil-manifest.json` list the SHA-256 of every file.
<!-- lil-license-section:end -->
