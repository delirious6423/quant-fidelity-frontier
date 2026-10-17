---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: text-generation
tags:
- gguf
- yvex
- text-generation
model_name: "Qwen3.8-27B Text \u2014 YaiLabs BF16 GGUF"
---

# Qwen3.8-27B Text — YaiLabs BF16 GGUF

YaiLabs-derived GGUF files from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). The original model was authored by Qwen / Alibaba Cloud; YaiLabs performed the conversion and text-only selection using [YVEX](https://github.com/yailabs/yvex).

**Text-only release:** this file contains 851 BF16 tensors for the text decoder, embeddings and output path, selected from 1,199 upstream tensors. It is not the complete upstream multimodal model. Vision execution and MTP acceleration are outside this qualification.

These exact bytes passed structural and tensor verification and the YVEX **load → chat → unload** lifecycle on the NVIDIA DGX Spark configuration documented below. The files are complete, unsharded representations; download only the representation you intend to use.

## Downloads

| File | Exact bytes | SHA-256 |
|---|---:|---|
| [Qwen3.8-27B-Text-BF16.gguf](Qwen3.8-27B-Text-BF16.gguf) | 53815809152 | `1fce07008eaa78e04eedd1a031144f48eb6af617f2b5c508811ba91dca7e00f1` |

## Download and consume

With a current Hugging Face CLI, download the selected file and verify its SHA-256:

```sh
hf download yailabs/Qwen3.8-27B-Text-GGUF Qwen3.8-27B-Text-BF16.gguf SHA256SUMS --revision v1.0 --local-dir .
sha256sum Qwen3.8-27B-Text-BF16.gguf
```

Compare the result with `SHA256SUMS`. The `v1.0` release tag names the verified release commit. For automation, resolve that tag once and pass its full Hub commit ID as `--revision`; do not rely on a floating `main` revision. For DeepSeek, substitute the other filename to select the second representation.

YVEX acquisition remains `yvex model pull SOURCE`; an externally downloaded file can be inspected through its existing local-path intake:

```sh
yvex model pull ./Qwen3.8-27B-Text-BF16.gguf --reference --dry-run --json
```

Acquisition is distinct from runtime preparation. A GGUF download alone does not create an admitted runtime binding/profile. The qualification used YVEX’s canonical binding preparation and an exact-SHA variant, followed by `model load`, hosted chat, `model unload`, and host shutdown. Use a YVEX revision containing the exact physical admission records (qualification revision below or a compatible successor). This release does not claim drop-in execution in every GGUF reader or third-party runtime.

## Representation details

### `Qwen3.8-27B-Text-BF16.gguf`

GGUF v3; architecture `qwen3_5`; 851 tensors; 1/1 files; 32-byte alignment.

| Tensor storage type | Tensor count |
|---|---:|
| BF16 | 851 |

Full tensor descriptors: [CSV](tensors/qwen3.8-text-bf16.csv). Machine-readable identity, lineage, configuration and validation: [release manifest](qwen3.8-text-bf16.release.json).

Observed text conversion/emission plus native roundtrip: **1506.520111 seconds**, 2026-09-05T22:36:29.200819+00:00 to 2026-09-05T23:01:35.720728+00:00. Producing YVEX revision: `628d226aef82899fa1d95dd158d746a9938755d5`; executable SHA-256: `4bde2b2dd34043bc3a5c2a0b8a99a5dfa1f06326bc0e8e0aa07343ac2a1262b4`.

## Provenance and verification

Exact upstream: [`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0). Retained source shards were verified against the pinned file digests before transformation; [source-file identities](provenance/source-files.json) are included. Commands in the release manifests preserve effective arguments with portable placeholders replacing private local paths.

Each output passed full SHA-256 hashing, native emitter roundtrip, comparison with the sealed physical tensor plan, a GGUF metadata reader, and the pinned GGML reader at `af97976c7810cdabb1863172f31c432dab767de7`. Tensor manifests describe these exact output files.

## Exact YVEX qualification

Validation used YVEX `1c7b7218e7264f3d3142fcfd046f85834809a30b` (tree `2ce910bfd6096debc8d86dd76765f2bb8f1f9dd2`) on NVIDIA DGX Spark / GB10: aarch64, 20 ARM CPU cores, 130,663,165,952 bytes RAM, Ubuntu 24.04.4 LTS, kernel `6.17.0-1021-nvidia`, NVIDIA driver `580.159.03`, CUDA toolkit `13.0.88`.

The CUDA target-only lifecycle selected each artifact by its full SHA-256, loaded it, generated a nonempty text response to a short English greeting request, unloaded it, and stopped the host cleanly. Context capacity was 4096; prefill chunk size 64; generation limit 32 tokens; temperature 0; top-p 1; reasoning disabled. Exact dates, outputs and lifecycle duration appear in each release manifest.

This is bounded compatibility evidence for these bytes and this configuration. It is not a general model-quality evaluation, a universal DGX Spark guarantee, an independent numerical-quality score, or a performance benchmark. Build durations include the observed machine load and scheduling. Exact peak working storage and separate historical download timings were not captured; sampled preparation-directory high-water marks are lower bounds.

## License and attribution

The derivative weights remain governed by the upstream **Apache-2.0** license. See the unmodified [LICENSE](LICENSE) and [NOTICE](NOTICE). YaiLabs claims conversion/quantization and validation authorship only, not original training or upstream endorsement.
