---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- llama.cpp
- quantized
- staging
---

# Nyquant staging: Qwen3.8-27B (unreleased research artifacts)

**This is not a release.** It is off-site storage for research builds from the Nyquant quantization lab: candidate GGUFs, build manifests and checksums.

- Nothing here is recommended for use.
- No quality claim is made here.
- Files can be replaced or superseded at any time.
- Released files, once a build passes our evaluation rules, will live in a separate repository with a full model card.

## What you will find
- `<build-id>/<build-id>.gguf`: candidate quantized builds, named by build id, not product name.
- `<build-id>/MANIFEST.json`:
  - complete file size and sha256;
  - the llama.cpp commit used;
  - the per-tensor type map;
  - the **provenance** of every tensor: which public file it was taken from, or how it was produced.
- `<build-id>/NOTICE`: attribution for any third-party source the build includes.

## Provenance and licenses
- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Apache-2.0.
- **Derived builds:** some candidates combine our own quantized tensors with tensors taken unchanged from publicly released Apache-2.0 GGUF quantizations of the same base model. Each such build's `NOTICE` and `MANIFEST.json` name the source repository and file of every reused tensor. Each source's license and copyright notices apply to those parts.
- Files in this repository are provided under Apache-2.0, subject to the licenses of their sources.

## Contact and status
These artifacts belong to work in progress. Evaluation results are published only with a release, after paired, equal-size, equal-runtime comparisons.
