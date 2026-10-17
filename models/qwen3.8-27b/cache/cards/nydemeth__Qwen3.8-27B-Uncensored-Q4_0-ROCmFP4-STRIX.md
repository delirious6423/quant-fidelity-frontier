---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.8
  - qwen
  - abliterated
  - uncensored
  - rocm
  - rocmfp4
  - strix-halo
library_name: llama.cpp
pipeline_tag: text-generation
---

# Qwen3.8 27B Uncensored — ROCmFP4 STRIX

A Strix Halo-optimized GGUF quantization of the refusal-reduced `Qwen3.8-27B` model, with its separate MTP/speculative-decoding assistant.

This release targets AMD Strix Halo systems using a ROCmFPX build that supports the custom `Q4_0_ROCMFP4_STRIX` format. It is not a standard upstream llama.cpp quantization.

## Files

| File | Format | Size | Purpose |
|---|---:|---:|---|
| `Qwen3.8-27B-Uncensored-Q4_0_ROCMFP4_STRIX.gguf` | GGUF `Q4_0_ROCMFP4_STRIX` | 14,760,069,984 bytes (13.75 GiB) | Main model |
| `mtp-Qwen3.8-27B-Uncensored-Q4_0.gguf` | GGUF Q4_0 | 2,034,268,960 bytes (1.89 GiB) | MTP/speculative-decoding assistant |
| `SHA256SUMS` | SHA-256 | — | Checksums for both GGUF files |

Total download size is approximately 16.79 GB (15.64 GiB). The MTP file is optional and is not a standalone language model; it requires the matching main model.

This package does not include a vision projector and is intended for text-only serving. The MTP assistant can be omitted when speculative decoding is not wanted.

## Usage

Use a recent ROCmFPX build with Qwen3.8/Qwen3.5 hybrid-architecture and MTP (`nextn`) support. For example:

```bash
llama-server \\
  --model Qwen3.8-27B-Uncensored-Q4_0_ROCMFP4_STRIX.gguf \\
  --model-draft mtp-Qwen3.8-27B-Uncensored-Q4_0.gguf \\
  --jinja
```

The ROCmFP4 STRIX main model was produced and tested with ROCmFPX revision:

```text
c49ebdbd5c9f01ec242369f9e7f7967855f80cba
```

The MTP assistant was validated locally with the matching main model for speculative drafting. Compatibility with other llama.cpp/ROCmFPX revisions is not guaranteed.

## Provenance

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- Refusal-reduced source model: [`orcarouter/Qwen3.8-27B-Uncensored`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored)
- Related upstream GGUF releases: [`orcarouter/Qwen3.8-27B-Uncensored-GGUF`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF)
- Conversion/runtime project: [ROCmFPX](https://github.com/charlie12345/ROCmFPX)

The source model is described by its publisher as an abliterated/refusal-reduced derivative. This repository distributes the associated quantized artifacts and does not claim to reproduce or improve the abliteration method.

## Privacy

This repository contains model weights, documentation, and checksums only. No local prompts, conversations, logs, prompt caches, credentials, host paths, or personal files are intentionally included. As with any pretrained model, memorized data from the upstream training corpus cannot be ruled out solely by inspecting the quantized weights.

## License and responsible use

The base model is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The uploader makes no additional license claim beyond the applicable upstream terms. Preserve upstream attribution and license obligations when redistributing these artifacts.

This model has substantially reduced refusal behavior. It is intended for controlled research, red-teaming, interpretability, and guardrail evaluation—not unmoderated public or production deployment. Users are responsible for complying with the license, applicable law, and Hugging Face policies.
