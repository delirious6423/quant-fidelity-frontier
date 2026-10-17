---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
tags:
- qwen3.8
- qwen35
- gguf
- llama.cpp
- multimodal
- iq1_m
- q4_k
---

# Qwen3.8-27B IQ1_M GGUF

A runnable multimodal GGUF quantization of `Qwen/Qwen3.8-27B`, produced directly from the pinned two-part BF16 GGUF source at revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`.

## Quantization policy

- The 64 calibrated main transformer blocks use imatrix-aware **IQ1_M**.
- The auxiliary MTP block (`blk.64`) has no entries in the pinned importance matrix, so exactly eight two-dimensional weight tensors are **manually overridden to Q4_K** instead of being forced into an uncalibrated extreme-low-bit type. Other supporting tensors may independently use Q4_K or their normal GGUF types under the IQ1_M policy.
- Those eight matrices contain 424,673,280 parameters, 1.5544% of the 27,320,697,856-parameter model.
- Norm and other non-quantized tensors retain their normal GGUF types.
- The resulting complete model is 2.3045 whole-file bits per weight. This is different from the nominal IQ1_M tensor rate because metadata and the protected Q4_K/F32 tensors are included.

The eight manual block-64 Q4_K override tensors are:

- `blk.64.attn_k.weight`
- `blk.64.attn_output.weight`
- `blk.64.attn_q.weight`
- `blk.64.attn_v.weight`
- `blk.64.ffn_down.weight`
- `blk.64.ffn_gate.weight`
- `blk.64.ffn_up.weight`
- `blk.64.nextn.eh_proj.weight`

## Toolchain

The quantizer is built from llama.cpp `62061f91088281e65071cc38c5f69ee95c39f14e` plus the official MTP accounting fix from PR #24986, merge commit `b3ce5cedf4c007b78a45befe839fa3abada03c0b`. The executable SHA256 is `5d3a8456974b28569322dea7ee33941e3c9f09750e100dfce358326110b29187` and the applied patch SHA256 is `01a53c23afd4ed81b79ffc697b1c4a4b83443253b6095732eee3f8fc58e96b40`. Every loaded llama/ggml shared library is pinned in `QUANTIZER_TOOLCHAIN.json`.

## Files

- `Qwen3.8-27B-IQ1_M.gguf`: model GGUF, 7.33 GiB, SHA256 `131cdf5c1c4b547081543382b00434e9ebf3f8eb369ef3714550086074f80bdf`
- `mmproj-Qwen3.8-27B-BF16.gguf`: vision projector, 888.01 MiB, SHA256 `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53`
- `Qwen3.8-27B-IQ1_M.gguf.validation.json`: structural, text-runtime and vision-runtime acceptance results
- `GGUF_REPORT.json`: tensor-type histogram, architecture and whole-file BPW
- `MTP_Q4K_AUDIT.json`: exact dry-run audit of the eight protected MTP tensors
- `QUANTIZER_TOOLCHAIN.json`: executable, library, patch, source and argv hashes
- `PROVENANCE.md`: source revisions, commands and verification details
- `CHECKSUMS.sha256`: hashes for every release file
- `PROVENANCE.json`: machine-readable model/projector identity contract consumed by the uploader

## RTX 3090 CUDA runtime validation

**PASS (18 August 2026).** The uploaded `Qwen3.8-27B-IQ1_M.gguf` was loaded and generated coherent text on an NVIDIA GeForce RTX 3090 using Ollama 0.32.14 with the CUDA backend.

- CUDA offload: 66/66 layers
- Canonical output: `The answer is 4, and the capital of France is Paris.`
- Generation speed: 24.06 tokens/s
- Observed peak VRAM: 7,772 MiB
- Model SHA256: `131cdf5c1c4b547081543382b00434e9ebf3f8eb369ef3714550086074f80bdf`

The complete request, response, raw GPU samples, runtime logs, hashes and reproduction script are in [`RTX3090_VALIDATION/`](RTX3090_VALIDATION/README.md).

## llama.cpp example

```bash
llama-mtmd-cli \
  -m Qwen3.8-27B-IQ1_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-BF16.gguf \
  --image image.jpg \
  -p "Describe this image."
```

Extreme low-bit quantization trades quality for a much smaller artifact. Use a higher-bit quant when accuracy matters more than footprint.


## Multi-word RTX 3090 generation gate

A second CUDA acceptance explicitly required a string longer than one word. It **passed** with a 22-word grammatical response, `done_reason=stop`, 68% peak GPU utilization and 7,772 MiB observed VRAM. The sanitized record is [`MULTIWORD_RTX3090_VALIDATION.json`](MULTIWORD_RTX3090_VALIDATION.json).

## Agentic/tool-use validation

The original Ollama 0.32.14 structured-call test remains preserved: it scored **0/6** because intended function calls appeared as ordinary assistant text while the API returned `tool_calls=null`.

A direct llama.cpp investigation separated that interface failure from the model's actual serialization ability:

- Direct llama.cpp with one selected tool and string `tool_choice="required"`: **1/1 native structured call passed**.
- Direct llama.cpp with three available tools: **failed raw multi-tool selection**, repeating argument text until the token limit.
- The included loopback adapter normalizes OpenAI named `tool_choice` objects, deterministically selects one explicit tool, and then uses llama.cpp's native structured-call serialization.
- Adapter validation: **6/6 explicit multi-tool cases passed**, plus **1/1 named OpenAI tool-choice object passed**. The returned functions and JSON arguments were exact.

This is a controller-plus-model result, not a claim that raw IQ1_M performs unrestricted semantic routing across arbitrary tool catalogues. The repaired path is reproducible with [`AGENTIC_VALIDATION/RUN_AGENTIC_ADAPTER.sh`](AGENTIC_VALIDATION/RUN_AGENTIC_ADAPTER.sh); full evidence, source, limitations and machine-readable results are in [`AGENTIC_VALIDATION/`](AGENTIC_VALIDATION/README.md).

## One-command RTX 3090 reproduction

[`RUN_RTX3090_FROM_REPO.sh`](RUN_RTX3090_FROM_REPO.sh) verifies the exact model SHA256, installs the validated Ollama 0.32.14 release if Ollama is absent, forces the `cuda_v12` path, imports the local GGUF and generates a multi-word response. It expects a working NVIDIA driver (`nvidia-smi`) and network access only if Ollama must be installed.
