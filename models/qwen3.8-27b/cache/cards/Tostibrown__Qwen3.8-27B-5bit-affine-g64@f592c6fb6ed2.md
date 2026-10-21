---
base_model:
- Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- qwen
- qwen3.8
- qwen3_5
- quantized
---

# Qwen3.8-27B — MLX 5-bit affine g64

A regular MLX quantization of the original `Qwen/Qwen3.8-27B` checkpoint for Apple Silicon.

## What is Mei?

[Mei](https://github.com/tijs/mei) is a small native Swift/MLX, OpenAI-compatible inference server for Apple Silicon. It is designed to provide a reliable local serving path with chunked prefill, streaming and non-streaming parity, tool calls, and KV-cache reuse without requiring the full Osaurus application.

## Why was this model created?

Mei needed a reproducible MLX model that the project controlled end to end, rather than relying only on a third-party conversion. At the time of conversion there was no suitable published 5-bit MLX artifact for this Qwen3.8 checkpoint. This model was therefore produced directly from the pinned original Qwen source so Mei could evaluate the memory/performance trade-off of a 5-bit dense Qwen model and keep a complete, inspectable conversion record for end users.

It is a secondary Mei comparator, not a replacement for the project's primary Ornith-1.5 target. The repository exists so users can reproduce the Mei setup with the exact model artifact used during validation.

## Important identity

This model is **not** an Unsloth Dynamic (UD) quantization and is **not** converted from GGUF. It was produced directly from the original Qwen source checkpoint. No UD-equivalence claim is made.

## Quantization

- Parameters: approximately 27B
- Bits: 5
- Group size: 64 (`g64`)
- Mode: affine
- Floating-point dtype: `bfloat16`
- Output: 4 safetensor shards, approximately 18.5 GB total
- Source: `Qwen/Qwen3.8-27B`
- Source revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Produced with `mlx_lm.convert`

## Validation with Mei

Bounded validation was performed with Mei on Apple Silicon using the generic profile:

- Model load: passed
- Hello probe: 12.81 tokens/s
- Short decode: 13.28 tokens/s
- Peak memory: 21.93 GB
- Context cap: 65,536 tokens
- `SmallVector` crash occurrences: 0

These are single bounded-gate measurements, not a full parity benchmark. Full behavioral comparison against the `unsloth/Qwen3.8-27B-GGUF` `Qwen3.8-27B-UD-Q5_K_M.gguf` reference remains separate work. Results will vary with Apple Silicon model, unified-memory pressure, server settings, and prompt length.

## Usage

Download the model:

```bash
hf download Tostibrown/Qwen3.8-27B-5bit-affine-g64 \
  --local-dir ~/.local/share/mei/models/Qwen3.8-27B-5bit-affine-g64
```

With `mlx-lm`:

```bash
mlx_lm.generate \
  --model ~/.local/share/mei/models/Qwen3.8-27B-5bit-affine-g64 \
  --prompt "Explain how a transformer works."
```

With Mei, clone [the Mei project](https://github.com/tijs/mei), download this repository, and use the downloaded directory as `--model-dir`. Mei automatically selects its disk-backed KV-cache path for this dense Qwen architecture.

```bash
git clone https://github.com/tijs/mei.git
cd mei
hf download Tostibrown/Qwen3.8-27B-5bit-affine-g64 \
  --local-dir ~/.local/share/local-model-bench/mei-models/Qwen3.8-27B-5bit-affine-g64
MEI_MODEL_DIR="$HOME/.local/share/local-model-bench/mei-models/Qwen3.8-27B-5bit-affine-g64" \
MEI_SERVED_MODEL_ID="Tostibrown/Qwen3.8-27B-5bit-affine-g64" \
scripts/start_mei_server.sh
```

Follow Mei's repository instructions for the current build and launcher options. The exact CLI flags may change as Mei evolves.

## Files and provenance

`conversion-provenance.json` records the source revision and tree digest, quantization recipe, output digest and size, conversion command template, and non-UD/non-GGUF claims. Absolute local filesystem paths have been replaced with portable repository-relative identifiers in this public copy.

The Mei source repository contains the server, acceptance probes, benchmark artifacts, and machine-readable model lineup used to validate this artifact:

- Project: [github.com/tijs/mei](https://github.com/tijs/mei)
- Model lineup: [`configs/model-lineup.json`](https://github.com/tijs/mei/blob/main/configs/model-lineup.json)
- Conversion wrapper: [`tools/convert_mlx_quant.py`](https://github.com/tijs/mei/blob/main/tools/convert_mlx_quant.py)

## Caveats

- This is a regular affine 5-bit MLX quant, not the UD-Q5_K_M GGUF recipe.
- The model was validated for loadability and bounded generation; it has not yet completed the full Mei-vs-llama.cpp parity matrix.
- The source model's license and usage terms remain applicable. Check the original [Qwen model repository](https://huggingface.co/Qwen/Qwen3.8-27B) before redistribution or commercial use.
