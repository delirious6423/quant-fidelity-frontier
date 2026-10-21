---
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
- gguf
- llama.cpp
- gemma
- gemma-4
- multimodal
- quantized
- image-text-to-text
- imatrix
- mix-quant
language:
- en
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
---

# Gemma 4 26B A4B it Mix-Quant 13GB GGUF

## File
- Model: `google_gemma-4-26b-a4b-it-mix-13GB.gguf`
- Multimodal projector: `mmproj-gemma-4-26b-a4b-it-f16.gguf`

## Size
- Exact file size: `13,638,779,232` bytes
- Approximate readable size:
  - `12.70 GiB`
  - `13.64 GB`
- Multimodal projector exact size: `1,193,058,432` bytes

## What This Is
This is the smaller mixed-quant target built from the F16 text GGUF for `google/gemma-4-26B-A4B-it`.

It is not a pure uniform quant.
It is a mixed recipe built for `llama.cpp`, with multimodal support preserved through the separate projector file.

## Quantization Type
This release is a `GGUF` quantized model for `llama.cpp`.

Quantization family:
- `GGUF`
- `llama.cpp`
- mixed tensor quantization (`Mix-Quant`)
- `imatrix`-guided quantization

This is not a single uniform `Q4` or `Q3` file.
It is a mixed-precision build where different tensor groups keep different quant types according to sensitivity and size budget.

## Mix Formula
This published 13GB file follows a mixed recipe in this style:
- `token_embd -> q5_k`
- `output -> q5_k`
- `router -> q8_0`
- `attn_q -> q6_k`
- `attn_k -> q6_k`
- `attn_v -> q6_k`
- `attn_output -> q6_k`
- `ffn_gate_up_exps -> mixed q4_k / q3_k`
- `ffn_down_exps -> q4_0`

Notes:
- this file is MoE, so expert tensors are not laid out like the dense 31B recipe
- `ffn_gate_up_exps` is the main mixed expert block
- the 13GB release is therefore closer to a `Q6_K + Q4_K/Q3_K` expert mix than to a `Q3-centered` dense recipe

## Importance Matrix (`imatrix`)
This release follows the same `imatrix`-guided quantization idea used in the 31B line.

Core formula:

`I_j = Σ_t x_{t,j}^2`

Where:
- `x_{t,j}` is the activation value of channel `j` for token/sample step `t`
- `I_j` is the accumulated importance score of that channel across calibration text

Practical meaning:
- channels that activate more often and with larger magnitude get larger importance values
- more important directions are better preserved during quantization
- less important directions can be compressed more aggressively

`imatrix` does not use benchmark scores directly.
It estimates sensitivity from activations collected on calibration data.

## Multimodal Support
Yes. Multimodal remains supported when paired with:
- `mmproj-gemma-4-26b-a4b-it-f16.gguf`

Notes:
- the projector was preserved separately
- the 13GB main file is text-side quantized in the same release style as the 31B line
- image-text usage depends on loading `mmproj` together with the main model

## Road
The working road was:
1. Keep the original HF Gemma 4 26B A4B it model as the source of truth.
2. Export the text model to F16 GGUF.
3. Preserve the multimodal projector as a separate file.
4. Build a mixed 13GB quantized release for local `llama.cpp` inference.
5. Publish the main GGUF together with the projector file.

## Self Tests
Observed checks for the published 13GB release:
- the GGUF file is valid and readable
- the multimodal projector file is present in the repository
- the release remains a multimodal package when used with `mmproj`

Note:
- local experimental variants and local runtime behavior may differ from this published file
- the README here describes the actual uploaded Hugging Face GGUF file, not a guessed local preset name

## Environment Build
Minimal setup:
1. Install CUDA and a recent NVIDIA driver.
2. Build `llama.cpp` with CUDA support.
3. Keep the 13GB GGUF and `mmproj` together if you need vision.
4. Load both files together for multimodal inference.

Example server:
```sh
llama-server \
  -m 'google_gemma-4-26b-a4b-it-mix-13GB.gguf' \
  --mmproj 'mmproj-gemma-4-26b-a4b-it-f16.gguf' \
  -ngl 999 -fa on --ctx-size 4096 -np 1 --port 18081
```

## Datasets And License Notes
This repository is a GGUF release of the Google base model.

License:
- `Apache-2.0`
- official license link: `https://ai.google.dev/gemma/docs/gemma_4_license`

## Practical Summary
Use this version if you want:
- a published mixed 13GB GGUF for `Gemma 4 26B A4B it`
- multimodal support preserved through the separate `mmproj`
- a MoE mixed quant release documented in recipe style instead of a generic quant summary
