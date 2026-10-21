---
license: apache-2.0
library_name: llama.cpp
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
quantized_by: ROCmFPX
language:
- en
tags:
- gguf
- rocm
- amd
- strix-halo
- gfx1151
- rocmfpx
- quantized
- magicquant
---

# Qwen3.8-27B-ROCmFPX-GGUF

> **Engine requirement:** these GGUFs use ROCmFPX tensor types added to the fork in
> August 2026. A fork build **newer than 2026-08-06** is required; older builds fail to
> load with `tensor 'output.weight' has invalid ggml type 102`. Stock llama.cpp cannot
> load these files at all -- for stock llama.cpp use the sibling
> [MagicQuant repo](https://huggingface.co/lmcoleman/Qwen3.8-27B-MagicQuant-GGUF).

> **Measured quality (2026-08-16, wikitext-2 PPL, ctx 512, 100 chunks, BF16
> baseline 6.7443):** MQ-Q6 measures **6.7412** — a tie with the stock Q6_K
> (6.7470). MQ-Q4 measures **6.9240** (+2.7% vs baseline), a real quality loss
> against its MagicQuant source (Q4_K_M: 6.7522). If quality at Q4 size matters
> more than fork-native types, use the sibling repo's Q4_K_M; MQ-Q4 remains the
> right pick only where the fork's FP4 execution path is the point.


> ## ⚠️ These files do NOT load on standard llama.cpp
> They use AMD-native `*_ROCMFPX` tensor types from the experimental
> [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) llama.cpp fork (build from source).
> For files that work with stock llama.cpp / LM Studio / Ollama, use the sibling repo:
> [lmcoleman/Qwen3.8-27B-MagicQuant-GGUF](https://huggingface.co/lmcoleman/Qwen3.8-27B-MagicQuant-GGUF).

Derivative of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized using MagicQuant hybrid evolutionary per-tensor search and quantized to AMD-native [ROCmFPX](https://github.com/ciru-ai/ROCmFPX) formats (fork-only) tuned for Strix Halo (gfx1151).

## Base Model

This is a derivative of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
All credit for the base model architecture and weights goes to the original authors.
The base model's license applies to this derivative.

## Quantization Method

Quantized using **[MagicQuant](https://github.com/lucasmcoleman/MagicQuant)** hybrid evolutionary per-tensor quantization,
based on the methodology by **[magiccodingman](https://github.com/magiccodingman/MagicQuant-Wiki)**:

- Tensors are classified into sensitivity groups (Embeddings, Head, Query, Key, Output, FFN Up/Down, MoE Experts, Router)
- An evolutionary search finds the optimal quantization type per group, balancing size vs. perplexity
- **Q4/Q5/Q6 tier targets** are searched, and each one ships only if it earns its place (see below)
- Small-row tensors and sensitivity-critical layers (embeddings, output head, router) are kept at F32/F16/BF16
- This is NOT a uniform quantization -- each tensor group gets its own optimal type

A tier name here is a **size band**, not a promise that every tensor uses that
exact type. A "Q5" is whatever mix of schemes landed in the Q5 size band with
the lowest measured perplexity loss -- which is the point of the search.

## Tiers this build does not produce

- **Q5** -- rendering MagicQuant's Q5 config into ROCmFPX types predicts 20.68 GiB against a 50.89 GiB BF16 baseline (ratio 0.4063), which is the Q6 band, not Q5.

These were not built at all. This is a property of how the schemes round into the ROCmFPX type ladder for this particular model, not a temporary gap, so a file for them will not appear in a later build either.

## ROCmFPX (AMD-native, fork-only)

These GGUFs use AMD-native quantization schemes from the experimental
**[ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX)** llama.cpp fork,
tuned for and benchmarked on AMD Strix Halo (Radeon 8060S iGPU, gfx1151, unified memory):

- `ROCmFP3/4/6/8` tensor types with straight and "agent" presets (agent presets keep
  tool-calling / JSON-structured output reliable at low bit-widths)
- Files load **only** on the fork -- it is an experimental upstream research
  build, so build from the pinned commit that produced these files (the
  default branch may have moved on since):

```bash
git clone https://github.com/ciru-ai/ROCmFPX.git ROCmFPX
cd ROCmFPX
git checkout 68f23f34c12d7e61177a034b0d8d3fea2129565e
# then build per the fork's own README
```

## GGUF Files

| File | Size | Quant | Perplexity vs BF16 |
|------|------|-------|--------------------|
| [Qwen3.8-27B-ROCMFPX-MQ-Q4.gguf](./Qwen3.8-27B-ROCMFPX-MQ-Q4.gguf) | 15.7 GB | MagicQuant Q4 layout in ROCmFPX types (hybrid, fork-only) | 6.7611 (+0.25%) |
| [Qwen3.8-27B-ROCMFPX-MQ-Q6.gguf](./Qwen3.8-27B-ROCMFPX-MQ-Q6.gguf) | 22.2 GB | MagicQuant Q6 layout in ROCmFPX types (hybrid, fork-only) | 6.7579 (+0.20%) |
| [mmproj-Qwen3.8-27B-f16.gguf](./mmproj-Qwen3.8-27B-f16.gguf) | 0.9 GB | F16 (unquantized) | not measured |


Perplexity measured on wikitext-2 (100 chunks, ctx 512) against the BF16 baseline of **6.7443**. Lower is better; the percentage is the increase over BF16. These are the same measurements the tier selection is based on, so a tier that shipped is one that earned its size.

**Recommended: Q4 (14.64 GiB).** It is the smallest tier that is statistically tied with the best measured quality here. Q6 is 41% larger for 0.048 percentage points of perplexity, which is below what this measurement can resolve -- so the extra bytes buy nothing you can detect.

## Usage

Requires a from-source build of the [ROCmFPX fork](https://github.com/ciru-ai/ROCmFPX)
(stock llama.cpp, LM Studio, and Ollama cannot load these files):

```bash
# Interactive chat (--jinja uses the model's embedded chat template)
llama-cli -m Qwen3.8-27B-ROCMFPX-MQ-Q4.gguf -c 8192 --jinja -cnv

# Server mode
llama-server -m Qwen3.8-27B-ROCMFPX-MQ-Q4.gguf -c 8192 --port 8080 -ngl 99 -fa on --jinja
```

### Vision (image input)

```bash
llama-server -m Qwen3.8-27B-ROCMFPX-MQ-Q4.gguf --mmproj mmproj-Qwen3.8-27B-f16.gguf -c 8192 --port 8080 -ngl 99 -fa on
```

## Serving: MTP Speculative Decoding

This model includes **MTP ("nextn") draft tensors**, enabling self-speculative
decoding -- measured **~1.6-1.9x faster generation** with a ~95% first-token
accept rate (no separate draft model needed; it drafts from itself):

```bash
llama-server -m Qwen3.8-27B-ROCMFPX-MQ-Q4.gguf -c 8192 --port 8080 --host 127.0.0.1 -ngl 99 -md Qwen3.8-27B-ROCMFPX-MQ-Q4.gguf --spec-type draft-mtp -ctk q8_0 -ctv q8_0 -fa on
```

**Memory cost:** MTP needs its own draft context alongside the main context,
so serving with it uses roughly **2x the model's memory** compared to serving
without ``-md``/``--spec-type draft-mtp``.

## Caveats

- The base model's license (apache-2.0) applies to all derivative files
- **Fork-only files**: stock llama.cpp, LM Studio, and Ollama cannot load these -- build [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX) from source
- Quantization reduces precision -- verify outputs for your specific use case
- The hybrid quantization assigns different precision to different tensor groups, which means quality characteristics may differ from uniform quantizations

## Limitations

- Quantized models may exhibit subtle differences from the full-precision fine-tune
- This model inherits any limitations and biases present in the base model

---
*Generated with [MagicQuant](https://github.com/lucasmcoleman/MagicQuant)*
