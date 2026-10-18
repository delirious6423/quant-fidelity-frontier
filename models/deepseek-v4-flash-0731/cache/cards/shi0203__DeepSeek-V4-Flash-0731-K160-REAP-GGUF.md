---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - deepseek
  - deepseek-v4
  - gguf
  - reap
  - quantized
  - mixture-of-experts
  - text-generation
language:
  - en
---

# DeepSeek V4 Flash 0731 — REAP K160 GGUF

> **Experimental model**
>
> This is an unofficial REAP K160 compact variant of DeepSeek V4 Flash 0731.
> Quality evaluation is still in progress, and some capabilities are significantly degraded compared with the full model.

## What is this?

This repository contains a **REAP K160 / 160-expert compact variant of DeepSeek V4 Flash 0731**, created by pruning routed experts directly from an already-working quantized GGUF.

The current Q2 variant starts from the asymmetric / mixed Q2 quantization published by [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf), and removes the routed experts excluded by the K160 REAP expert selection from [0xSero/DeepSeek-V4-Flash-0731-REAP](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP).

**The retained expert weights are not dequantized and requantized.**

In simplified form:

    DeepSeek-V4-Flash-0731
            │
            ├── antirez Q2 GGUF
            │      │
            │      └── keep already-quantized weights
            │
            └── 0xSero REAP K160 expert selection
                   │
                   └── remove pruned experts
                            │
                            ▼
                This K160 compact Q2 GGUF

The goal is to reduce memory usage while preserving as much of the behavior of the known-working Q2 quantization as possible.

## Why does this exist?

There are already quantized REAP K160 variants available.

However, this model takes a different conversion path.

Instead of starting from an already-pruned K160 checkpoint and quantizing its remaining weights again, this variant starts from an existing working Q2 GGUF and **only removes the experts excluded by the K160 pruning map**.

This avoids an additional:

    quantized / mixed-precision weights
            ↓
       dequantization
            ↓
      requantization to Q2

step for the retained expert weights.

The practical motivation is simple: I want a K160 model small enough to run comfortably on memory-constrained local hardware while remaining useful for real agentic workloads.

## Current status

The model is still under evaluation.

### Confirmed

- Loads and runs with REAP-compatible DS4 forks.
- Tested with the ROCm REAP runtime on AMD Strix Halo.
- Produces coherent English output.
- Has successfully completed a real **agentic task involving tool use** in my testing.
- Uses only 160 routed experts per MoE layer instead of the original 256.

### Not yet fully evaluated

- General benchmark quality
- Coding benchmark performance
- Agentic task success rate across a larger test set
- Long-context reliability
- Comparison against the full Q2 model
- Comparison against other K160 quantizations
- ROCmFP2 variants

More results will be added as testing progresses.

## Important limitation: Japanese

**Japanese capability is effectively unusable in my testing after REAP K160 pruning.**

Normal Japanese conversation cannot be reliably sustained.

This is not a small quality regression; for practical purposes, **do not use this model for Japanese-language workloads**.

English agentic use is the primary target of my current testing.

## Q2 quantization

The current Q2 variant is derived from the imatrix quantization in:

[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)

The original quantization uses an asymmetric mixed-precision layout approximately structured as follows:

| Tensor group | Quantization |
|---|---|
| Routed expert gate/up | `IQ2_XXS` |
| Routed expert down | `Q2_K` |
| Attention projections | `Q8_0` |
| Shared experts | `Q8_0` |
| Output | `Q8_0` |
| Other tensors | Mixed higher precision |

This repository preserves the already-quantized tensors of the retained experts rather than requantizing them.

## Files

### Current Q2 variant

    DeepSeek-V4-Flash-0731-K160-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf

REAP pruning metadata / plan:

    DeepSeek-V4-Flash-0731-K160-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf.reap_plan.json

Additional quantization variants may be added to this repository later.

In particular, a **ROCmFP2** variant is under consideration.

## Inference

This model uses a **compact REAP K160 layout**.

The routed expert tensors physically contain fewer expert slots than the original DeepSeek V4 Flash model, so a runtime with explicit compact REAP support is required.

### Recommended runtimes

| Backend | Runtime |
|---|---|
| Metal / CUDA / CPU | [eouya2/ds4-for-reaped](https://github.com/eouya2/ds4-for-reaped) |
| ROCm | [shin902/ds4-rocm](https://github.com/shin902/ds4-rocm) |

### Metal / CUDA / CPU

Use:

[https://github.com/eouya2/ds4-for-reaped](https://github.com/eouya2/ds4-for-reaped)

This DS4 fork adds support for compact REAP GGUF layouts, including routed-expert tensors whose physical expert dimension has been reduced.

### ROCm

Use:

[https://github.com/shin902/ds4-rocm](https://github.com/shin902/ds4-rocm)

This is my DS4 fork with ROCm support for the compact REAP model layout.

It is the runtime used for my current testing on AMD Strix Halo.

### Upstream DS4

The original inference engine is:

[antirez/ds4](https://github.com/antirez/ds4)

However, upstream DS4 currently does not directly load this compact K160 GGUF.

The original model stores 256 routed experts, while this model physically stores only 160. Attempting to load this compact model with an incompatible upstream build results in a tensor shape validation error such as:

    ds4: tensor blk.0.ffn_gate_inp.weight has dim[1]=160, expected 256

Use one of the REAP-compatible forks listed above instead.

## Hardware target

My primary test system is:

    AMD Ryzen AI MAX+ 395 (Strix Halo)
    96 GB unified memory
    ROCm
    Linux

The broader goal is to make DeepSeek V4 Flash practically usable on hardware where the full-size model leaves too little memory headroom for context, KV cache, agent workloads, and other system processes.

## Planned evaluation

I plan to compare:

    Full asymmetric Q2
            vs
    This REAP K160 Q2
            vs
    Other K160 quantizations

Across several dimensions:

- Model size
- Peak memory usage
- Prefill speed
- Decode speed
- General quality benchmarks
- Coding capability
- Tool-call reliability
- Agentic task completion
- Repetition / generation failures
- Long-context behavior

The purpose is not only to ask whether the model "runs", but whether the resulting compressed model remains **practically usable**.

## Model provenance

### Base model

[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

### Q2 quantization source

[antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)

### REAP K160 expert selection

[0xSero/DeepSeek-V4-Flash-0731-REAP](https://huggingface.co/0xSero/DeepSeek-V4-Flash-0731-REAP)

The K160 expert selection is used to determine which routed experts are retained.

The actual retained expert weights in this Q2 release come from the already-quantized antirez Q2 GGUF rather than from quantizing the 0xSero K160 checkpoint.

## Inference runtime provenance

### Original engine

- [antirez/ds4](https://github.com/antirez/ds4)

### REAP-compatible runtimes

- Metal / CUDA / CPU: [eouya2/ds4-for-reaped](https://github.com/eouya2/ds4-for-reaped)
- ROCm: [shin902/ds4-rocm](https://github.com/shin902/ds4-rocm)

## License

MIT.

DeepSeek V4 Flash 0731 is released under the MIT License.

This repository is an unofficial derivative built using resources from the projects referenced above. See each upstream repository for its original license notices and attribution information.

## Disclaimer

This is an **unofficial experimental derivative**.

It is not an official release from DeepSeek or the upstream projects referenced above.

Compatibility information, benchmark results, model quality, and recommended runtimes may change as the model and inference software continue to evolve.