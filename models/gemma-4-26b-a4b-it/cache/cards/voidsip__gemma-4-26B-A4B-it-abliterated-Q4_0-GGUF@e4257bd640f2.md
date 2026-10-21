---
license: gemma
base_model:
  - huihui-ai/Huihui-gemma-4-26B-A4B-it-qat-q4_0-unquantized-abliterated-GGUF
  - google/gemma-4-26B-A4B-it
tags:
  - gguf
  - q4_0
  - moe
  - abliterated
  - uncensored
  - gemma4
  - llama.cpp
  - freetoken
pipeline_tag: text-generation
---

# Gemma-4-26B-A4B-it Abliterated — Q4_0 GGUF

A **Q4_0** GGUF requantization of
[huihui-ai/Huihui-gemma-4-26B-A4B-it-qat-q4_0-unquantized-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-gemma-4-26B-A4B-it-qat-q4_0-unquantized-abliterated-GGUF).

I did not train, abliterate, or otherwise modify the weights' behaviour — this is
**only a format conversion**. All credit for the base model goes to Google, and all
credit for the abliteration goes to huihui-ai.

## Why Q4_0 specifically

Every existing abliterated Gemma-4-26B-A4B release ships as either **Q4_K** GGUF or
**compressed-tensors NVFP4**. Neither loads in runtimes that want plain Q4_0, and there
was no abliterated Q4_0 build of this model. This fills that gap.

Two reasons Q4_0 is the right target for *these* weights:

1. The base is Google's **QAT** (quantization-aware trained) line, tuned so that
   quantizing to `q4_0` recovers close to bf16 quality. Q4_0 is the format the QAT
   process was designed around, so it is a better-matched target here than it would be
   for a normal post-training quantization.
2. **FreeToken** only accepts two expert formats for Gemma-4 — ModelOpt NVFP4 or
   `q4_0`. Its expert-bank loader has no `compressed-tensors` provider, so the existing
   NVFP4 abliterated builds fail to load their experts (silently dropping all 46,110
   expert tensors on checkpoint conversion, or raising
   `no expert-bank provider for expert_quant='compressed-tensors'`).

## Details

| | |
|---|---|
| Architecture | Gemma-4 MoE, 25.2B total / ~3.8B active |
| Experts | 128, top-8 routing |
| Quantization | Q4_0 |
| Size | 13,755 MiB (4.57 BPW), from 48,150 MiB bf16 |
| Tensors | 658, all converted |
| Produced with | `llama-quantize` (llama.cpp b10731) |

Conversion was a straight `llama-quantize <bf16.gguf> <out.gguf> Q4_0` with no imatrix
and no per-tensor overrides.

## Measured performance

On a single **RTX 5090** (32 GB, driver 610.88), llama.cpp CUDA build, all layers
offloaded (`-ngl 99`):

- **~219 tok/s** decode
- **~213 tok/s** prompt eval
- 13.4 GiB weights, leaving substantial room for KV cache at 64K context

Because only ~3.8B of the 26B parameters activate per token, decode reads a small
fraction of the weights — this is several times faster than a dense ~27B model of
comparable quality on the same card.

## Usage

**llama.cpp**

```bash
llama-server -m Huihui-gemma-4-26B-A4B-abliterated-Q4_0.gguf \
  -a gemma4-26b-a4b-abliterated \
  --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --parallel 1
```

Tool calling works and returns well-formed `tool_calls`. The model is a reasoning model
and emits `reasoning_content` alongside `content`; give it enough `max_tokens` that it
can finish thinking before it starts answering.

**FreeToken**

FreeToken loads this natively (`Gemma4GGUFForCausalLM`), but on Windows its GGUF
dequantize kernel is JIT-compiled at startup and therefore needs `ninja`, `nvcc`
(`CUDA_HOME` set), and MSVC `cl.exe` present. Without those it fails during CUDA graph
capture with `Ninja is required to load C++ extensions` or
`CUDA_HOME environment variable is not set`.

## License and use restrictions

This is a derivative of Google's Gemma and remains under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). By using it you agree to those
terms, **including the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)**,
which continues to apply to this model and to anything derived from it.

Be aware of what this model is: refusal behaviour has been removed at the weight level
by the upstream abliteration. It will not decline requests that the original model would
have declined. That does not widen what you are permitted to do with it — the Prohibited
Use Policy still governs, and the absence of a refusal is not permission. You are
responsible for how you use it, and for passing these terms on if you redistribute it.

## Attribution

- Base model: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B) (Gemma Terms)
- Abliteration: [huihui-ai](https://huggingface.co/huihui-ai) — norm-preserving biprojected abliteration with Expert-Granular Abliteration (EGA) for MoE expert weights
- This repo: Q4_0 requantization only
