---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- quantization
- experimental
- hadamard
---

# Qwen3.8-Flash-Next Q2_0 H64-S4 GGUF

I wanted a smaller GGUF I could run, and I wanted to know more than just whether it loaded. This is where that experiment landed. [How we got here](JOURNEY.md) has the story, including the first Q2 export, the H64-S4 experiment, the interrupted run, and what I could and couldn't verify.

**You need a runtime that supports the matching Hadamard transform.** A runtime that ignores the rotation metadata can compute the wrong result. The tested llama.cpp changes, source patch, and build instructions are in [`runtime/`](runtime/README.md).

## What's in the files

The export has nine GGUF shards, about **68.9 GB** in total. It is based on `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`.

I used mixed precision to spend the bits where the recipe calls for them:

- The 144 routed expert gate, up, and down matrices use Q2_0 ternary weights, fitted in groups of 64 with FP16 scales.
- Before fitting those expert weights, the exporter applies normalized Walsh-Hadamard transforms in blocks of 64. A fixed sign pattern from seed 4 makes the transform reproducible. The runtime applies the matching input transform.
- The output head, attention weights, and shared experts use Q8_0 where selected by the recipe. The per-layer n-gram table uses Q4_0. Protected control tensors retain their source types.
- The MTP head and vision projector are not included. The inference checks used MTP disabled.

This is post-training quantization. There was no recovery training, and this is not a trained Bonsai release.

## How the test went

I tested the full model on three RTX 3090 GPUs, with the per-layer token table on CPU and a 65,536-token context allocation. The context setting shows what I configured; it is not a claim that answer quality was checked across the full context window.

In a small diagnostic run, H64-S4 passed **21 of 27** prompts with thinking off. All five generated code samples passed isolated execution checks. JSON formatting, Spanish, and both context-recall prompts also passed. Six short arithmetic and logic answers missed. When I retried those same six with thinking on, **all six passed**.

These prompts are a useful smoke check, not a broad model benchmark. The original full Q2 file had passed its own load and prompt checks, but I could not find it later to run the same prompts side by side. So I can't claim H64-S4 gives better answers than that Q2 export.

A separate weight-sample sweep found 2.91% lower normalized reconstruction error for H64-S4 than direct Q2 on the 26 samples used to choose the setting. That result helped me decide to try the full export; it does not prove a general accuracy gain.

See the [original prompts and responses](evaluation/H64_VALIDATION_20261001.json), the [thinking-mode follow-up](evaluation/H64_THINKING_FOLLOWUP_20261001.json), and the [tensor checks](evaluation/byte-verification.txt).

## Download and run

Download all nine shards into the same folder and check them with `sha256sum -c SHA256SUMS`. Start the matching runtime with shard `00001`; it will read the other shards beside it. The [runtime guide](runtime/README.md) has the tested build and launch settings.

The Qwen Community License 1.0 is included as [`LICENSE`](LICENSE). The runtime patch carries the llama.cpp project license in [`runtime/LICENSE`](runtime/LICENSE).
