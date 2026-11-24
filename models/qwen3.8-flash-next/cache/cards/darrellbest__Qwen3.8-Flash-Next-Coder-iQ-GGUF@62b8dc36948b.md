---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
tags:
- gguf
- qwen4exp
- imatrix
- coding
- agent
---

# Qwen3.8-Flash-Next-Coder-iQ-GGUF

A coding- and agent-calibrated GGUF quant of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). It keeps all 512 experts per layer and ships the MTP head.

## `v1/`: preview

> **Preview.** The importance matrix behind this build was collected on our coding-agent corpus, using only its
> first 220 of 367 chunks. That corpus covers:
>
> - KannaCode agent benchmark transcripts
> - SWE-bench Lite issues and patches
> - Magma source code
> - Hermes function calling
> - MATH-500 and GSM8K
> - a little general chat
>
> **The final v2 build replaces it.** v2's importance matrix adds real Claude Code agent sessions to the agent mix
> (60/40 with the KannaCode data).

| Tensors | Type |
|---|---|
| Routed experts gate/up (512 × 48 layers) | **Q4_K**, imatrix |
| Routed experts down | **IQ4_NL**, imatrix. The 640-wide rows rule out K-quants |
| Attention, GDN, hyper-connections, indexer, shared experts, router, embeddings, lm_head | Q8_0 (unchanged from the Q8_0 reference) |
| N-gram (PLE) table | Q8_0 (unchanged) |
| Norms, convs, SSM | F32 |

- **Size:** 131 GB (5.78 bpw over all weights). About 54 GB of that is the n-gram table, a CPU-side lookup that
  never needs to be resident.
- **Source:** quantized with `llama-quantize --imatrix` from the Q8_0 build in
  [darrellbest/Qwen3.8-Flash-Next-Base-GGUF](https://huggingface.co/darrellbest/Qwen3.8-Flash-Next-Base-GGUF),
  which is Unsloth's Q8_0. There was no room for a 360 GB BF16 GGUF.
- **MTP head:** `v1/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` is Unsloth's Q8_0 MTP head, untuned.
- **Vision:** `mmproj-F16.gguf` at the repo root is the vision projector.
- **Needs** a llama.cpp with the `qwen4exp` architecture.

Work in progress: the v2 files, KL / top-1 results against the Q8_0 + BF16-engram reference, and the MLX / W4A16
twins are still to come.
