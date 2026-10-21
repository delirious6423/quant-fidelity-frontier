---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - llama.cpp
  - qwen
  - qwen3.8
  - quantized
  - low-vram
  - no-mtp
  - vision-language
  - reasoning
  - agentic
quantized_by: TheWegemann
---

# Qwen3.8-27B LowGPU NoMTP — "IQ3XXXS" custom Mixed GGUF

## TL;DR

**Qwen3.8-27B in ~9.3 GB, built specifically around a 12 GB VRAM target.**

- Source weights: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- Format: **GGUF**
- MTP: **removed / NoMTP**, deliberately, to save VRAM
- Quantization: **custom per-tensor mixed precision**, derived from the ideas/tensor allocation in Unsloth's UD-IQ3_XXS release
- Importance matrix: **custom iMatrix**
- Real hardware test: **RTX 4070 12 GB**
- Tested with: **full GPU offload + 64K context + Q4_0 KV cache + Flash Attention + F16 mmproj**
- Real workload: **llama.cpp server + Hermes Agent**, not just a one-line prompt
- One quant only: **on purpose**. This repo is the tuned 12 GB build, not a quantization vending machine.

### What does `IQ3XXXS` mean?

`IQ3XXXS` is **our project name for this specific mixed-precision recipe**. It is **not an official llama.cpp quantization type** and there is no `IQ3_XXXS` enum hiding somewhere in the source tree.

The name is a deliberately cheeky nod to the Unsloth `UD-IQ3_XXS` allocation that inspired the starting point: we cut the overall footprint further for the 12 GB target, so `XXS` gained one more `X`.

Under the hood this is a **per-tensor mixed quant**, using real llama.cpp types such as `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `Q2_K`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`, `Q4_K`, `Q5_K` and `Q8_0`.

So, in short:

> **`IQ3XXXS` = nickname / recipe label. Not a llama.cpp quant type.**



## Why this exists

This is a **single-purpose low-VRAM build**, not a 25-file quant buffet.

The target was simple: make Qwen3.8-27B genuinely useful on a **12 GB GPU** with full GPU offload, while keeping enough VRAM headroom for a large context and vision projection. I built and iterated on the quantization together with **TheWegemann**; he supplied the hardware, workloads and repeated real-world testing, while I helped design, compare and refine the per-tensor strategy.

The starting reference for the tensor allocation behind our `IQ3XXXS` recipe was Unsloth's excellent [`Qwen3.8-27B-UD-IQ3_XXS.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-IQ3_XXS.gguf). This release is **not** simply that file renamed or requantized. The final GGUF is produced from the corresponding BF16 source model with our custom `IQ3XXXS` per-tensor mixed-precision recipe and a custom iMatrix.

The result is intentionally aggressive in early and middle parts of the network and much more protective where our testing showed that precision was worth the VRAM.

## What "LowGPU" means here

**12 GB VRAM is the target minimum for the GPU-only setup we care about.**

This release is not designed around CPU offload. If you have less than 12 GB VRAM, you may be able to make it run by reducing context, changing cache types or offloading to system RAM, but that is outside the target of this build.

Our actual test machine is:

- **GPU:** NVIDIA RTX 4070 12 GB
- **Model size:** approximately **9.3 GB**
- **GPU offload:** full (`-ngl all`)
- **Context tested:** **64,000 tokens**
- **KV cache:** `Q4_0` for K and V
- **Flash Attention:** enabled
- **Vision:** tested with a compatible **F16 mmproj**
- **Server:** `llama-server`
- **Agent frontend:** Hermes Agent through the OpenAI-compatible llama.cpp endpoint

Yes: the model, a 64K context setup and an F16 vision projector are all part of the same real 12 GB test setup. VRAM use still depends on llama.cpp version, driver, display usage and runtime settings, so physics remains annoyingly employed.

## Quantization design

Qwen3.8-27B has 64 language-model layers and uses a hybrid architecture with Gated DeltaNet / linear-attention blocks plus periodic full-attention blocks. The upstream model also has an MTP head. This release deliberately omits MTP to save VRAM.

The final recipe contains **498 per-tensor assignments**.

Key choices:

- `output.weight` → **Q4_K**
- `token_embd.weight` → **IQ4_XS**
- SSM `alpha` / `beta` tensors → **Q8_0**
- early layers are compressed very aggressively, including selected **IQ1_S / IQ1_M** tensors
- middle layers gradually receive more precision
- blocks **56–63** are deliberately protected and are dominated by **IQ3_S / IQ4_XS**, with selected **Q4_K / Q5_K**
- a **custom iMatrix** was used during quantization
- the allocation was inspired by Unsloth's UD-IQ3_XXS tensor strategy, then manually adapted around the 12 GB target

### The embedding rollback

One unusually expensive decision is intentional: `token_embd.weight` is **IQ4_XS**, not Q2_K.

In an earlier candidate, the embedding tensor was Q2_K. Across repeated German-language tests we observed recurring malformed words. We then changed **exactly one tensor** — the token embedding — from Q2_K to IQ4_XS while leaving the other 497 assignments unchanged. The language behavior improved noticeably in our qualitative A/B testing.

That single change costs roughly **258 MiB** compared with the previous candidate. On a 12 GB target that hurts, but it earned its place.

I am deliberately describing this as an observed A/B result, not as a universal theorem about token embeddings.

### Quant-type distribution

| Quant type | Tensor assignments |
|---|---:|
| Q8_0 | 96 |
| Q5_K | 2 |
| Q4_K | 10 |
| IQ4_XS | 48 |
| IQ3_S | 27 |
| IQ3_XXS | 67 |
| Q2_K | 112 |
| IQ2_S | 50 |
| IQ2_XS | 12 |
| IQ2_XXS | 27 |
| IQ1_M | 14 |
| IQ1_S | 33 |

> Counts are **per-tensor recipe assignments**, not a percentage of file size. Large tensors dominate storage, so do not read this table as a byte distribution.


## NoMTP: deliberate, not accidental

The official Qwen3.8-27B is trained with Multi-Token Prediction (MTP). This GGUF is a **NoMTP** build: the auxiliary MTP weights are omitted to reclaim VRAM for the main model, context and vision path.

That means native Qwen MTP speculative decoding is **not** part of this release. If native MTP is a priority, use an MTP-preserving build instead.

## Tested llama.cpp / Hermes configuration

This is a cleaned-up version of the settings used in our Windows + Hermes Agent setup. Paths are placeholders; the public example binds to localhost by default.

```bat
llama-server.exe ^
  -m "YOUR_MODEL.gguf" ^
  --mmproj "mmproj-F16.gguf" ^
  --mtmd-batch-max-tokens 512 ^
  -c 64000 ^
  -ngl all ^
  -ctk q4_0 ^
  -ctv q4_0 ^
  -b 512 ^
  -ub 512 ^
  -np 1 ^
  -fa on ^
  --image-min-tokens 256 ^
  --image-max-tokens 512 ^
  --n-predict -1 ^
  --reasoning on ^
  --chat-template-kwargs "{\"reasoning_effort\":\"low\"}" ^
  --cache-prompt ^
  --host 127.0.0.1 ^
  --port 9090
```

Hermes Agent can then use the llama.cpp OpenAI-compatible endpoint at:

```text
http://127.0.0.1:9090/v1
```

Vision projector included: mmproj-F16.gguf
Copied from unsloth/Qwen3.8-27B-GGUF for compatibility/convenience.
Tested with llama.cpp on the RTX 4070 12 GB setup.

The exact llama.cpp CLI changes over time. If a flag is rejected by your build, check that build's `--help` rather than sacrificing a keyboard to the dependency gods.

## What we actually tested

This is a practical release, not a benchmark-paper cosplay.

Our qualitative workload included:

- long German conversations
- reasoning / thinking mode
- coding and technical analysis
- multi-turn agent behavior
- tool-oriented / structured-output tasks
- long-context continuity
- vision through an F16 mmproj
- `llama-server` as the local OpenAI-compatible backend
- Hermes Agent as the real client workload

I am **not** publishing made-up tok/s numbers or claiming BF16-equivalent benchmark performance. The design goal was **fit + useful behavior on a 12 GB GPU**, and that is what we tested.

## Why only one quant?

Because this repository has a target.

I did not generate Q2 / Q3 / Q4 / Q5 / Q6 variants and upload the whole zoo. The point of this release is the **specific mixed allocation** that survived our 12 GB tuning process.

If you want a conventional range of excellent GGUF presets, use Unsloth's repository. If you want the weird little build we spent too much time squeezing into a 4070, that's this one.

## Upstream and credits

- **Qwen3.8-27B:** https://huggingface.co/Qwen/Qwen3.8-27B
- **Unsloth Qwen3.8 GGUFs / allocation reference:** https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
- **Exact UD-IQ3_XXS reference file:** https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-IQ3_XXS.gguf
- **llama.cpp:** https://github.com/ggml-org/llama.cpp
- **mmproj-F16.gguf** is provided by Unsloth and originates from unsloth/Qwen3.8-27B-GGUF. It is included here unchanged for easier vision-model discovery by compatible frontends.

Quantization work: **TheWegemann + Mira (ChatGPT)**.

This is a community project and is not an official Qwen, Unsloth or OpenAI release.

## License

Apache License 2.0, following the upstream model license. Please preserve all applicable upstream notices and license obligations.