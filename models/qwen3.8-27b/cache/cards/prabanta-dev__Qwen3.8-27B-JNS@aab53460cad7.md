---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- janas
- jns
- local-inference
- cpu-inference
---

# Qwen3.8-27B for Janas-LLM

**On a laptop running Debian 13 Linux, with an Intel Core Ultra 9 185H, its integrated Arc GPU and 32 GB of memory, Janas-LLM reads a prompt at 26.4 tok/s and writes at 4.4 tok/s: llama.cpp, on the same laptop, 25.6 and 2.1.**

| On the same laptop | Janas-LLM | llama.cpp |
|---|---:|---:|
| reading a prompt of 1,024 tokens | 26.4 tok/s | 25.6 tok/s |
| writing (16 tokens) | 4.4 tok/s | 2.1 tok/s |

Measured on 12 October 2026: `llama-bench -p 1024 -n 16` and `janas-bench --prompt 1024 --gen 16`, both at their best configuration, one round each. On other machines the figures will differ. Details: [Janas README](https://github.com/prabanta-dev/janas#readme).

## The machine these figures come from

- **Debian 13 (trixie) Linux**, x86-64, kernel 6.12.111
- Intel Core Ultra 9 185H: 6 performance cores, 8 efficiency cores, 2 low-power cores; AVX2, FMA, F16C, AVX-VNNI
- its integrated Intel Arc GPU, through the Vulkan driver of Mesa 26.1.6
- 32 GB of DDR5-5600 memory
- a 1 TB NVMe SSD, about 5.7 GB/s reading
- llama.cpp build `2b18470` (18 September 2026), Vulkan backend

A dense model: every weight is read for every token, so it wants the memory to hold all of its 17.6 GB. The file is unsloth's UD mix, whose IQ4_XS, IQ3_S and IQ4_NL weights reach the GPU's fast prompt from Janas-LLM of 12 October 2026 on (before, 18.7 tok/s reading). Its multi-token prediction block is inside the file: `janas-chat` drafts with it by itself.

These are the model's weights converted to **JNS**, the format of Janas-LLM, the inference engine in C of [Janas](https://github.com/prabanta-dev/janas), for ordinary computers, with or without a GPU. The files are read by Janas-LLM only.

## Get it

Janas-LLM for Linux x86-64 (Ubuntu 22.04, Debian 12 and later), then the model:

```sh
curl -LO https://github.com/prabanta-dev/janas/releases/latest/download/janas-linux-x86_64.tar.gz
tar xf janas-linux-x86_64.tar.gz && cd janas
./janas-get qwen3.8-27b
./janas-chat qwen3.8-27b
```

`janas-get` downloads the files of this repository and checks them against the fingerprints below; `janas-chat` with no model shows the catalog and downloads the one you choose. Building Janas-LLM from its sources instead: [prabanta-dev/janas](https://github.com/prabanta-dev/janas).

## Files

| File | Size | SHA-256 | Converted from |
|---|---:|---|---|
| `qwen3.8-27b-ud-q4kxl.jns` | 17.56 GB | `138d1db3af7e48d3bd36a428f7a81dfd8813434521aced84459c546e4830ab5b` | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `Qwen3.8-27B-UD-Q4_K_XL.gguf` (catalog name `qwen3.8-27b`: Qwen3.8-27B, dense, unsloth's UD mix, with its MTP block) |

## What was changed

These files are a modified work of the original model, as the Apache License 2.0 requires to be stated. Nothing was retrained or pruned: each file is a GGUF file quantized by the people named above, rewritten by Janas-LLM's `gguf2jns` in the order Janas reads it, with the metadata carried over.

## Licence and attribution

Licensed under the **Apache License 2.0**, as the original model: see `LICENSE`.

- Model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), by Alibaba Cloud (the Qwen team).
- Quantized GGUF: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), by its authors.
- Converted to JNS by the Janas project with Janas-LLM ([prabanta-dev/janas](https://github.com/prabanta-dev/janas), GPL-3.0-or-later; the licence of the engine does not apply to these files).

`LICENSE` is the original repository's own.
