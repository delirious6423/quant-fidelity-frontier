---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- janas
- jns
- local-inference
- cpu-inference
- mixture-of-experts
---

# gemma-4-26B-A4B-it for Janas-LLM

**On a laptop running Debian 13 Linux, with an Intel Core Ultra 9 185H, its integrated Arc GPU and 32 GB of memory, Janas-LLM reads a prompt at 84 tok/s and writes at 18 tok/s, 26 tok/s with its own assistant in a chat: llama.cpp, on the same laptop, 76 and 12.**

| On the same laptop | Janas-LLM | llama.cpp |
|---|---:|---:|
| reading a prompt of 1,024 tokens | 84 tok/s | 76 tok/s |
| writing (16 tokens) | 18 tok/s | 12 tok/s |
| writing in a chat, with its own assistant | 26 tok/s | not measured |

Measured on 9-11 October 2026: `llama-bench -p 1024 -n 16` and `janas-bench --prompt 1024 --gen 16`, both at their best configuration; the drafts with `janas-bench` on a chat reply, the mean of three rounds (a draft guesses the next tokens and the model checks them: the same text, faster). On other machines the figures will differ. Details: [Janas README](https://github.com/prabanta-dev/janas#readme).

## The machine these figures come from

- **Debian 13 (trixie) Linux**, x86-64, kernel 6.12.107
- Intel Core Ultra 9 185H: 6 performance cores, 8 efficiency cores, 2 low-power cores; AVX2, FMA, F16C, AVX-VNNI
- its integrated Intel Arc GPU, through the Vulkan driver of Mesa 26.1.6
- 32 GB of DDR5-5600 memory
- a 1 TB NVMe SSD, about 5.7 GB/s reading
- llama.cpp build `2b18470` (18 September 2026), Vulkan backend

These are the model's weights converted to **JNS**, the format of Janas-LLM, the inference engine in C of [Janas](https://github.com/prabanta-dev/janas), for ordinary computers, with or without a GPU. The files are read by Janas-LLM only.

## Get it

Janas-LLM for Linux x86-64 (Ubuntu 22.04, Debian 12 and later), then the model:

```sh
curl -LO https://github.com/prabanta-dev/janas/releases/latest/download/janas-linux-x86_64.tar.gz
tar xf janas-linux-x86_64.tar.gz && cd janas
./janas-get gemma-4-26b-a4b
./janas-chat gemma-4-26b-a4b
```

`janas-get` downloads the files of this repository and checks them against the fingerprints below; `janas-chat` with no model shows the catalog and downloads the one you choose. Building Janas-LLM from its sources instead: [prabanta-dev/janas](https://github.com/prabanta-dev/janas).

## Files

| File | Size | SHA-256 | Converted from |
|---|---:|---|---|
| `gemma-4-26b-a4b-it-udq4km.jns` | 18.30 GB | `35c3845bbc21de52bf630d3c7455b4bb5379c34487b5925e020ec2e573931666` | [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (catalog name `gemma-4-26b-a4b`: Gemma-4-26B-A4B-it, mixture of experts, with its assistant) |
| `gemma-4-26b-it-mtp.jns` | 0.46 GB | `6ba24ee91fe6fe904b29919acbb654a66d9caa0aba67db1df9f921a76e5bcedf` | [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) `mtp-gemma-4-26B-A4B-it.gguf` |

## What was changed

These files are a modified work of the original model, as the Apache License 2.0 requires to be stated. Nothing was retrained or pruned: each file is a GGUF file quantized by the people named above, rewritten by Janas-LLM's `gguf2jns` in the order Janas reads it, with the metadata carried over. Each (layer, expert) has a slot of its own, so that the experts a token needs can be read from disk on their own. The assistant (a small model that drafts the next tokens) was converted from its own GGUF in the same way.

## Licence and attribution

Licensed under the **Apache License 2.0**, as the original model: see `LICENSE`.

- Model: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), by Google.
- Quantized GGUF: [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF), by its authors.
- Converted to JNS by the Janas project with Janas-LLM ([prabanta-dev/janas](https://github.com/prabanta-dev/janas), GPL-3.0-or-later; the licence of the engine does not apply to these files).

The original repository carries no LICENSE file; its card declares Apache 2.0. `LICENSE` here is the licence's text as the Apache Software Foundation publishes it.
