---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- janas
- jns
- local-inference
- cpu-inference
- mixture-of-experts
---

# Qwen3.8-Flash-Next for Janas-LLM

**On a laptop running Debian 13 Linux, with an Intel Core Ultra 9 185H and 32 GB of memory, Janas-LLM runs this 177-billion-parameter model (66.5 GB, larger than the memory) on the CPU alone: it reads a prompt at 24.0 tok/s and writes at 10.95 tok/s. llama.cpp, on the same laptop and the same file: 4.82 and 3.12.**

| On the same laptop, CPU only, 12 threads | Janas-LLM | llama.cpp |
|---|---:|---:|
| reading a prompt of 1,024 tokens | 24.0 tok/s | 4.82 tok/s |
| writing 200 tokens after it | 10.95 tok/s | 3.12 tok/s |

Measured on 23 October 2026 with the same 1,024 tokens of Italian prose and 200 tokens written greedily: Janas-LLM's `bench_long` (the engine of 23 October 2026, a 12 GiB cache of experts read with `O_DIRECT`, the mean of two runs) and llama.cpp `2b18470` through its public API (built with `GGML_CPU_REPACK=OFF`, the weights mapped from the file, one run), on CPUs 0-11. llama.cpp's Vulkan backend could not run the file on this laptop: it copies 35.9 GB of weights into pinned host buffers, and with `no_host` reads 63 GB into anonymous memory. On other machines the figures will differ.

Also measured on the same laptop:

- **With its integrated Arc GPU** Janas-LLM reads the prompt at 32.7 tok/s, the whole of each block on the GPU (the engine of earlier that day, before the CPU's mixes got faster), at a peak of 16 GiB of memory.
- **In a chat**, with the model's own multi-token prediction block (in this repository), a reply is 23% faster greedy (8.26 to 10.12 tok/s) and 39% faster at the chat's temperature (8.59 to 11.96), the text the same with drafts and without.
- **`JANAS_HC_MIX=q8`**, the mixes of the residual streams rewritten at eight bits when the model opens, writes at 11.9 tok/s.

Against llama.cpp on the same file, token by token, the most likely next token agrees at 390 of 400 positions of C code and 367 of 400 of Italian prose.

## The machine these figures come from

- **Debian 13 (trixie) Linux**, x86-64, kernel 6.12.111
- Intel Core Ultra 9 185H: 6 performance cores, 8 efficiency cores, 2 low-power cores; AVX2, FMA, F16C, AVX-VNNI
- its integrated Intel Arc GPU, through the Vulkan driver of Mesa 26.1.6
- 32 GB of DDR5-5600 memory
- a 1 TB NVMe SSD, about 5.7 GB/s reading

A mixture of 512 experts, ten of them for each token: Janas-LLM streams from the NVMe SSD the experts each token needs and keeps in memory the ones that keep coming back (96% of them from memory while writing, here). The model's residual runs in four streams, its n-gram embeddings are read from the file a row at a time, and past 2,051 positions its attention is sparse.

These are the model's weights converted to **JNS**, the format of Janas-LLM, the inference engine in C of [Janas](https://github.com/prabanta-dev/janas), for ordinary computers, with or without a GPU. The files are read by Janas-LLM only.

## Get it

Janas-LLM for Linux x86-64 (Ubuntu 22.04, Debian 12 and later), then the model:

```sh
curl -LO https://github.com/prabanta-dev/janas/releases/latest/download/janas-linux-x86_64.tar.gz
tar xf janas-linux-x86_64.tar.gz && cd janas
./janas-get qwen3.8-flash-next
./janas-chat qwen3.8-flash-next
```

`janas-get` downloads the files of this repository and checks them against the fingerprints below; `janas-get qwen3.8-flash-next --from-gguf` downloads ISTA-DASLab's GGUF instead and converts it on your machine, to the same fingerprints. Building Janas-LLM from its sources: [prabanta-dev/janas](https://github.com/prabanta-dev/janas).

## Files

| File | Size | SHA-256 | Converted from |
|---|---:|---|---|
| `qwen3.8-flash-next-gsq-q20.jns` | 66.48 GB | `b5b4dbe4e3ae01bffc41acefa0d1e62afdfe4b5b9bc24c2a3fab591a41f9087f` | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) at `ed59f920`, `Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf` and `-00002-of-00002.gguf` (catalog name `qwen3.8-flash-next`) |
| `qwen3.8-flash-next-mtp-q4km.jns` | 1.91 GB | `d9c9dd94ef2e52391613db98c2e34c574341c48ab8752454696534d0f69a29ca` | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) at `766911a6`, `MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` |

## What was changed

Nothing was retrained or pruned. The model file is ISTA-DASLab's GGUF, quantized by them with GSQ-RCO to two bits (Q2_0), rewritten by Janas-LLM's `gguf2jns` in the order Janas reads it, with the metadata carried over; each (layer, expert) has a slot of its own, so that the experts a token needs can be read from disk on their own. The multi-token prediction file is the block of unsloth's GGUF, copied as it is, in its own types, by Janas-LLM's `hf2jns_mtp`, without that file's token embeddings and output head (the engine takes the model's own). Both were made with the Janas-LLM of 22 October 2026; two runs of `hf2jns_mtp` give the same bytes.

## Licence and attribution

Licensed under the **Qwen Community License 1.0**, as the original model: see `LICENSE`, the original repository's own. In short, and the licence is what counts: use, copies, changes and redistribution are allowed with its notice; a product or service above 100 million monthly active users or US$ 20 million of monthly revenue must show the model's name; a business offering models as a service or an AI work assistant needs a separate licence from Qwen before using it for any commercial purpose.

- Model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) at `de4b8e4d`, by Alibaba Cloud (the Qwen team).
- Quantized GGUF: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), by its authors; its card says Apache 2.0, but the model it quantizes is under the Qwen Community License, which these files follow.
- Multi-token prediction block: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), by its authors.
- Converted to JNS by the Janas project with Janas-LLM ([prabanta-dev/janas](https://github.com/prabanta-dev/janas), GPL-3.0-or-later; the licence of the engine does not apply to these files).
