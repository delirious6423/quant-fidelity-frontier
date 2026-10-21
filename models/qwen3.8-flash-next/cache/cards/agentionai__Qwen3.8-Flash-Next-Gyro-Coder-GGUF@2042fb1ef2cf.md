---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- qwen3.8-flash-next
- qwen4exp
- code
- experimental
- agentionai
---

<img src="gyro-coder-header.png" alt="AgentionAI — Gyro Coder 3.8 Flash Next" width="100%">

# Qwen3.8-Flash-Next Gyro-Coder — GGUF

**A 125B-parameter coding model on one 24 GB card.**

Qwen3.8-Flash-Next is a 125B mixture-of-experts model, the kind that normally needs a workstation full of GPUs.
Gyro-Coder brings it to a single RTX 3090, 4090 or 7900 XTX: 19.9 GiB of GPU memory.

Two things make that possible:
- **Only the experts that code.** Each layer keeps the 256 of its 512 experts that matter for programming, picked by
  an evolutionary search over a space of 10⁷³²⁸ possible combinations (see below). That leaves 65B parameters on the GPU.
- **2.125 bits per weight, without the usual damage.** The experts are stored with our own trellis encoder, the
  same one that puts Gyro-M ahead of IQ3_XXS and UD-Q2_K_XL quants that need 20-25% more memory.

On held-out Python it picks the same next token as the full model 93% of the time. Give it clear instructions
(implement this, fix that, refactor, explain) and it delivers. For open-ended work and general knowledge,
[Gyro-S](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) or Gyro-M is the better pick.

Runs in our llama.cpp build (Docker, see Quick start). [Strata](https://github.com/Niko1221/Strata) support is coming.

| Headline | Detail |
|---|---|
| **19.92 GiB of GPU weights** | 21.1 GiB with 64k context (q8_0 KV): fits a 24 GB card |
| **93.1% top-1 agreement on Python** | KLD 0.165 |
| **13/16 coding tasks passed** | auto-graded, 8 tasks x 2 seeds; the same experts at 8 bits pass 15/16 |

## Pick your file

| Your hardware | File | GPU memory (weights) |
|---|---|---:|
| 24 GB card (RTX 3090 / 4090, 7900 XTX) | **`Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf`** | 19.92 GiB |
| 32 GB, general use as well as code | [Gyro-S](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) | 27.64 GiB |
| 48 GB+, general use as well as code | [Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) | 34.98 GiB |

The `TQ2_0` in the name is only a size class for the Hub: **stock llama.cpp cannot load this file.**

## A needle in a very large haystack

Each layer keeps 256 of its 512 experts. That is about 10¹⁵² possible choices for one layer, and about **10⁷³²⁸**
for the whole model: written out, the number fills about four pages. If you had tested a billion candidate models
every second since the Big Bang, you would have covered about 10⁻⁷³⁰¹ of them.

The layers don't act alone either: what one layer keeps changes what every later layer sees, and each candidate is
a full model that has to be built and tested before you know whether it is any good.

We started by mapping which experts carry which domains. From there, an evolutionary search proposed and scored
about 2,200 candidate expert sets across 14 runs. The best were built as full models and put through coding, loop
and knowledge tests. This is the one that passed.

## Quality

KL divergence and top-1 agreement against the full model (Q8_0, all 512 experts), `-c 2048`.

| Set | KLD ↓ | top-1 ↑ |
|---|---:|---:|
| Python (held-out source files) | 0.165 | 93.1% |
| shell | 0.417 | 80.9% |
| knowledge (descriptions of everyday things) | 0.301 | 80.8% |
| held-out (2026 technical writing and code) | 0.579 | 70.8% |

Half the experts are gone, so general text and knowledge lose the most (Gyro-M: held-out 0.291); Python stays close
to the full model.

**Coding tests:** 13 of 16 passed (8 auto-graded tasks: LRU cache, interval merging, bug fix from a trace, refactor,
bash script, SQL, CSV parsing in JS, multi-file change; 2 seeds each, thinking on, 64k tokens). The same expert set at
8 bits passes 15 of 16.

## Speed

Strix Halo (Vulkan, MTP draft): 11–22 tok/s per stream with two streams running at once (measured while
generating long answers).

## Quick start

**1. Download Gyro-Coder** (70.6 GiB on disk; put it on a fast NVMe drive):

```bash
pip install -U huggingface_hub
hf download agentionai/Qwen3.8-Flash-Next-Gyro-Coder-GGUF Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf --local-dir ~/models
```

**2. Run it** with our [llama.cpp build](https://gitlab.com/agentionai-group/llama.cpp) (Vulkan, in Docker):

```bash
# AMD (and Intel) GPUs
docker run --rm -it --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
  -v ~/models:/models -p 8080:8080 agentionai/agention-llama:server \
  -m /models/Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf -ngl 999 -fa on --jinja --ngram-on-disk \
  -c 65536 -ctk q8_0 -ctv q8_0 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05
```

**NVIDIA:** replace `--device /dev/dri --group-add ...` with `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all`.

Web chat at http://127.0.0.1:8080, OpenAI-compatible API at `http://127.0.0.1:8080/v1`: point your coding agent or
editor plugin there.

**Strata:** support is coming. Strata 0.2.2 does not know this file yet (Gyro-S and Gyro-M already run there); we will
update this card when it does.

## Recommended settings

- Sampling: `--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.05` (the settings all our tests used).
- Keep `--ngram-on-disk`, with the model on fast NVMe: the 50.7 GiB n-gram table stays on disk, which is why the file
  (70.6 GiB) is much larger than its GPU footprint.
- Give it clear instructions; it does best on well-specified tasks.

## Status

Experimental: the format and kernels may still change, and the files will be re-published when they do.
Tensor format: trellis-coded quantization with Hadamard rotation, in the family of QTIP (Tseng et al., 2024).
Encoder, trellis types and kernels: AgentionAI.

Made by AgentionAI.
