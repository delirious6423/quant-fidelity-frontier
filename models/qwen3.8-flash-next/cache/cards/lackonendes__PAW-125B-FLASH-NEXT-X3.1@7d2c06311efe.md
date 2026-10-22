---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
library_name: gguf
tags:
- quantization
- trellis-codec
- moe
- paw
- gguf
---

# PAW-125B-FLASH-NEXT-X3.1

_If this model is useful to you, support the work by following [@nawariokra](https://x.com/nawariokra) on X, or buy me a coffee on [Ko-fi](https://ko-fi.com/lackonendes)._

<a href="https://ko-fi.com/F4P3296XP7"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="32"></a>


**`Qwen/Qwen3.8-Flash-Next` (125B MoE, 6B active) on one or two 24 GB GPUs plus CPU, with the full 262,144-token context.**
Two GGUF shards: **47.5 GiB of weights** and a 48.3 GiB n-gram table (95.8 GiB in all). It runs on our own engine [TRUSS](https://github.com/irawanw/truss).

| | |
|---|---|
| model | Qwen3.8-Flash-Next: 48 layers (36 Gated DeltaNet + 12 sparse-attention), 512 experts per layer, 10 routed + 1 shared per token |
| experts | 24,576 routed experts: the 12,000 most used at **3.5 bit**, the other 12,576 at **2.5 bit** (about 3.0 bit on average) |
| everything else | Q8_0 (attention, DeltaNet, shared experts, embeddings, output) |
| context | 262,144 tokens |
| decode at 150K context | **115-128 tok/s on two RTX 3090s**; 57-60 tok/s on one RTX 3090 (peak 80+ on short prompts) |
| prefill, 150K-token prompt | **about 4,500 tok/s on two RTX 3090s**; about 2,200 tok/s on one |
| files | `flashnext-x31-00001-of-00002.gguf` 47.5 GiB (weights) + `flashnext-x31-00002-of-00002.gguf` 48.3 GiB (n-gram table) + `flashnext-mtp-x3k3.gguf` 0.97 GiB + `flashnext-mmproj-f16.gguf` 0.84 GiB (vision encoder, optional) |

**Stock llama.cpp cannot read this file.** It needs [TRUSS](https://github.com/irawanw/truss). See
[Requirements](#requirements) and [SERVING.md](SERVING.md).

**Images:** `flashnext-mmproj-f16.gguf` is the Qwen3.8-Flash-Next vision encoder (Qwen3-VL style). With it the server
takes `image_url` parts next to text: a 448x448 picture is 196 tokens, the long side is capped at 1,344 (1,764 tokens).
Give the ViT its own card and start the server with `--mmproj`; see [SERVING.md](SERVING.md#images-vision).

## Demos

Example scenes made with this model: web pages it wrote, screen-recorded in a browser.

**Watermelon jelly (WebGPU)**

<video controls width="100%" src="https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1/resolve/main/assets/watermelon_jelly.mp4"></video>

**Rocket launch scene**

<video controls width="100%" src="https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1/resolve/main/assets/rocket_launch.mp4"></video>

**Paris nocturne (voxel garden with fireworks)**

<video controls width="100%" src="https://huggingface.co/lackonendes/PAW-125B-FLASH-NEXT-X3.1/resolve/main/assets/paris_nocturne.mp4"></video>

## What sits where

Flash-Next carries a 51.2-billion-parameter n-gram embedding table. It is most of the file, and it is not loaded
into GPU or RAM: it stays on disk and only the few rows each token needs are read.

| where | what | size |
|---|---|---:|
| **NVMe (disk)** | n-gram embedding table: 51.2 B parameters (320,001,536 rows x 160), int8, read per token | 47.7 GiB (+0.6 GiB scales) |
| **GPU (24 GB card)** | dense weights: attention, DeltaNet, shared experts, router, embeddings, output (Q8_0) | 4.6 GiB |
| | routed experts kept on the GPU (the most used, plus a cache that follows recent routing) | 13.5 GiB |
| | KV cache and DeltaNet state at 262,144 tokens (int8 KV) | 3.9 GiB |
| | total in use at 256k context | about 23.6 GiB |
| **CPU RAM** | the other routed experts (pinned, computed on the CPU or copied to the GPU on demand) | 38.8 GB |
| | whole server process | about 43 GB resident |
| **Disk, total** | both files | 96.8 GiB |

Of the 42.0 GiB of routed experts (24,576 experts), about 6.4 GB are the hottest experts held on the GPU all the
time. The rest of the GPU's 13.5 GiB expert space is a cache of recently used experts, and the remainder of the bank
is in RAM.

**With two 24 GB cards** each card holds its half of the layers (about 23.5 GiB in use on each at 262,144 context),
and 13,466 of the 24,576 routed experts (55%) stay on the GPUs instead of the RAM tier (the server then uses
about 25 GB of RAM instead of 43). That, not running layers in
parallel, is where most of the two-card speedup comes from: far fewer experts have to be computed on the CPU or
copied over PCIe.

## Speed

All numbers at the full 262,144-token context through the OpenAI-compatible server with the MTP draft head, on a
Threadripper 3960X (24 cores), 125 GB RAM and NVMe. The machine is shared with other users, so decode varies by a few
percent from run to run.

### Two RTX 3090s (layer split): the fastest setup

One card runs layers 24-47, the output head and the MTP drafter; the other runs layers 0-23. Plain PCIe, no NVLink.
Prompt chunks are pipelined across both cards; each decode step drafts up to 7 tokens.

| | |
|---|---:|
| **decode, real agent sessions** (16 recorded requests replayed with the served settings: temperature 0.8, thinking on, 12 tools, up to 150K-token context) | **115-116 tok/s** (single requests 85-145) |
| **decode, 150K-token context** (one prompt, 512 tokens, greedy) | **128 tok/s** |
| **prefill** (a fresh 150K-token prompt) | **about 4,500 tok/s** |
| time to first token for a 150K-token prompt | about 33 s |
| tokens per decode step | about 3.1 (agent sessions) |

Decode speed depends mostly on how many drafted tokens are accepted: tool calls that copy code or file contents run
fastest, long free-form reasoning slowest.

### One RTX 3090

Measured on one RTX 3090 (PCIe 4.0 x8, power limit 250 W), serving real agent sessions (MTP head with 3 drafts,
about 2.7 tokens per step). Decode varies by about 10% from run to run.

| | |
|---|---:|
| **decode, peak** (short prompt, up to 3K-token answers: 75-88 tok/s, best 87.5) | **80+ tok/s** |
| **decode** (150K-token context, median of served requests) | **57-60 tok/s** |
| **prefill** (a fresh 150K-token prompt) | **2,170-2,270 tok/s** |
| time to first token for a 150K-token prompt | about 70 s |
| load time | about 2 minutes |

Decode speed falls as the context grows: peak is with a short prompt, the 57-60 figure is with 150K tokens of context.

## Quality

### KL divergence against the BF16 base model

Lower is better; 0 means the same next-token probabilities as BF16. Measured by running the real BF16 model and each
quantized version side by side over the same text: 64 chunks of 2,048 tokens of code/chat and 64 chunks of WikiText-2
(plain English), scoring the second half of each chunk (65,472 tokens per corpus). In every row only the 24,576 routed
experts come from the quantized file; everything else stays BF16, so the numbers compare the expert quantization alone.

| experts from | weights without n-gram table | code KL | plain-English KL |
|---|---:|---:|---:|
| **PAW-125B X3.1 (this model)** | 47.5 GiB | **0.044** | 0.087 |
| turboderp EXL3 3.05 bpw | 47.3 GiB | 0.089 | **0.057** |
| ISTA-DASLab GSQ-RCO IQ3_S | 51.1 GiB | 0.102 | 0.085 |
| ISTA-DASLab GSQ-RCO IQ3_XXS | 43.8 GiB | 0.172 | 0.139 |

### Benchmark


| benchmark | BF16 (Qwen card) | PAW-125B X3.1 | retention |
|---|---:|---:|---:|
| GPQA-Diamond | 91.7 | 92.4 (183/198) * | 100.8% |
| LiveCodeBench v6 | 91.9 | 74.9 (131/175, in progress) | 81.5% (partial) |
| IFBench | 81.3 | 76.7 (115/150) | 94.3% |

\* union over up to four attempts per question, so not a single-pass score.

IFBench is prompt-level loose accuracy, the same metric the BF16 figure uses: an item counts once if any phase of
the ladder gets it right, so a key attempted in several phases is not counted several times. The ladder was stopped
after P5 by design (the 3-item P6 queue was never run and counts as wrong). LiveCodeBench P3 is still running.

## Requirements

- **Engine:** [TRUSS](https://github.com/irawanw/truss), commit `4407f65` or newer. Stock llama.cpp cannot read this file.
- **GPU:** one 24 GB card, or two for about twice the decode and prefill speed (tested on RTX 3090s, Ampere sm_86).
  **CPU:** AVX2, 24 cores recommended. **RAM:** 64 GB.
- **Disk:** 97 GB on **NVMe**. The n-gram table is read from disk while decoding; a spinning or USB disk will be slow.

## Files

| file | contents | size |
|---|---|---:|
| `flashnext-x31-00001-of-00002.gguf` | **the weights**: Q8_0 dense weights, routed experts (3.5 / 2.5 bit), tokenizer and config | 47.54 GiB |
| `flashnext-x31-00002-of-00002.gguf` | the n-gram embedding table (int8 + fp16 row scale) | 48.28 GiB |
| `flashnext-mtp-x3k3.gguf` | MTP draft head for speculative decoding | 0.97 GiB |

The two `flashnext-x31` shards are one model in llama.cpp's split-GGUF layout: put them in the same directory and pass the `-00001-of-00002` file to `--model`; TRUSS finds the second shard by name. They are byte-identical to the former single file (every tensor hash-checked); a bench on the split reproduces the single file's token counts and speed. The MTP head is optional but gives about 2.7 tokens per step instead of 1. Start commands are in
[SERVING.md](SERVING.md).

## Limitations

- **TRUSS only**, on one or two 24 GB GPUs plus CPU (tested on RTX 3090s). Smaller GPUs have not been tried.
- **Benchmarks are not in yet** (see Quality).
- Output at temperature 0 can differ slightly between runs, because which experts run on the CPU or the GPU depends
  on timing.
- Long-context (256k) generation quality has not been measured.

## Integrity

```
7ee7f7bf2f96ffb62a6fe9ec5e09084606f31b8d184a67f3bced374264cfb52e  flashnext-x31-00001-of-00002.gguf
28262010f5207b6798189e414e637f4d5ac351e47b971dcd8f62c042873f2139  flashnext-x31-00002-of-00002.gguf
21de8ab563f562b52245e05e0a9af30609bfe3814a7d9c5b6179a78a9331736b  flashnext-mtp-x3k3.gguf
```

## License

Apache 2.0, inherited from the base model.

## Credits and attribution

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp: its EXL3 format (a QTIP-style trellis
  quantizer) is the basis of the PAW X3 / X3.1 codec, and v1.5.1 is the encoder used here.
- [Strata](https://github.com/Niko1221/Strata) by Niko1221: the serving shape TRUSS's one-GPU path is built on (hot
  experts on the GPU, the rest on the CPU, speculative decoding). TRUSS reimplements it independently.
- [TRUSS](https://github.com/irawanw/truss): our own from-scratch inference engine for this model.
- Base model [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Qwen.
