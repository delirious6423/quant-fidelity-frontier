---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - unsloth/Qwen3.8-Flash-Next-GGUF
base_model_relation: quantized
tags:
  - qwen3.8
  - basalt
  - mugearite
  - moe
  - rtx-5090
  - windows
---

# Qwen3.8-Flash-Next UD-IQ1_M

Unsloth's UD-IQ1_M GGUF of Qwen3.8-Flash-Next, converted to a `.basalt` pack for Mugearite, a custom fork of
[Basalt](https://github.com/jesdga95/basalt). It is the speed pack beside
[Mugearite/Qwen3.8-Flash-Next-IQ3_S-Q4](https://huggingface.co/Mugearite/Qwen3.8-Flash-Next-IQ3_S-Q4): a 71 GB file
that decodes about 20% faster, with higher KL divergence and a GPQA-Diamond score within noise of it.

It is a `.basalt` version 1 file. Mugearite reads it. Basalt 0.2 and later read only version 2 files, so it does not
run on upstream Basalt.

Experimental.

## Files

| File | Size | SHA-256 |
|---|---|---|
| `Qwen3.8-FN-UD-IQ1_M.basalt` | 66.4 GiB | `c6d1a26568de3a7766e89534d5927932e47a6aa75cdbd4981b7a03062cbef997` |
| `mugearite-ud-iq1m-1slot.json` | | Mugearite server config, one request at a time, 262,144-token context |
| `mugearite-ud-iq1m-8slot.json` | | Mugearite server config, 8 requests at once, 262,144 tokens each |
| `mugearite-ud-iq1m-light.json` | | Mugearite server config for a 16 GB card and 32 GB of RAM (see Smaller machines) |
| `LICENSE` | | Qwen Community License 1.0 |

## What is in it

Despite the name, it holds no IQ1_M tensors:

| Part | Type |
|---|---|
| routed experts, gate and up | IQ2_XXS in 23 layers, IQ2_XS in 7, IQ2_S in 18 |
| routed experts, down | Q2_0 in 46 layers, IQ4_NL in 2 |
| attention | Q6_K |
| Gated DeltaNet q/k/v | Q5_K |
| Gated DeltaNet output projections, shared experts, hyper-connections | Q8_0 |
| output | Q6_K |
| token embedding | IQ4_XS |
| n-gram table (`per_layer_token_embd`) | IQ4_NL |

Every tensor is Unsloth's stored bytes, converted without re-quantization. The MTP draft head (its dense layer, its
experts and its draft vocabulary) is byte for byte the one in Mugearite/Qwen3.8-Flash-Next-IQ3_S-Q4. The vision
projector is stored at BF16.

## Quality

Wikitext-2, 100 chunks of 512 tokens, against a llama.cpp Q8_0 KL-divergence base (base PPL 3.2808), int8 KV cache:

| | file | PPL | KLD mean | same top-1 |
|---|---:|---:|---:|---:|
| this pack | 71.3 GB | 3.3781 | 0.2563 | 83.05% |
| GSQ-RCO IQ3_S | 87.6 GB | 3.3738 | 0.1520 | 87.29% |
| Mugearite IQ3_S-Q4 | 119.6 GB | 3.2309 | 0.0979 | 89.65% |

## Reasoning benchmark

GPQA-Diamond, served through Mugearite's OpenAI-compatible API, one question at a time, thinking at the default
effort, the model's own sampling (temperature 1.0, top_p 0.95, top_k 20) and a 262,144-token budget:

| | GPQA-Diamond | mean time per answer |
|---|---:|---:|
| this pack | 90.91 (180/198, 1 sample) | 47.9 s |
| Mugearite IQ3_S-Q4 | 92.93 (184/198, 1 sample) | 60.3 s |

A single GPQA run varies by about 2 points, so the two scores are within noise of each other. This pack thinks about
9% more tokens per answer, and still answers about 20% sooner.

## Speed

Development measurements on one RTX 5090 at 450 W with a Ryzen 7 9800X3D, not benchmark runs.

One request at a time (the 1-slot config), a first read of 10,000 or 100,000 tokens, then follow-up turns of about
330 tokens:

| | Mugearite IQ3_S-Q4 | this pack |
|---|---:|---:|
| decode at 10K / 100K context, tok/s | 288 / 281 | 355 / 334 |
| a follow-up read at 10K / 100K context, ms | 254 / 292 | 140 / 168 |
| first read at 10K / 100K, tok/s | 8,900 / 9,800 | 9,200 / 10,000 |

Several requests at once, all tokens counted together:

| | tok/s |
|---|---:|
| 4 requests, 80,000-token conversations, 256-token replies | about 410 |
| 8 requests, the same | about 435 |
| 8 coding agents in live use, 30,000-70,000 tokens each | 550-600 |

Its smaller experts are where the speed comes from: more of them fit in VRAM. With 8 requests at once, 97.7% of routed
expert lookups found their expert in VRAM, against 90.8% for IQ3_S-Q4 in the same test (247 tok/s).

## Running it

Mugearite on Windows, an NVIDIA Blackwell card (RTX 50 series) and an AVX-512 CPU:

```
basalt serve --config mugearite-ud-iq1m-1slot.json --api-key <your key>
```

Clients send the config's alias, `qwen3.8-flash-next-ud-iq1m`. Put the pack in a `models` folder beside the config,
or change `model`. Keep the server on 127.0.0.1, or set a real API key before you expose it.

The 1-slot and 8-slot configs are for an RTX 5090 and a machine with a lot of RAM: they lock the n-gram table in RAM
(about 31 GiB) and allow up to 96 GiB more for parked conversations. For less, see the next section.

The 1-slot and 8-slot configs leave 1,500 MiB of VRAM unused for the image encoder (`vram-reserve-mib`). The light
config takes no images and keeps the default 700 MiB.

## Smaller machines

`mugearite-ud-iq1m-light.json` is for a 16 GB card and 32 GB of RAM:

- The experts the GPU does not hold stay in the file, and the engine keeps as many of them in RAM as fit, the most
  used first (`resident-experts`). With enough RAM that is all of them. With less, the rest are read from the SSD
  when they are needed.
- The n-gram table is read from the SSD (`ple-io direct`) and never enters RAM.
- Conversations you switch away from are parked on disk, in a `conversations` folder beside the config (up to
  32 GiB), not in RAM.
- One request at a time, a 131,072-token context, and no image input.

Measured on the RTX 5090 with the expert cache limited to what a 16 GB card holds, on a machine with enough RAM for
every expert:

| | |
|---|---:|
| decode | 171-196 tok/s |
| a fresh read of 50,000 tokens | 17.5 s |
| follow-up reads of 600-1,300 tokens | 0.7-1.3 s |
| returning to a parked conversation | about 1 s |

That run held about 36 GiB of RAM. A 32 GB machine has not been measured. There, expect it to slow down whenever an
expert has to come from the SSD, and the first prompt after a start to take several minutes while those experts are
read in.

## Where it comes from

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), under the Qwen Community License 1.0
  (see `LICENSE`).
- UD-IQ1_M by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (revision `084c631a`), converted
  with Mugearite's copy of the Basalt converter.
