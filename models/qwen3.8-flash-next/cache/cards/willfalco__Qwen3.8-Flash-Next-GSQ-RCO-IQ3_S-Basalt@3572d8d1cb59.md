---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- basalt
- gsq-rco
- iq3_s
- q8_0
- moe
- mtp
---

# Qwen3.8-Flash-Next GSQ-RCO IQ3_S for Basalt

[ISTA-DASLab's GSQ-RCO IQ3_S quantization](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), packed as `.basalt` files for
[Basalt](https://github.com/jesdga95/basalt). Basalt is a local inference engine for NVIDIA Blackwell (sm_120) GPUs; it keeps the
experts that do not fit in VRAM in system RAM. Its own published packs carry NVIDIA's NVFP4 and Unsloth's GGUF quantizations; this repo
carries ISTA-DASLab's GSQ-RCO in two n-gram table variants.

## Which file to pick

| Pack | File size | N-gram table (`per_layer_token_embd`) | RAM / VRAM | Speed | KLD vs BF16 | PPL (base `3.2733`) |
|---|---:|---|---|---|---:|---:|
| **`Qwen3.8-FN-GSQ-RCO-IQ3_S-Q8NG.basalt`** *(recommended)* | 113.2 GB | `Q8_0` (54.4 GB, 0.5% row error vs BF16) | same | same | `0.14169` | `3.3556` |
| **`Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt`** | 87.6 GB | `IQ4_NL` (28.8 GB, 7.6% row error vs BF16) | same | same | `0.15065` | `3.3847` |

Basalt does not load the n-gram table into RAM or VRAM—by default (`--ple-io direct`) it reads 16 rows per token straight from the SSD
with a small row cache (~95 MB for `IQ4_NL`, ~180 MB for `Q8_0`). Upgrading the table from `IQ4_NL` to `Q8_0` costs only +25.6 GB of
disk space, with identical speed and memory usage.

## What is in the files

Both packs share the exact same experts, dense weights, MTP drafter and vision projector:

- **The model:** all 1,224 tensors of ISTA-DASLab's `IQ3_S` GGUF (routed experts, dense weights, and in the 87.6 GB pack the `IQ4_NL`
  n-gram table), copied byte for byte. Nothing was re-quantized.
- **`Q8_0` n-gram table (`-Q8NG` pack only):** replaces `per_layer_token_embd.weight` (320M × 160) with
  [Unsloth's `Q8_0` GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) shard 3 (`Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf`,
  revision `f5d175de`), a single BF16 → `Q8_0` quantization rather than an intermediate FP8 conversion.
- **MTP drafter:** the BF16 `mtp.*` tensors of Qwen's checkpoint (revision `de4b8e4d`), with the tuned (self-distilled) dense part and
  the draft vocabulary of Basalt's own `Qwen3.8-FN-UD-IQ3_XXS` pack. The GGUF itself has no MTP head.
- **Vision projector:** ISTA-DASLab's `mmproj-Qwen3.8-Flash-Next-BF16.gguf`.

Built with the converter of Basalt v0.5.0. For both packs, `--verify` found every GGUF tensor byte-identical (`1224 compared, 0 differ`)
and every section SHA-256 checksum valid. It also prints 29 lines such as `mtp/dense holds fc_embedding.weight, which the source does
not have`; those are the drafter tensors taken from Qwen's checkpoint rather than the GGUF, so they are expected.

## Quality (KLD vs BF16)

Measured with Basalt v0.5.0 `tools/ppl_gate.py` against Basalt's `flashnext-bf16-base.kld` reference over 100 wikitext-2 chunks
(25,500 tokens, `T=3`, `--kv int8`, `--resident-experts`, frozen tiers `--adapt-swaps 0 --peer-adapt-swaps 0`, `--gpu 0,1`). Both runs
had the exact same expert split (`31.3%` VRAM / `41.4%` peer GPU / `27.4%` CPU) and `25 s` load time:

| Metric | `IQ4_NL` n-gram | `Q8_0` n-gram (`-Q8NG`) | Paired delta [95% CI] |
|---|---:|---:|---|
| **KLD mean** | `0.15065` | `0.14169` | `-0.00897` `[-0.01138, -0.00655]` (−6.0%, measurable) |
| **KLD median / p99** | `0.03436` / `1.7066` | `0.03282` / `1.5817` | `-0.00154` / `-0.1249` (−7.3% p99) |
| **PPL** (BF16 base `3.2733`) | `3.3847 ± 0.0451` | `3.3556 ± 0.0445` | `-0.0291` (−26% of excess PPL over BF16) |
| **NLL** (`ln(PPL/base)`) | `1.21926` (`0.03347`) | `1.21062` (`0.02483`) | `-0.00864` `[-0.01286, -0.00442]` (−25.8% excess NLL, measurable) |
| **Same top-1 %** | `87.53%` | `87.87%` | `+0.34%` `[+0.11, +0.56]` (measurable) |
| **Mean \|Δlogp\| (base top-1)** | `0.21901` | `0.20936` | `-0.00965` |
| **100-chunk runtime** | `285.2 s` (`429` ms pp / `2423` ms dec) | `285.7 s` (`428` ms pp / `2428` ms dec) | `+0.5 s` (no speed penalty) |

## Run it

You need an NVIDIA Blackwell GPU (RTX 50 series or RTX PRO), driver R580 or newer, the NVIDIA Container Toolkit, Linux x86-64 with AVX2,
and about 60 GB of free RAM: the experts the GPUs do not hold (50.4 GB) stay there, page-locked.

**Option 1 — `Q8_0` n-gram pack (recommended, 113.2 GB):**

```sh
hf download willfalco/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Basalt --exclude "Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt" --local-dir Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Basalt
cd Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Basalt
docker compose -f Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Q8NG-basalt.yaml up -d
docker logs -f mod0
```

**Option 2 — standard `IQ4_NL` n-gram pack (87.6 GB):**

```sh
hf download willfalco/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Basalt --exclude "Qwen3.8-FN-GSQ-RCO-IQ3_S-Q8NG.basalt" --local-dir Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Basalt
cd Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Basalt
docker compose -f Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-basalt.yaml up -d
docker logs -f mod0
```

Both compose files use the same service and container name (`mod0`), so if you download both packs you can switch between them with
`docker compose -f <either yaml> up -d`.

The first start builds Basalt v0.5.0 from its git tag, with FFmpeg 6 added for video input (10 to 20 minutes). The server then answers
on port 8080 with the OpenAI and Anthropic APIs, as model `mod0` (Basalt refuses other model names). It has no API key; add
`--api-key=KEY` to `command:` to require one.

What the compose files set:

- `--gpu=0,1`: two 16 GB cards. The first runs every layer, the second holds and computes experts. For one card use `--gpu=0`.
- 256K context, int8 KV cache, 2 requests at once, each allowed the whole 256K.
- Image and video input.
- Default sampling for requests that send none: temperature 1.0, top_p 0.95, top_k 20.
- Basalt's options that change answers (route tail skip, route prior, relaxed drafting) stay off, as they are by default.

## Speed on our machine

2x RTX 5070 Ti 16 GB (PCIe 4.0 x8 each), Ryzen 9 3950X (AVX2, no AVX-512), 128 GB DDR4, the compose file above. Read from the server's
log during ordinary use, not a benchmark (both packs run at the same speed):

| | |
|---|---|
| One request, prose | 117 to 140 tok/s |
| One request whose answer repeats its context | up to 260 to 330 tok/s |
| Two requests at once | about 190 tok/s in total |
| Prompt reading | 2,090 tok/s at 8K tokens, 4,456 at 35K |
| Loading the weights | 52 GiB in 21 s |

## Rebuild them yourself

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-basalt-prep.sh` rebuilds `Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt` from ISTA-DASLab's GGUF. It builds
  Basalt v0.5.0, fetches the MTP tensors from Qwen's checkpoint, reads the tuned drafter's two sections from Basalt's `UD-IQ3_XXS` pack
  by HTTP range (SHA-256 checked), converts and verifies.
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-Q8NG-basalt-prep.sh` rebuilds `Qwen3.8-FN-GSQ-RCO-IQ3_S-Q8NG.basalt` by fetching Unsloth's `Q8_0`
  n-gram shard (SHA-256 checked), swapping its 192-byte GGUF header to match ISTA-DASLab's shard-2 metadata with type `Q8_0`, copying
  the drafter and vision sections from the first pack, converting and verifying.

Instructions for both are at the top of each script.

## Credits

- **[Qwen](https://huggingface.co/Qwen)** for Qwen3.8-Flash-Next.
- **[ISTA-DASLab](https://huggingface.co/ISTA-DASLab)** for the GSQ-RCO quantization, and for the IQ3_S GGUF and the vision projector in
  these files.
- **[Unsloth](https://huggingface.co/unsloth)** for the `Q8_0` n-gram table shard from
  [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) used in the `-Q8NG` pack.
- **[jesdga](https://github.com/jesdga95)** for [Basalt](https://github.com/jesdga95/basalt), its converter, and the tuned MTP drafter
  and draft vocabulary from the [official Basalt packs](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt).
- **[Niko1221](https://github.com/Niko1221)** for [Strata](https://github.com/Niko1221/Strata), the engine Basalt forks.
- **Ian Ranson and contributors** for [Infernix](https://github.com/Wallawalla47/Infernix), whose kernels Basalt adapts.
- **[ggml and llama.cpp](https://github.com/ggml-org/llama.cpp)** for the GGUF format and its quantization types.
- **[NInfer](https://github.com/Neroued/ninfer)**, whose ideas Basalt builds on.

## License

The weights are under the [Qwen Community License 1.0](LICENSE). It allows use, redistribution and derivative works as long as the
license notice comes along. Products with more than 100 million monthly active users or US$20 million monthly revenue must show the
model name, and companies running a Model-as-a-Service or AI work assistant business need a separate license from Qwen for commercial
use; read the full text before you rely on it.

ISTA-DASLab publishes its GGUF under Apache-2.0 ([copy](LICENSE-APACHE-2.0.txt)). Unsloth's `Q8_0` GGUF shard is also distributed under
the Qwen Community License 1.0. These files repackage those tensors, unchanged, into Basalt's file format. Basalt itself is
MIT-licensed; the compose files build it from source, and no Basalt code is included here.
