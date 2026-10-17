---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - moe
  - mixture-of-experts
  - quantized
  - 4-bit
  - apple-silicon
  - ssd-streaming
  - finchmoe
---

# Qwen3.8-Flash-Next-125B — FinchMoE `.finch` streaming repack (4-bit)

> [!WARNING]
> **This is an unofficial community repack, not a Qwen release.** It is not
> affiliated with, endorsed by, or supported by Qwen or Alibaba. The upstream
> model is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next);
> all credit for the model belongs there.

> [!IMPORTANT]
> **This repository is not loadable by `transformers`, vLLM, SGLang, llama.cpp,
> MLX or Ollama.** It is not a safetensors or GGUF checkpoint. It is in a custom
> layout designed for one engine — [FinchMoE](https://github.com/haihengh/finchMoE),
> a Swift/Metal runtime that streams routed experts from disk instead of holding
> the model in memory. Please read [How to use](#how-to-use) before downloading
> 162 GB you may not be able to run.

## What this is

A 4-bit affine quantized repack of **Qwen3.8-Flash-Next** into the `.finch`
streaming format, produced by the FinchMoE installer. The routed-expert pool —
by far the largest part of the model — is stored as one file per layer, laid out
so the engine can `pread` individual experts on demand and cache a small working
set in memory.

The point of the format is that the model does not need to fit in RAM. On a
16 GB Mac, the engine keeps at most 16 experts per layer resident and streams
the rest from an SSD as the router asks for them.

## Contents

| Path | Size | What it is |
| --- | ---: | --- |
| `packed_experts/layer_00.bin` … `layer_47.bin` | 63 GB | 48 files, 1,417,674,752 B each — the routed-expert pool, 512 experts per layer at a 2,768,896-byte stride |
| `ple_shards/shard_000.bin` … | 95 GB | n-gram embedding (PLE) shards, read on their own path every decode step |
| `model_weights.bin` | 3.6 GB | every non-expert weight: attention, linear attention, router, shared experts, embeddings, head |
| `tokenizer/` | 23 MB | tokenizer, vocab, merges and the chat template |
| `manifest.json` | 31 KB | architecture, quantization scheme, and per-file SHA-256 for every artifact |
| `verified-install.json` | 31 KB | the install receipt — SHA-256 and size for every file above |

Total: **162 GB**. There is no `LICENSE` requirement to accept before download
(the upstream repository is not gated), but see [License](#license).

## Quantization

Affine, group size 64, per group scale and bias in BF16:

| Component | Bits |
| --- | ---: |
| Routed experts | 4 |
| Shared experts | 4 |
| Attention | 4 |
| Embedding | 4 |
| Linear attention (GDN) | 8 |
| Router | 8 |

The two 8-bit components are the ones whose precision the engine found it could
not spend: the router's gate scores and the Gated DeltaNet state path. Nothing
in the pack is below 4 bits.

## How to use

You need the [FinchMoE](https://github.com/haihengh/finchMoE) engine. This
repository is a model payload, not a runnable program.

```sh
git clone https://github.com/haihengh/finchMoE
cd finchMoE

# place this repository at models/Qwen3.8-Flash-Next-125B.finch
# (or point --model at it)

./.build/release/FinchMoECLI \
  --model models/Qwen3.8-Flash-Next-125B.finch \
  --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json \
  --max-context 2048 --max-new 32 --temperature 0 \
  --expert-cache-slots 16 --counters
```

**Verify the download before running it.** `verified-install.json` carries a
SHA-256 and a byte count for every file, and the engine checks them:

```sh
FinchMoECLI --model models/Qwen3.8-Flash-Next-125B.finch --verify full-sha256
```

The first full verification hashes ~162 GB and takes about a minute of pure I/O
on a fast external SSD. After it passes once, `--verify trusted-install` skips
the re-hash and uses the receipt.

> [!NOTE]
> `manifest.json` records `modelID` as `local/Qwen3.8-Flash-Next-125B` — the
> name the install was built under locally. It is cosmetic and the engine does
> not resolve it. Do not "fix" it: `verified-install.json` contains the SHA-256
> of `manifest.json`, so editing the field invalidates the receipt.

## Requirements and measured performance

The engine targets Apple Silicon and streams from storage, so a fast path to
the weights matters more than free RAM. These are the numbers measured on the
machine this repack was built and validated on — a **Mac mini, 16 GB unified
memory, weights on an external Samsung 990 EVO 2 TB behind a USB4 bridge**,
greedy decoding, `--max-context 2048`:

| | |
| --- | --- |
| Decode | 2.76–2.91 tok/s |
| Prefill | 5.9–6.0 tok/s |
| Peak compressed memory | ~3.1 GB |

Treat those as an illustration of the storage-bound shape rather than a
specification: they move with the drive. On this box the same engine reads
another install's expert pool at roughly twice the rate, and how many reads are
in flight, how large they are and how they are spaced all turn out to matter far
less than the storage path itself. `--counters` prints the full breakdown.

Memory: the model never needs to fit. A 16 GB machine runs it; the constraint is
that the weights should live on an SSD rather than a spinning disk or a network
mount.

## Format

`manifest.json` is the authoritative description. In brief:

```
magic              FINCH
format version     1.0
numLayers          48
expertsPerLayer    512
expertStride       2,768,896 bytes
experts selected   10 per token (top-k)
```

Expert `e` of layer `L` begins at `e * expertStride` in
`packed_experts/layer_LL.bin`. Every expert is the same size, which is what
lets the engine read one without an index lookup — the trade is that the file
is padded to the largest expert in the layer.

## Provenance

| | |
| --- | --- |
| Upstream model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Upstream license | Qwen Community License 1.0 |
| This repack | 4-bit affine, group size 64, produced by the FinchMoE installer |
| Source snapshot | recorded in `manifest.json` as `sourceSnapshotHash` |

The repack was validated with an oracle comparison against the reference path —
the logits agreed on the argmax, and the install generates coherent text. That
is a correctness check on the packing and the quantization, not an evaluation of
answer quality: no downstream benchmark suite has been run on this artifact, and
none is claimed here. If you need a quality number for a Qwen3.8-Flash-Next
quantization, measure it yourself on your own workload.

## License

**Qwen Community License 1.0** — the full text is included as
[`LICENSE`](LICENSE) in this repository, as the license requires.

The license permits redistribution and derivative works, including quantized
ones like this. Two conditions attach to *commercial use by the licensee* — a
model-name display requirement above 100M monthly active users or US$20M monthly
revenue, and a requirement to obtain a separate license from Qwen before using
the model or its derivatives to run a Model-as-a-Service or an AI coding /
office-productivity assistant. Redistributing weights is not itself
Model-as-a-Service under the license's definition, but if you intend to host
this behind an API, read clause 2 first. This note is a summary, not legal
advice; the LICENSE file governs.

If you are the rights holder and want this repository changed or removed, open
an issue and it will be actioned.
