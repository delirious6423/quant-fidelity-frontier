---
license: mit
library_name: gguf
pipeline_tag: text-generation
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- gguf
- quantized
- deepseek-v4
- deepseek-v4-flash
- moe
- iq2_xxs
- q2_k
- imatrix
- siliang-engine
- expert-major
- experimental
---

# DeepSeek-V4-Flash-0731 IQ2_XXS/Q2_K - Siliang Expert-Major Repack

This is an experimental repack of the 0731 imatrix GGUF from
[`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf/blob/88e011c80d9ec39458f5493ddacab57364143960/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf),
derived from
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
for [`Siliang Engine`](https://github.com/SpeederX/siliang-engine).

The DeepSeek-V4 neural architecture, logical weights, imatrix-derived codes,
and mixed quantization are unchanged. The modification is a Siliang-specific
expert-major GGUF storage layout and metadata extension for routed
Mixture-of-Experts weights. No fine-tuning or additional quantization pass was
performed.

> [!IMPORTANT]
> This file is not a general-purpose GGUF. It requires Siliang Engine v0.1.1
> or a compatible later build that understands the `siliangem.*` metadata
> contract. Plain upstream `llama.cpp`, DS4/DwarfStar, and other GGUF
> applications do not understand the packed expert-major tensors.

## Modification notice

This repository redistributes modified model material. The source GGUF's 129
routed-expert tensors were physically rearranged into 43 Siliang expert-major
regions. Expert IDs, tokenizer metadata, model architecture, and intended
model behavior were not deliberately changed by the repack.

The packed regions use raw byte-container tensors. This does not mean the
weights were converted to I8: the original routed types remain `IQ2_XXS` for
gate/up and `Q2_K` for down, and their logical geometry is recorded in the
`siliangem.*` metadata.

## What the repack changes

The stock GGUF stores each routed-expert projection as a tensor spanning all
experts. This repack instead places the projections for each expert together:

```text
[expert 0: gate | up | down][expert 1: gate | up | down] ...
```

Each layer is represented by one packed region. Siliang Engine reconstructs
the original logical gate, up, and down tensors from the embedded geometry.
This lets one arena miss fetch an expert with one sector-aligned contiguous
read rather than three separate projection reads.

The layout is designed to reduce fragmented read submissions, reduce reliance
on Windows mmap page faults for routed-expert weights, and lower effective
expert-fetch latency. It does not guarantee lower total bytes read, fewer
physical page-ins, or a specific speedup. Results depend on storage, available
RAM, memory pressure, routing behavior, and arena sizing.

## Compatibility

- Engine: [Siliang Engine v0.1.1](https://github.com/SpeederX/siliang-engine/releases/tag/v0.1.1)
  or a compatible later build.
- Accelerated runtime: Windows. Linux support is not available in the current
  experimental release.
- Memory mapping must remain enabled. Siliang deliberately rejects this layout
  with `--no-mmap` because its logical expert tensors are strided views.
- Arena acceleration is opt-in. Without a positive `SILIANGEM_CACHE_MIB`,
  Siliang uses its mmap path and does not reserve an arena.
- The source GGUF includes a baked chat-v2 template. For advanced 0731 message
  encoding and `reasoning_effort` behavior, consult DeepSeek's
  [official encoding implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding)
  and validate the behavior of your client.
- The separate `DeepSeek-V4-Flash-DSpark-support-0731.gguf` companion is not
  included or repacked here. This artifact makes no speculative-decoding
  claim.
- The experimental arena path is validated with one model and one active
  context per process. Multi-context arena use is not claimed here.
- This is a monolithic GGUF. Split-GGUF arena operation is outside the
  supported path for this artifact.

## Quickstart

Download and extract a Siliang Engine Windows release. From its root, choose
an arena budget measured for your machine and leave enough memory for Windows,
non-expert weights, the KV cache, and GPU shared-memory pressure:

```powershell
.\scripts\siliang-env.ps1 -CacheMiB <measured-MiB>

& ".\bin\llama-cli.exe" `
    -m "<path>\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731-siliang-expert-major.gguf" `
    -cnv

.\scripts\siliang-env.ps1 -Reset
```

Larger arenas are not automatically better. Do not treat the 18 GiB benchmark
configuration below as a safe default for another machine. The GGUF advertises
a 1,048,576-token context, but that maximum context was not validated by the
Siliang benchmark and has substantial memory requirements.

## Artifact details

| Field | Value |
| --- | --- |
| File | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731-siliang-expert-major.gguf` |
| Size | 86,923,096,064 bytes (80.953442 GiB) |
| Retained SHA-256 | `83d09412bbfbbcc0fa9f77afeee763d5f72da1ad055be50675111f585ab8b99a` |
| GGUF | Version 3, 512-byte alignment |
| Model architecture | `deepseek4` |
| Context metadata | 1,048,576 tokens |
| MoE geometry | 43 layers, 256 routed experts per layer, 6 selected per token, 1 shared expert |
| Routed quantization | `IQ2_XXS` gate/up, `Q2_K` down |
| Packed layout | 43 tensors, `gate | up | down` per expert |
| Expert stride | 7,096,320 bytes |
| Metadata namespace | Nine `siliangem.*` keys; no retired `behemoth.*` keys |

The exact source GGUF is 86,720,111,488 bytes with SHA-256
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
The repack adds 202,984,576 bytes (193.581 MiB, 0.234%) primarily for canonical
per-expert padding and alignment.

## Validation

From a Siliang Engine checkout, run the structural checker:

```powershell
python .\scripts\check_expert_major.py `
    --model "<path>\DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731-siliang-expert-major.gguf"
```

The expected result includes:

```json
{
  "alignment": 512,
  "expertCount": 256,
  "layerCount": 43,
  "packedTensorCount": 43,
  "partNames": ["gate", "up", "down"],
  "partTypeIds": [10, 16],
  "status": "expert-major-metadata-ok"
}
```

This artifact passes that structural check. A current 64-sample verifier also
matched randomly selected layer/expert payload slices byte-for-byte against
the exact stock source. The controlled benchmark below produced byte-identical
256-token output across all six stock-layout and expert-major cells, with no
mmap fallback.

These checks do not constitute a general semantic-equivalence proof or replace
application-specific correctness evaluation.

## Performance evidence

The strongest retained measurement directly compares the stock tensor layout
with this expert-major layout while keeping the same runtime, physical disk,
request, and 18 GiB Siliang arena in both arms.

| Path | Median decode | Observed range |
| --- | ---: | ---: |
| Stock GGUF, 18 GiB arena | 2.274 tok/s | 2.269-2.407 tok/s |
| Expert-major GGUF, same arena | 2.774 tok/s | 2.689-2.850 tok/s |

This fully cold experiment used three fresh process starts per arm and purged
the Windows standby list before every cell. All six requests generated exactly
256 tokens with byte-identical output. The measured median layout change was
1.22x, or +22.0%.

The stock path issued 34,866 engine-level expert read requests per cell. The
expert-major path issued 11,653, a 66.6% reduction. Total owned-process read
operations fell from 37,945 to 14,418 (-62.0%), while process read bytes were
77.786 versus 78.059 GiB (+0.35%). This is evidence for fewer read operations,
not fewer bytes. The combined soft-and-hard page-fault median fell by 6.3%, but
that counter does not prove fewer physical disk page-ins. Per-expert latency
was not independently measured.

The test system used Windows 11, an AMD Ryzen 5 2600, 24 GB of RAM, an NVIDIA
GeForce RTX 2070 with 8 GB of VRAM, and a WD_BLACK SN850X NVMe SSD operating
through a PCIe 3.0 x4 platform link. The benchmark used source build 10270
(`071327508`) with `GGML_NATIVE=ON`; it was not a packaged v0.1.1 performance
run. The 18 GiB arena created high memory pressure. These numbers are
experimental single-system evidence, not a release-package or general
performance guarantee. Full methodology, raw repetitions, identities, and
caveats are in the
[`Siliang Engine performance notes`](https://github.com/SpeederX/siliang-engine/blob/main/docs/PERFORMANCE.md#current-fully-cold-expert-major-layout-benchmark-2026-08-10).

## Source, license, and attribution

- Base model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- Exact source GGUF: [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf/blob/88e011c80d9ec39458f5493ddacab57364143960/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf)
- Runtime: [`SpeederX/siliang-engine`](https://github.com/SpeederX/siliang-engine)
- Model report and citation: [`DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence`](https://arxiv.org/abs/2606.19348)
- Official 0731 message encoding: [`deepseek-ai/DeepSeek-V4-Flash-0731/encoding`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/main/encoding)

The model material is distributed under the MIT License. The complete license
text is included in [`LICENSE`](LICENSE). Review the upstream model card for
capabilities, safety considerations, intended use, and broader model
limitations.
