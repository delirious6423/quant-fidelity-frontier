---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- quantized
- qwen4exp
- qwen3.8
- qwen3.8-flash-next
- moe
- iq2_xs
- imatrix
- siliang-engine
- expert-major
- experimental
---

# Qwen3.8-Flash-Next UD-Q2_K_XL (single expert schema) - Siliang Expert-Major Repack

This is an experimental repack of the `UD-Q2_K_XL` GGUF from
[`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
derived from [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
for [`Siliang Engine`](https://github.com/SpeederX/siliang-engine).

> [!IMPORTANT]
> This file is not a general-purpose GGUF. It requires Siliang Engine v0.1.8
> or a compatible later build that understands the `siliangem.*` metadata
> contract. Plain upstream `llama.cpp` and other GGUF applications do not
> understand the packed expert-major tensors.

## Modification notice

This repository redistributes modified model material under the Qwen
Community License (see [`LICENSE`](LICENSE)). Two things were changed relative
to the Unsloth `UD-Q2_K_XL` source:

1. **One layer was requantized.** In the source, the routed-expert gate and up
   projections of layer 2 are `IQ3_XXS`, while every other layer uses
   `IQ2_XS`. Those two tensors (`blk.2.ffn_gate_exps.weight`,
   `blk.2.ffn_up_exps.weight`) were requantized from the BF16 weights to
   `IQ2_XS` with Unsloth's importance matrix, so all 48 routed layers share one
   expert schema. This changes those weights.
2. **The routed experts were repacked expert-major.** The 144 routed-expert
   tensors were rearranged into 48 Siliang expert-major regions. Expert IDs,
   tokenizer metadata, the chat template, the model architecture, and every
   other tensor are unchanged.

No fine-tuning was performed.

### Why a single expert schema

Siliang Engine keeps a GPU cache (K) of routed experts. When every routed layer
uses the same expert size, all layers can share one K, which bounded prefill
and the decode-only K extension need. With the original mixed layer 2, the
arena falls back to per-layer K slices.

### Quality cost of the requantized layer

Teacher-forced against the original `UD-Q2_K_XL` layout (same engine, all
experts on GPU, one 64-token continuation of a fixed prompt), the next-token
top-1 agreed at 61 of 65 positions, with a mean top-10 KL divergence of 0.095
(maximum 0.76). This is one prompt, not a perplexity or benchmark campaign.

## What the repack changes

The stock GGUF stores each routed-expert projection as a tensor spanning all
experts. This repack places the projections of each expert together:

```text
[expert 0: gate | up | down][expert 1: gate | up | down] ...
```

Each layer is represented by one packed region. Siliang Engine reconstructs
the logical gate, up, and down tensors from the embedded geometry, so one
arena miss fetches an expert with one sector-aligned contiguous read instead
of three projection reads. The raw byte-container tensors do not mean the
weights were converted: the routed types remain `IQ2_XS` (gate/up) and
`IQ4_NL` (down), recorded in the `siliangem.*` metadata.

## Compatibility

- Engine: [Siliang Engine v0.1.8](https://github.com/SpeederX/siliang-engine/releases/tag/v0.1.8)
  or a compatible later build. v0.1.8 is the first release qualified for
  `qwen4exp`.
- Accelerated runtime: Windows with CUDA.
- Memory mapping must stay enabled unless you use Siliang's managed no-mmap
  backing (`--load-mode none` with a nonzero L2). Siliang rejects plain
  `--no-mmap` for this layout because its logical expert tensors are strided
  views.
- Use `--parallel 1` with the L1 arena.
- The GGUF carries Unsloth's chat template, which supports thinking; a request
  turns it on or off with `chat_template_kwargs.enable_thinking` and can cap it
  with `thinking_budget_tokens`.
- The GGUF advertises a 262,144-token context. Siliang measurements used 32k
  and 64k.
- This is a monolithic GGUF. Split-GGUF arena operation is outside the
  supported path for this artifact.

## Quickstart

Download and extract the Siliang Engine v0.1.8 CUDA release, then start the
server with the 32k agent profile from the engine's
[`docs/CONFIGURATION.md`](https://github.com/SpeederX/siliang-engine/blob/main/docs/CONFIGURATION.md):

```powershell
& ".\bin\llama-server.exe" -m "<path>\Qwen3.8-Flash-Next-UD-Q2_K_XL-uniform-iq2xs-siliang-expert-major.gguf" `
  -c 32768 -b 32768 -ub 3072 --parallel 1 -ngl 99 -ncmoe 48 --no-op-offload -t 8 -tb 8 `
  --expert-cache --expert-cache-l2-mib 12288 --expert-cache-l2-policy lfu `
  --expert-cache-l1-k 512 --expert-cache-l1-k-decode 1440 `
  --expert-cache-exchange-r 20 --expert-cache-elevator-p 20 --expert-cache-l1-policy slfu `
  --expert-cache-prefill --expert-cache-prefill-layer-major --no-checkpoint-ubatch `
  --expert-cache-hybrid --expert-cache-staging-threads 2 --expert-cache-l2-pinned-mib 8192 `
  --ctx-checkpoints 4 --cache-ram 0 --reasoning-budget 16384 `
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

These values fit a 24 GB RAM / 8 GB VRAM machine. Measure your own L2 and
pinned sizes; larger is not automatically better.

## Artifact details

| Field | Value |
| --- | --- |
| File | `Qwen3.8-Flash-Next-UD-Q2_K_XL-uniform-iq2xs-siliang-expert-major.gguf` |
| Size | 78,875,438,592 bytes (73.46 GiB) |
| SHA-256 | `3f4d846472e06d990e0a4ed8debb56c652e10cb1f9757e755e1ab658450f45a9` |
| GGUF | Version 3, 512-byte alignment, 1,128 tensors |
| Model architecture | `qwen4exp` |
| Context metadata | 262,144 tokens |
| MoE geometry | 48 layers, 512 routed experts per layer, 10 selected per token |
| Routed quantization | `IQ2_XS` gate/up, `IQ4_NL` down (all 48 layers) |
| Packed layout | 48 tensors, `gate \| up \| down` per expert |
| Expert stride | 1,875,456 bytes (padded to a multiple of 170,496) |

## Reproduction

All sources are in [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):

- `UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-0000{1,2,3}-of-00003.gguf`, merged
  into one GGUF with `llama-gguf-split --merge`;
- `BF16/Qwen3.8-Flash-Next-BF16-00005-of-00008.gguf`, of which only the byte
  ranges of the two layer-2 tensors are needed (listed in
  [`requant/bf16_layer2.json`](requant/bf16_layer2.json));
- `imatrix_unsloth.gguf_file` (SHA-256
  `a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154`).

Steps, from a Siliang Engine v0.1.8 checkout built with CMake:

```powershell
# 1. requantize the two layer-2 tensors from BF16 to IQ2_XS with the imatrix
python requant\requant_layer.py <siliang-engine-dir> blk2_gate_bf16.bin blk.2.ffn_gate_exps.weight imatrix_unsloth.gguf blk2_gate_iq2_xs.bin
python requant\requant_layer.py <siliang-engine-dir> blk2_up_bf16.bin blk.2.ffn_up_exps.weight imatrix_unsloth.gguf blk2_up_iq2_xs.bin

# 2. repack expert-major, swapping in the requantized parts (type 17 = IQ2_XS)
python <siliang-engine-dir>\tools\make_expert_major_gguf.py `
  --src Qwen3.8-Flash-Next-UD-Q2_K_XL-merged.gguf `
  --dst Qwen3.8-Flash-Next-UD-Q2_K_XL-uniform-iq2xs-siliang-expert-major.gguf `
  --replace-part blk.2.ffn_gate_exps.weight=17:blk2_gate_iq2_xs.bin `
  --replace-part blk.2.ffn_up_exps.weight=17:blk2_up_iq2_xs.bin `
  --verify --samples 256
```

`requant_layer.py` quantizes each expert with its own imatrix slice through
`ggml_quantize_chunk`, the same call `llama-quantize` uses.

## Validation

From a Siliang Engine checkout:

```powershell
python .\scripts\check_expert_major.py --model "<path>\Qwen3.8-Flash-Next-UD-Q2_K_XL-uniform-iq2xs-siliang-expert-major.gguf"
```

This artifact reports `expert-major-metadata-ok` with 48 packed tensors,
512 experts, part names `gate, up, down` and part type IDs `17, 20`. The
converter's verifier matched 256 random layer/expert slices byte-for-byte
against the source (and against the replacement files for layer 2). The
engine-side qualification for this model is in the Siliang Engine v0.1.8
release notes.

These checks do not constitute a general semantic-equivalence proof or replace
application-specific evaluation.

## Performance evidence

Single-system observations with Siliang Engine v0.1.8 (Windows 11, Ryzen 5
2600, 24 GB RAM, RTX 2070 8 GB, NVMe SSD on a PCIe 3.0 x4 link), 32k agent
profile above:

| Measurement | Result |
| --- | --- |
| Prefill, 29k-token prompt | 229 tok/s |
| Decode after that prompt | 7.9 tok/s |
| Decode on a short prompt | 8.9 tok/s |
| Decode in real agent sessions (sampling on, 5k-25k context) | 5.1-7.7 tok/s |
| 20-token follow-up / 900-token increment on a cached conversation | 2.0-2.5 s / about 16 s of prefill |

Short increments are bounded by SSD reads: a 900-token increment routes to
about 19,300 of the 24,576 (layer, expert) pairs, while a 12 GiB L2 holds 6,870
experts. These numbers are experimental single-system evidence, not a
performance guarantee.

## License

The Qwen3.8 Flash Next weights are licensed by Qwen under the Qwen Community
License 1.0, included in [`LICENSE`](LICENSE). This repository keeps that
license and its notices. The `requant/` script is part of the reproduction
recipe and is provided under the same terms as Siliang Engine (MIT).
