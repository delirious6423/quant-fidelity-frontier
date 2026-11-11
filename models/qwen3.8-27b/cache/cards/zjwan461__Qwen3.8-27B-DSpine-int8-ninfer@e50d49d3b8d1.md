---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - multimodal
  - speculative-decoding
  - dspine
  - int8
  - cuda
  - rtx-4080
---

# Qwen3.8-27B + DSpine - NInfer int8 (community build)

A **community** build of the Qwen3.8-27B int8 NInfer artifact with Tencent's DSpine draft model
embedded as a fourth component. The engine, its artifact format and the quantization recipe belong
to [NInfer](https://github.com/Neroued/ninfer) (Neroued); this repository only redistributes a
converted artifact and is not affiliated with Alibaba or Tencent.

`qwen3_8_27b_dspine.ninfer` is a **NInfer artifact**: the native weight container of the NInfer
single-GPU inference engine. It is consumed by `ninfer.exe` / `ninfer-serve.exe`; it is **not** a
`transformers`, `safetensors` or `sglang` checkpoint, and no Python runtime can load it.

> [!IMPORTANT]
> **These weights only run with the engine built from
> [`zjwan461/ninfer-sm89-windows`](https://github.com/zjwan461/ninfer-sm89-windows), branch
> `main`.** Upstream NInfer does not contain the DSpine draft backend, so the upstream
> engine cannot load the `dspine` component. No other runtime (transformers, sglang, llama.cpp,
> vLLM) can read the `.ninfer` container at all - without this repository the file is unusable.
> The numbers below come from commit `1455d43a` (same tree as the merge).

The text, vision and MTP components were imported byte for byte from one local build of the int8
recipe (SHA-256 `49bf76388e139defbe416c8cc0079abc1c78b59be2a41ff56ae960aeacf5fc19`); the same bytes are published in the sibling
[DFlash2 repository](https://huggingface.co/zjwan461/Qwen3.8-27B-DFlash2-int8-ninfer), so results on the two repositories are directly comparable. That
build has the same layout as
[neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) (1190 objects,
1513 bindings) but different quantized values: the published upstream file is
`81f924d440c27261...`, so this artifact is *not* a byte-identical copy of it. The DSpine draft was
converted from
[tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16).

| File | Size | SHA-256 |
|---|---:|---|
| `qwen3_8_27b_dspine.ninfer` | 20 639 484 160 B (19.22 GiB) | `b90e49707977977749e75c2a25389c9163b1fce17a1d71dde65704808e5ce759` |
| `qwen3_8_27b_dspine.ninfer.conversion.json` | 550 702 B | listed in `SHA256SUMS` |
| `artifact-manifest.json` | inventory and provenance | - |
| `LICENSE` | Apache-2.0 | - |

## Components

| Component | Contents | Storage |
|---|---|---|
| `text` | 64-layer Qwen3.8-27B text backbone (24 Q heads, 4 KV heads, head_dim 256, hybrid linear/full attention) | Q8 (g32) for most projections, Q5/Q6 for a few, mixed per recipe |
| `vision` | 27-layer Qwen3.5 vision tower + merger | Q4 |
| `mtp` | multi-token-prediction head (single layer + shared head) | mixed |
| `dspine` | 5-layer DFlash-shaped draft, block 16, sliding window 8192, code-bridge rank 512 / code size 1024, head top-k 16 | Q8 backbone, code bridge in BF16 |

The conversion report records every tensor's source file, format and quantization parameters.

## Usage

Build the engine first. This is mandatory: the artifact cannot be served by anything else.

```bat
git clone https://github.com/zjwan461/ninfer-sm89-windows
cd ninfer-sm89-windows
git switch feat/dspine-draft
:: configure and build with CMake + MSVC 2022 + CUDA 13.x, then use build*/apps/ninfer.exe
```

Then run:

```bat
ninfer.exe qwen3_8_27b_dspine.ninfer ^
  --prompt "Write a Python function that merges two sorted lists." ^
  --max-new 160 --greedy --no-thinking --kv-dtype rk4v4-e8 ^
  --spec dspine --draft-tokens 15
```

`--spec dspine` selects the DSpine draft; `--spec mtp` uses the MTP head instead. `--spec dflash2`
is **not** available in this file because no DFlash2 component is embedded (see Limitations).

Serving:

```bat
ninfer-serve.exe qwen3_8_27b_dspine.ninfer --host 127.0.0.1 --port 8080 ^
  --max-context 262144 --max-concurrency 2 --kv-dtype rk4v4-e8 ^
  --spec dspine --draft-tokens 15
```

The artifact was built and tested on `sm_89` (RTX 4080 SUPER, 32 GB) with the repository's
`build-4080s` configuration. Other architectures need their own build; the file format itself is
architecture independent.

## Measured behaviour

RTX 4080 SUPER 32 GB (sm_89, 80 SM build), `--kv-dtype rk4v4-e8`, greedy, thinking off,
CUDA Graphs on, one prompt per row. Indicative numbers, not a benchmark suite. "Accepted length"
is the accepted tokens per verification round, the round's bonus token included.

| Prompt | DSpine `--draft-tokens 15` | DFlash2 `--draft-tokens 8` | DFlash2 `--draft-tokens 15` |
|---|---:|---:|---:|
| merge-sorted function | 10.60 | 7.95 | 10.53 |
| HumanEval-style completion | 9.15 | 7.93 | 10.82 |
| grade-school word problem | 5.54 | 5.54 | 6.54 |

Decode throughput across those prompts ranges from about 120 tok/s (the word problem, where the
accepted length is low) to 233 tok/s (the code prompts). The DFlash2 columns come from a separate
artifact ([the DFlash2 sibling](https://huggingface.co/zjwan461/Qwen3.8-27B-DFlash2-int8-ninfer)) that carries the same text weights.

The DSpine draft matches an independent host implementation of the reference mathematics:
per-layer bridge `rms` within `6e-5`, `codes`/`packed`/`out`/`next` bit-identical for 98.9-99.99% of
elements, identical draft tokens, and exact causal-prefix stability between `--draft-tokens 7`
and `15`.

## Limitations

- **No DFlash2 component.** Only `text`, `vision`, `mtp` and `dspine` are embedded, so
  `--spec dflash2` needs the [DFlash2 sibling repository](https://huggingface.co/zjwan461/Qwen3.8-27B-DFlash2-int8-ninfer).
- **Quantized draft.** The draft backbone is Q8 (the upstream DSpine evaluation is BF16), and the
  target features and output head are Q8 as well. On reasoning-style prompts this can cost DSpine
  more than DFlash2, because the code bridge re-encodes the target residual at every layer.
- **Equal-width comparison.** DSpine is not universally ahead: at equal draft width DFlash2 reaches
  a similar or slightly higher accepted length on the prompts above. The DSpine advantage appears
  at the published comparison point (DSpine 16 vs DFlash2 8).
- **Runtime required.** The weights are useless without the engine from
  [`zjwan461/ninfer-sm89-windows`](https://github.com/zjwan461/ninfer-sm89-windows)
  (branch `feat/dspine-draft`); build it with CUDA 13.x + MSVC 2022 for `sm_89` or `sm_120a`.
- **DSpine support is recent.** The draft backend is not in upstream NInfer, so a build from the
  upstream repository cannot load this file's `dspine` component.

## License and attribution

Apache-2.0, following the upstream licenses of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
[tencent/Qwen3.8-27B-DSpine-b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16) and
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (all declared
Apache-2.0 at the time of conversion).

This is an **unofficial** quantization and container conversion. Architecture and draft-model
research credit belongs to the DSpine and DFlash2 authors; the runtime, artifact format and
quantization belong to the NInfer project.