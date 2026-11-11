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
  - dflash2
  - int8
  - cuda
  - rtx-4080
---

# Qwen3.8-27B + DFlash2 - NInfer int8 (community build)

A **community** build of the Qwen3.8-27B int8 NInfer artifact whose speculative backend is z-lab's
DFlash2 draft. The engine, its artifact format and the quantization recipe belong to
[NInfer](https://github.com/Neroued/ninfer) (Neroued); this repository only redistributes a
converted artifact and is not affiliated with Alibaba or z-lab.

`qwen3_8_27b_a8.ninfer` is a **NInfer artifact**: the native weight container of the NInfer
single-GPU inference engine. It is consumed by `ninfer.exe` / `ninfer-serve.exe`; it is **not** a
`transformers`, `safetensors` or `sglang` checkpoint, and no Python runtime can load it.

> [!IMPORTANT]
> **These weights only run with the NInfer engine.** No other runtime (transformers, sglang,
> llama.cpp, vLLM) can read the `.ninfer` container. DFlash2 is supported by upstream NInfer; the
> build used for the numbers below is
> [`zjwan461/ninfer-sm89-windows`](https://github.com/zjwan461/ninfer-sm89-windows), branch
> `main`, commit `1455d43a` (same tree as the merge).

The text, vision and MTP components are the same bytes as the sibling
[`zjwan461/Qwen3.8-27B-DSpine-int8-ninfer`](https://huggingface.co/zjwan461/Qwen3.8-27B-DSpine-int8-ninfer)
artifact (both were converted from one local build of the int8 recipe), so DFlash2 and DSpine
results on those two repositories are directly comparable.

| File | Size | SHA-256 |
|---|---:|---|
| `qwen3_8_27b_a8.ninfer` | 20 437 521 664 B (19.03 GiB) | `49bf76388e139defbe416c8cc0079abc1c78b59be2a41ff56ae960aeacf5fc19` |
| `artifact-manifest.json` | inventory and provenance | - |
| `LICENSE` | Apache-2.0 | - |

## Components

| Component | Contents | Storage |
|---|---|---|
| `text` | 64-layer Qwen3.8-27B text backbone (24 Q heads, 4 KV heads, head_dim 256, hybrid linear/full attention) | Q8 (g32) for most projections, Q5/Q6 for a few, mixed per recipe |
| `vision` | 27-layer Qwen3.5 vision tower + merger | Q4 |
| `mtp` | multi-token-prediction head (single layer + shared head) | mixed |
| `dflash2` | DFlash2 masked draft with the fused context-KV materializer | Q8 backbone |
| `proposal` | indexed shortlist head (131 072 rows) used by `--lm-head-draft` | Q4 |

## Usage

```bat
ninfer.exe qwen3_8_27b_a8.ninfer ^
  --prompt "Write a Python function that merges two sorted lists." ^
  --max-new 160 --greedy --no-thinking --kv-dtype rk4v4-e8 ^
  --spec dflash2 --draft-tokens 7
```

`--spec dflash2` selects the DFlash2 draft; `--spec mtp` uses the MTP head instead. `--spec dspine`
is **not** available in this file (no DSpine component; see the sibling repository).

Serving:

```bat
ninfer-serve.exe qwen3_8_27b_a8.ninfer --host 127.0.0.1 --port 8080 ^
  --max-context 262144 --max-concurrency 2 --kv-dtype rk4v4-e8 ^
  --spec dflash2 --draft-tokens 7
```

## Measured behaviour

RTX 4080 SUPER 32 GB (sm_89), `--kv-dtype rk4v4-e8`, greedy, thinking off, CUDA Graphs on, one
prompt per row. Accepted length is the accepted tokens per verification round, the bonus token
included.

| Prompt | DFlash2 `--draft-tokens 7-8` | DFlash2 `--draft-tokens 15` |
|---|---:|---:|
| merge-sorted function | 7.95 | 10.53 |
| HumanEval-style completion | 7.93 | 10.82 |
| grade-school word problem | 5.54 | 6.54 |

Decode throughput is 220-230 tok/s on the code prompts. The DSpine comparison at the published
block sizes (DSpine 16 vs DFlash2 8) lives in the sibling repository.

## Limitations

- **No DSpine component.** `--spec dspine` needs the sibling artifact.
- **Engine required.** The weights are useless without an NInfer build that supports DFlash2
  (`sm_89` / `sm_120a`, CUDA 13.x, MSVC 2022 on Windows).
- **Not byte-identical to `neroued/Qwen3.8-27B-NInfer`.** This file is a separate build of the same
  recipe: same layout (1190 objects, 1513 bindings) but different quantized values, hence a
  different SHA-256. Use that repository if you need the exact published upstream bytes.

## License and attribution

Apache-2.0, following the upstream licenses of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (both declared
Apache-2.0 at the time of conversion).

This is an **unofficial** quantization and container conversion. Draft-model research credit
belongs to the DFlash2 authors; the runtime, artifact format and quantization belong to the NInfer
project.