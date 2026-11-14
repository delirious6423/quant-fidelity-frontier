---
license: other
license_name: wirkung-model-and-runtime-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: wirkung
pipeline_tag: text-generation
num_parameters: 27800000000
language:
- en
- de
tags:
- wirkung
- origami
- origami-1.1
- o3f
- constant-memory
- cpu-inference
- mtp
---

# origami1.1_27b

Weight & constant-state update succeeding [`wirkung/Origami-27B`](https://huggingface.co/wirkung/Origami-27B) (`v1.0`). Compiled with `0` retraining steps from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).

## Lineage

| Component | Source |
|:--|:--|
| Predecessor (`v1.0`) | [`wirkung/Origami-27B`](https://huggingface.co/wirkung/Origami-27B) |
| Base model (`64` layers + native `MTP`) | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |

## Metrics (`v1.0` -> `v1.1`)

| Metric | `v1.0` | `v1.1` (`origami1.1_27b`) |
|:--|:--:|:--:|
| Bundle Size | `7.44 GB` | **`11.728 GB`** |
| End-to-End `ΔNLL` vs Base | `+0.4268` | **`+0.01147`** (`37.2x` lower) |
| End-to-End Top-1 Agreement vs Base | `76.17%` | **`93.74%`** (`+17.57 pp`) |
| Constant-State `ΔNLL` vs Exact (`1,024` tok) | `—` | **`+0.01827`** (`KL = 0.02009`, `96.68%` top-1) |
| Constant-State Passkey Recall (`T=640`, `T=2048`) | `—` | **`2/3`** (`3/3` exact mode) |
| Live KV Allocation Growth (`T=1K..128K`) | `0.0 B/tok` | **`0.0 B/tok`** (`ΔRSS = 0.0 MB`) |
| CPU Decode (`x86_64 AVX2`, 16 threads, constant-state) | `—` | **`4.74 tok/s`** (`15.10 tok/s` prefill · **`6.97 tok/s`** MTP) |

## Contents

- `layers/layer-000`..`layers/layer-063`: `400` `.o3f` operator weights
- `vocab/`: vocabulary & projection weights
- `mtp/`: speculative MTP module
- `assets/o1_crystal_v2_27b.bin`: constant-memory state sidecar
- `exact_core.safetensors`, `bundle.json`, `config.json`, tokenizer files, `LICENSE`, `NOTICE`

## Download

```bash
huggingface-cli download wirkung/origami1.1_27b --local-dir ./origami1.1_27b
```
