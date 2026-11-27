---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: atlas-ingine
tags:
- atlas-ingine
- qwen3.8
- qwen3_5
- quantization
- elastic-quantization
- nvfp4
- atlas3
- fp8
- cuda
- verified-ledger
---

# Qwen3.8-27B — `atlas-17g` (v2, for `atlas-ingine`)

## Overview

A **16.45 GB** (`15.33 GiB`, 4 shards) mixed-precision Elastic checkpoint of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (text-only, derived from [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)) for [**`Atlas3DSS/atlas-ingine`**](https://github.com/Atlas3DSS/atlas-ingine) (`main` branch). Targets **24 GB GPUs** where maximum attention precision (`FP8` across all layers) is preferred while saving **`3.45 GiB`** VRAM vs the `19.42 GiB` NVFP4 baseline.

- **Precision allocation (`atlas-17g`):** Boundary (4) MLP = `NVFP4`, Core MLP = `ATLAS3` (3.5 b/w), All Attention = `FP8`, `embed_tokens` / `lm_head` = `NVFP4` (group-16).
- **Weight shard bytes:** `16,451,453,828` bytes (SHA-256 pinned in `atlas-ingine` `native/src/backends/qwen38/qwen38_manifest.cpp` as `PinnedVerified`).
- **Scope:** Text-only checkpoint (`visual` and `mtp` tensors excluded).

## Measured Verification (`atlas-17g`)

- **Hardware:** NVIDIA Tesla V100-PCIE-32GB (`sm_70`, FP16 compute path)
- **Weights / Peak VRAM @ 4K:** `15.33 GiB` weights / `15.97 GiB` peak VRAM (`3.45 GiB` saved vs `19.42 GiB` NVFP4 baseline)
- **Teacher-forced parity vs `Qwen3.8-27B-NVFP4` (139 prompt tokens, 32 captured positions):** KLD mean `0.1640`, p99 `1.1045`, max `1.1378`, cosine mean `0.9731`, top-1 agreement `0.750` (`24/32`), all finite
- **Throughput (V100 SM70):** `10.71` prefill tok/s, `8.42` decode tok/s (`5.48 s` load)

## Qwen3.8-27B Elastic Pack Family

Baseline reference: `nvidia/Qwen3.8-27B-NVFP4` (`18.77 GiB` weights, `19.42 GiB` peak VRAM @ 4K).

| Pack | Precision Spec | Disk / Weights | Peak VRAM @ 4K | Saved vs Base | Teacher-Forced KLD Mean / P99 / Cos / Top-1 | Eval Hardware |
|---|---|---:|---:|---:|---|---|
| [**`Atlas3D/Qwen3.8-27B-atlas-17g-ingine`**](https://huggingface.co/Atlas3D/Qwen3.8-27B-atlas-17g-ingine) (v2) | Boundary(4) MLP=`nvfp4`, Core MLP=`atlas3`, All Attn=`fp8`, embed/lm_head=`nvfp4` | `16.45 GB` / `15.33 GiB` (4 sh) | `15.97 GiB` | `3.45 GiB` | `0.1640` / `1.1045` / `0.9731` / `0.750` | V100 (`sm_70`) |
| [**`Atlas3D/Qwen3.8-27B-atlas-14g-ingine`**](https://huggingface.co/Atlas3D/Qwen3.8-27B-atlas-14g-ingine) | Boundary(4) MLP=`nvfp4`, Core MLP=`atlas3`, Boundary Attn=`fp8`, Core Attn=`nvfp4`, embed/lm_head=`nvfp4` | `13.69 GB` / `12.75 GiB` (4 sh) | `12.64 GiB` | `6.02 GiB` | `0.1323` / `1.4008` / `0.9724` / `0.8125` | RTX PRO 6000 (`sm_120a`) |
| [**`Atlas3D/Qwen3.8-27B-atlas-11g-ingine`**](https://huggingface.co/Atlas3D/Qwen3.8-27B-atlas-11g-ingine) (v3) | Pure `atlas3+nvfp4` (zero `atlas2`): All MLP=`atlas3`, Boundary(2) Attn=`nvfp4`, Core Attn=`atlas3`, embed/lm_head=`nvfp4` | `12.18 GB` / `11.35 GiB` (3 sh) | `12.00 GiB` | `7.42 GiB` | `0.1415` / `0.8929` / `0.9553` / `0.781` | V100 (`sm_70`) |

## Quick Start (`atlas-ingine`)

### 1. Download and verify checksums
```bash
hf download Atlas3D/Qwen3.8-27B-atlas-17g-ingine --local-dir ./Qwen3.8-27B-atlas-17g-ingine
(cd ./Qwen3.8-27B-atlas-17g-ingine && sha256sum -c SHA256SUMS)
```

### 2. Build `atlas-ingine` (`main` branch)
```bash
git clone https://github.com/Atlas3DSS/atlas-ingine.git
cd atlas-ingine

# Ampere / Ada / Hopper / Blackwell (RTX 30/40/50, A100, H100, PRO 6000):
cmake -S native -B native/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="86;89;120a" -DINGINE_BUILD_APPS=ON
cmake --build native/build --target ingine-serve ingine-bench -j

# Volta V100 (sm_70, CUDA 12.9):
# cmake --preset legacy-fp16 -S native && cmake --build native/build-legacy-fp16 --target ingine-serve ingine-bench -j
```

### 3. Serve via OpenAI-compatible HTTP API
```bash
./native/build/apps/serve/ingine-serve \
  --engine \
  --model-dir ./Qwen3.8-27B-atlas-17g-ingine \
  --host 127.0.0.1 --port 8003 \
  --max-ctx 8192
```

## Provenance & Integrity

- Per-shard byte sizes and SHA-256 digests are listed in `atlas_elastic_manifest.json` and compiled into `native/src/backends/qwen38/qwen38_manifest.cpp`.
- Every file in this repository is covered by `SHA256SUMS`.
- Quantized with `tools/quantize_elastic_qwen38.py` in [`Atlas3DSS/atlas-ingine`](https://github.com/Atlas3DSS/atlas-ingine).

## Credits & License

- **Base model:** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)
- **Source quantized checkpoint:** [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) (Apache-2.0)
- **Inference engine:** [`Atlas3DSS/atlas-ingine`](https://github.com/Atlas3DSS/atlas-ingine) (Apache-2.0)
