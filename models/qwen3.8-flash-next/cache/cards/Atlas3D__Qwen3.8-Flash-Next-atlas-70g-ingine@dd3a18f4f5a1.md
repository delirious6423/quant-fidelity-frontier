---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: atlas-ingine
tags:
- atlas-ingine
- qwen3.8
- qwen3.8-flash-next
- qwen4exp
- moe
- quantization
- elastic-quantization
- nvfp4
- atlas3
- fp8
- cuda
- verified-ledger
---

# Qwen3.8-Flash-Next (177B) — `atlas-70g` (for `atlas-ingine`)

## Overview

An Elastic safetensors checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`qwen4exp`, 177B total MoE, 48 layers, 512 experts / top-10, 4-way Hyper-Connections, host-only Per-Layer Embeddings) for [**`Atlas3DSS/atlas-ingine`**](https://github.com/Atlas3DSS/atlas-ingine) (`main` branch). GPU weights occupy **`73.06 GB`** (49 shards) plus **`28.80 GB`** host-only NVFP4 PLE shards (81 shards, `0 GB VRAM`), sized to fit `AllResident` in VRAM on 96 GB GPUs or run via `TieredExpertCache` on 24–32 GB GPUs.

- **Precision allocation (`atlas-70g`):** All MoE = `NVFP4`, Dense = `FP8`, embed/output = `NVFP4`, Host-only PLE (`320,001,536 x 160`) = `NVFP4`.
- **Footprint:** 101,859,259,580 bytes total (`73,059,098,352` B GPU weights / 49 shards + `28,800,161,228` B host-only PLE / 81 shards).
- **Manifest SHA-256 (`atlas_elastic_manifest.json`):** `b8b0f6f7424251ada165cb4ed0a2da1ae4f2d1942195f0d413024a0fc6c3fca9`.
- **Scope:** Text-only checkpoint (vision `mmproj` excluded). Host-only Per-Layer Embeddings (`per_layer_token_embd.weight`, `320,001,536 x 160`) stay in host RAM/mmap (`0 GB VRAM`).

## Measured Verification (`atlas-70g`)

- **Hardware:** NVIDIA Tesla V100-PCIE-32GB (`sm_70`, `TieredExpertCache`: `4.43 GiB` dense + `25.28 GiB` expert cache, `9,819` slots)
- **Parity vs exact FP32 `UD-Q6_K_XL` oracle (48-layer forward pass):** `cosine = 0.97572218`, `top1 = true` (`11751 = ' Paris'`), `max_abs = 2.9197`, `rmse = 0.5587`
- **Warm-cache greedy decode (8 tokens, V100 `TieredExpertCache`):** `6.29 tok/s` (`3.03 s` load, `1.44 s` 5-token prefill)
- **HTTP serving:** Shared `qwen4exp-cuda` serving path (live `ingine-serve` HTTP smoke verified on `atlas-56g`)

## Known Limitations & Disclosures

1. **Real 177B SM120a `AllResident` speed is unmeasured:** All real-177B GPU evaluations for this release were executed on SM70 V100-PCIE-32GB (`TieredExpertCache`, `25.28 GiB` expert cache). Current SM120a Blackwell verification for the `qwen4exp` backend covers synthetic `qwen4exp_cuda_ops` and `qwen4exp_gpu_e2e` (`AllResident` and `TieredExpertCache`), not a real 177B SM120a `AllResident` throughput benchmark.
2. **`atlas-42g` is held back:** An experimental `atlas-42g` profile using unweighted 2.5-bit `ATLAS2` shifted top-1 on factual completion probes and is intentionally excluded from publication pending `ffn_down_exps=atlas3` or Hessian/DFly-SFT calibration.
3. **Single-GPU backend (`INGINE_TP_SIZE=1`):** The native `qwen4exp-cuda` backend currently runs single-GPU (`AllResident` when GPU weights fit in VRAM, or `TieredExpertCache` on 24–32 GB GPUs) and does not yet support `INGINE_TP_SIZE > 1`.

## Qwen3.8-Flash-Next (177B) Released Elastic Packs

Source checkpoint: `unsloth/Qwen3.8-Flash-Next-GGUF` `UD-Q6_K_XL` (revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`, `158 GB`).  
Evaluated on SM70 V100 (`TieredExpertCache`, `25.28 GiB` expert cache, `9,819` slots) against exact FP32 `UD-Q6_K_XL` oracle logits.

| Pack | Precision Spec | GPU Weights / Host PLE | vs FP32 Q6_K_XL Oracle (`cos` / `top1`) | Warm-Cache V100 8-tok Greedy |
|---|---|---|---|---|
| [**`Atlas3D/Qwen3.8-Flash-Next-atlas-70g-ingine`**](https://huggingface.co/Atlas3D/Qwen3.8-Flash-Next-atlas-70g-ingine) | All MoE=`nvfp4`, Dense=`fp8`, embed/output=`nvfp4` | `73.06 GB` (49 sh) / `28.80 GB` (81 sh) | `0.97572218` / `true` | `6.29 tok/s` |
| [**`Atlas3D/Qwen3.8-Flash-Next-atlas-56g-ingine`**](https://huggingface.co/Atlas3D/Qwen3.8-Flash-Next-atlas-56g-ingine) | Boundary(L0–1,L46–47) MoE=`nvfp4`, Core MoE=`atlas3`, Dense=`fp8`, embed/output=`nvfp4` | `59.22 GB` (49 sh) / `28.80 GB` (81 sh) | `0.97220816` / `true` | `6.12 tok/s` |

## Quick Start (`atlas-ingine`)

### 1. Download and verify checksums
```bash
hf download Atlas3D/Qwen3.8-Flash-Next-atlas-70g-ingine --local-dir ./Qwen3.8-Flash-Next-atlas-70g-ingine
(cd ./Qwen3.8-Flash-Next-atlas-70g-ingine && sha256sum -c SHA256SUMS)
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
  --model-dir ./Qwen3.8-Flash-Next-atlas-70g-ingine \
  --host 127.0.0.1 --port 8003 \
  --max-ctx 8192
```

## Provenance & Integrity

- All 130 shards (49 GPU weight shards + 81 host-only PLE shards) are listed with byte sizes and SHA-256 digests in `atlas_elastic_manifest.json`.
- Every file in this repository is covered by `SHA256SUMS`.
- Quantized with `tools/quantize_elastic_qwen4exp.py` in [`Atlas3DSS/atlas-ingine`](https://github.com/Atlas3DSS/atlas-ingine).

## Credits & License

- **Base model:** [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — licensed under **Qwen Community License 1.0** (full text in [`LICENSE`](LICENSE)).
- **Source GGUF checkpoint:** [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (`UD-Q6_K_XL`, revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`).
- **Inference engine:** [`Atlas3DSS/atlas-ingine`](https://github.com/Atlas3DSS/atlas-ingine) (Apache-2.0).
