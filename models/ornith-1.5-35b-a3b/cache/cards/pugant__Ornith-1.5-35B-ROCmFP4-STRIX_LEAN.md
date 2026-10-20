---
library_name: llama.cpp
license: mit
base_model: ornith-ai/Ornith-1.5-35B
tags:
  - rocmfpx
  - gfx1151
  - strix-halo
  - qwen35moe
  - moe
  - rocm
  - amdgpu
  - ROCmFP4
pipeline_tag: image-text-to-text
language:
  - en
  - multilingual
version: '1.5'
date: 2026-08-20
quantized_by: pugant
---

# Ornith-1.5-35B — ROCmFP4-STRIX_LEAN (Strix Halo / gfx1151)

> Version 1.5 — 2026-08-20

## TL;DR

`Ornith-1.5-35B` (35B params, 3B active per token, Qwen3.5-VL-MoE family) quantized to **`Q4_0_ROCMFP4_STRIX_LEAN`** (type 106 preset, ~4.29 BPW). Tuned for **AMD Strix Halo (gfx1151 / RDNA 3.5)** on the **ROCmFPX fork family** — we serve and benchmark these files on **our lab runtime** (full source: [pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa/tree/main/rocmfpx); upstream: [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX)). Runs the full vision + text multimodal model in ~17.7 GiB.

## ⚠️ Critical warnings — read before downloading

- **Requires a ROCmFPX fork build** — see [Runtime](#runtime). Upstream [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) loads these files too (any build with the custom GGML types, e.g. built via the `kyuz0/amd-strix-halo-toolboxes` container). The type 106 (`Q4_0_ROCMFP4_STRIX_LEAN`) tensor format is **INVALID in stock llama.cpp** — it will refuse to load. See [Usage](#usage) below.
- Profiled for **gfx1151 only** (Strix Halo / Ryzen AI Max+ 395, RDNA 3.5). Not tested on other GPUs.
- FP4 here is **software** on RDNA 3.5 (no FP4 silicon units): the win is **bandwidth / memory**, not raw compute throughput. The Strix Halo ceiling on this MoE is bandwidth-bound, which is exactly where FP4 helps.

## Benchmarks

Tested on Strix Halo (AMD Ryzen AI Max+ 395, 128 GB LPDDR5X). Methodology: `llama-bench -ngl 999 -fa on -p 512 -n 128`, **Vulkan (RADV) build** of the fork, GPU-exclusive window (production service stopped). Note: the [Ornith-1.0 release](https://huggingface.co/pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN) numbers below were measured on the ROCm (HIP) backend — different backend, not directly comparable.

| Model | Quant | Size | tg128 (tok/s) | pp512 (tok/s) |
|---|---|---:|---:|---:|
| **Ornith-1.5-35B (this)** | **ROCmFP4-STRIX_LEAN** | **17.73 GiB** | **82.45 ± 1.37** | **1161.00 ± 22.76** |
| Ornith-1.0-35B (ROCm) | ROCmFP4-STRIX_LEAN | 17.32 GiB | 66.68 | 1486 |
| grug-35b-v2 (ROCm) | ROCmFP4-STRIX_LEAN | 17.31 GiB | 70.92 | 1418 |
| Qwen3.6-35B-A3B base (Vulkan ref) | ROCmFP4-STRIX_LEAN | 17.73 GiB | 81.57 | 1164.67 |

Sanity check: Ornith-1.5 tracks its base-architecture sibling Qwen3.6-35B-A3B (same quant, same backend) within 1% — 82.45 vs 81.57 tg128. On this Vulkan build Ornith-1.5 decodes **23.7% faster than Ornith-1.0 on ROCm** at the same footprint class.

### Speculative decoding (MTP)

The source model ships a 1-layer MTP head (`nextn_predict_layers=1`, tensors `blk.40.nextn.*`), **included in this GGUF and activated at runtime only if you opt in** with the fork's `--spec-type draft-mtp` flags.

Measured on the same GPU-exclusive window, same server/prompt methodology (2 prompts × 2 runs, ctx 16k, Vulkan build of the fork):

| Config | prose (tok/s) | deterministic (tok/s) |
|---|---:|---:|
| **plain (no spec)** | **78.0** | **77.2** |
| MTP n-max 2 | 59.1 | 75.4 |
| MTP n-max 3 | 44.6 | 63.5 |
| MTP n-max 5 | 36.9 | 47.3 |

**Verdict: speculative decoding does not pay off on Ornith-1.5** — plain inference wins at every n-max (−24% prose at the best MTP setting). The measured draft acceptance explains why: position-1 acceptance is high (0.99 on deterministic tasks) but position-2 collapses to ~0.07, so the mean accepted length (1.4–1.7) never covers the draft+verify cost — the nextn layer is a full MoE layer. For comparison, on the same stack the Qwen3.6-35B-A3B base model accepts (0.87, 0.77, 0.64) and gains +37% with MTP n-max 3: **the 1.5 fine-tune degraded the MTP head beyond the first drafted token.** If you still want to experiment, use `--spec-draft-n-max 2`; above that it is pure overhead. MTP stays opt-in: with no spec flags the model runs plain inference at the headline speeds above.

### System configuration at bench time

Declared for reproducibility:

- **Bare metal host**: Bosgame BeyondMax Series, Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic
- **CPU power profile**: `balanced` (`powerprofilesctl get`) — default, **NOT forced to `performance`**. Representative of an out-of-the-box setup.
- **CPU scaling driver**: `amd-pstate-epp`, scaling_governor `performance` (amd-pstate-epp default), EPP `performance`
- **IOMMU / iGPU power**: auto (no manual tuning)

Note: tok/s above were measured on a non-tuned system (power profile `balanced`). Users who set `powerprofilesctl set performance` may see slightly higher numbers.

## Quantization details

Preset `Q4_0_ROCMFP4_STRIX_LEAN` (GGUF file_type 106, ~4.29 bits/weight):

- **Token embeddings** (`token_embd.weight`) → `Q5_K` (preserve vocab fidelity)
- **Attention K/V** (`blk.*.attn_qkv.weight`, `blk.*.attn_v.weight`) → `q4_0_rocmfp4` (high-precision path for attention state)
- **Expert FFN** (`blk.*.ffn_*_exps.weight`) → `q4_0_rocmfp4_fast` (max speed path; the bulk of MoE weights)
- MTP head (`blk.40.nextn.*`) kept in the file (BF16/F32 as in source)
- Zero quantization fallbacks across all 753 tensors

Reference fork: [`charlie12345/ROCmFPX`](https://github.com/charlie12345/ROCmFPX) (MIT).

Serving runtime: see [Runtime](#runtime).

## imatrix methodology

Precomputed by **bartowski** on 573 chunks (`calibration-v6` dataset), redistributed here as `imatrix-Ornith-1.5-35B-bartowski.gguf` with explicit attribution. The original is at [`bartowski/Ornith-1.5-35B-A3B-GGUF`](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF) (MIT). Unlike the Ornith-1.0 release, which used the unsloth imatrix computed on 1.0 weights.

## Files

| File | Size | Description |
|---|---:|---|
| `Ornith-1.5-35B-ROCmFP4-STRIX_LEAN.gguf` | ~17.73 GiB | Main model (type 106), MTP head included |
| `mmproj-Ornith-1.5-35B-BF16.gguf` | ~860 MB | Vision projector (BF16, from the official repo) |
| `imatrix-Ornith-1.5-35B-bartowski.gguf` | ~183 MB | Importance matrix (precomputed by bartowski; for re-quantization) |

## Usage

```bash
# Requires the kyuz0 Strix Halo toolbox (which builds charlie12345/ROCmFPX)
docker run --rm -p 1234:1234 --device /dev/kfd --device /dev/dri \
  -v /path/to/models:/models rocmfpx-llm-service \
  llama-server \
    -m /models/Ornith-1.5-35B-ROCmFP4-STRIX_LEAN.gguf \
    --mmproj /models/mmproj-Ornith-1.5-35B-BF16.gguf \
    -ngl 999 -fa on --jinja -c 32768 --host 0.0.0.0 --port 1234
```

Notes:
- **MTP is opt-in at runtime.** Add `--spec-type draft-mtp --spec-draft-ngl all --spec-draft-p-min 0.0 --spec-draft-p-split 0.10 --spec-draft-n-max <N>` to activate speculative decoding (see the MTP section above for measured n-max guidance). Plain inference (no spec flags) is the default and what the headline benchmark table reports.
- The `--mmproj` flag is required for the vision tower (multimodal). Without it, text-only still works.

## How to replicate

1. Build the `docker-llm-service-convert` image from `kyuz0/amd-strix-halo-toolboxes` + `charlie12345/ROCmFPX` (must contain `MODEL_ARCH.QWEN35MOE`).
2. Download the BF16 GGUF from [`ornith-ai/Ornith-1.5-35B-A3B-GGUF`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF).
3. Quantize with the bartowski imatrix: `llama-quantize --imatrix imatrix-Ornith-1.5-35B-bartowski.gguf <bf16>.gguf <out>.gguf Q4_0_ROCMFP4_STRIX_LEAN 16`.

## Attribution & model tree

```
Qwen3.5-VL-MoE (base architecture)
    └── ornith-ai/Ornith-1.5-35B (A3B) (MIT)
            └── this GGUF (ROCmFP4-STRIX_LEAN)
```

- Base model: [`ornith-ai/Ornith-1.5-35B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B) (MIT)
- BF16 GGUF + mmproj source: [`ornith-ai/Ornith-1.5-35B-A3B-GGUF`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) (MIT)
- imatrix: [`bartowski/Ornith-1.5-35B-A3B-GGUF`](https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF) (MIT)
- Quantization fork: [`charlie12345/ROCmFPX`](https://github.com/charlie12345/ROCmFPX) (MIT)
- Container runtime: [`kyuz0/amd-strix-halo-toolboxes`](https://github.com/kyuz0/amd-strix-halo-toolboxes)

## License

MIT (inherited from `ornith-ai/Ornith-1.5-35B` and its GGUF release). Derivative work: original model and its license are preserved. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

## Acknowledgements

Built on the shoulders of giants:

- [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) — Strix Halo container runtime
- [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) — llama.cpp fork with ROCmFP4 presets (type 106)
- [bartowski](https://huggingface.co/bartowski) — precomputed imatrix for Ornith-1.5-35B
- [ornith-ai / DeepReinforce Team](https://huggingface.co/ornith-ai) — Ornith-1.5-35B
- [llama.cpp](https://github.com/ggerganov/llama.cpp) community + Kawrakow (imatrix methodology)
- Hardware: Bosgame BeyondMax Series (Strix Halo bare metal host)

## Limitations & community feedback

- **Speed benchmark only.** No perplexity / MMLU / quality eval is included in this release. The MoE structure is preserved from the BF16 source except for the quantized tensor formats above; quality is expected to track standard Q4_K_M-class with the ROCmFP4 attention/K-V choices, but this is not measured here.
- Profiled for **gfx1151 only**. Not tested on other GPUs (feel free to share yours).
- MTP head present in weights; speculative decoding is opt-in at runtime (plain inference by default). No separate drafter head exists for Ornith-1.5 as of release date.

**We invite the community — especially fellow Strix Halo owners — to test and share quality results. Open a Discussion on this repo.**

## Citation

```bibtex
@misc{ornith152026,
  title  = {Ornith-1.5-35B},
  author = {DeepReinforce Team},
  year   = {2026},
  url    = {https://huggingface.co/ornith-ai/Ornith-1.5-35B}
}
```

## Disclaimer

No affiliation with AMD, Qwen, DeepReinforce, bartowski, unsloth, kyuz0, or charlie12345. Provided as-is, without warranty. Users must comply with the base model license (MIT).

---

## Runtime
**Benchmark environment:** one bare-metal AMD Strix Halo (Ryzen AI MAX+ 395, 128 GB) — full dated configuration and measurement policy: [BARE-METAL.md](https://github.com/pugant/strix-nebulosa/blob/main/BARE-METAL.md)

Requires a ROCmFPX fork build (custom tensor types — stock llama.cpp refuses the file).
**Recommended: our lab build** ([pugant/strix-nebulosa](https://github.com/pugant/strix-nebulosa), `main`) —
reasoning budget and persistent prompt cache on every model; drafter features
where the model ships one: see the engine section of its README.

Honest note: the fine-tune degraded the MTP head (pos-2 acceptance ~0.07) — run it plain, skip the drafter.

> Everything here is experimental and provided as-is, at your own risk.
