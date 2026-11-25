---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
pipeline_tag: image-text-to-text
tags:
  - deepseek
  - deepseek-v4.1
  - tensorfold
  - metal
  - apple-silicon
  - m5-ultra
  - lossless
  - engram
  - mxfp4
  - fp8
  - safetensors
---

# DeepSeek-V4.1-Flash-1017 (TensorFold Metal 4.1 Lossless Repack)

A **100% bit-exact, lossless repack** of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/08c8657f02291e11b6f6b63e0e16014f8e748518) (pinned commit `08c8657f02291e11b6f6b63e0e16014f8e748518`) prepared for zero-copy Metal 4.1 `mmap` serving on 512 GB Apple Silicon Macs using [`swbiggart/ds41-1017-studio`](https://github.com/swbiggart/ds41-1017-studio) (TensorFold `v1.3.3` + M5 Ultra Metal 4.1 patch stack).

Prepared by Scott Biggart · [@woolsey](https://x.com/woolsey) · Upstream weights by [DeepSeek AI](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) under the [MIT License](LICENSE).

---

## Why this repack exists

Serving the original checkpoint locally normally requires **~715 GB of peak SSD space** during download, Engram packing, and repacking. This repository bundles the finished, verified artifacts together (**~475.25 GiB / 510.3 GB total**) so you can download once and serve immediately — skipping `fetch`, `prepare`, and `repack`:

- **Zero quantization / zero weight modification:** Every single one of the **96,081 retained tensors** is copied verbatim byte-for-byte from DeepSeek's pinned checkpoint — keeping compact FP8 `wo_a`, compact 32×32 dense scales, exact MXFP4 routed experts, DSpark `mtp.0..2`, and vision weights untouched.
- **Zero-copy Metal 4.1 `mmap` layout (`286.42 GiB`, 45 shards):** Re-sharded by consumer (`model-top`, `model-layer-00..39`, `model-mtp-0..2`, `model-vision`) with **16 KiB page-aligned headers** and **128-byte aligned tensor offsets** (eliminating unaligned shard copies on Metal and dropping copied weight arenas from `7,789 MiB` to `1,567 MiB`, of which `1,263 MiB` is the reversible packed vocabulary head).
- **Pre-packed DSV41EN1 Engram tables (`engram/`, `188.83 GiB`):** Ready for TensorFold's Layer-1 host-RAM + Layer-14 overlapped SSD Engram streaming, omitting the 4 redundant raw `.engram.embed.*` tensors from the safetensors shards (`~203 GB` saved vs keeping both).
- **Pre-built Engram token map (`assets/`, `506 KiB`):** Verified against the pinned `tokenizer.json` and `config.json`.

---

## Repository layout

| Path | Size | Contents |
|---|---:|---|
| `model-top.safetensors` | 2.47 GiB | Embedding, final norm, LM head (3 tensors) |
| `model-layer-00..39.safetensors` | 275.66 GiB | 40 transformer backbone layers, one shard per layer |
| `model-mtp-0..2.safetensors` | 7.39 GiB | 3 DSpark / MTP speculative drafting stages |
| `model-vision.safetensors` | 0.90 GiB | Vision encoder & aligner weights (266 tensors) |
| `model.safetensors.index.json` | 6.6 MiB | Safetensors index for all 96,081 tensors |
| `repack-manifest.json` | 12 KiB | Schema v1 manifest with per-shard sizes, alignments & SHA-256s |
| `engram/engram-l1-r0of1.bin` | 94.42 GiB | Packed Layer-1 DSV41EN1 Engram table |
| `engram/engram-l14-r0of1.bin` | 94.42 GiB | Packed Layer-14 DSV41EN1 Engram table |
| `engram/engram-manifest.json` | 1.6 KiB | Engram table manifest & SHA-256 verification metadata |
| `assets/engram-token-map.bin` (+ `.json`) | 506 KiB | Precomputed Engram token map & checksums |
| `config.json`, `tokenizer*`, `chat_template.jinja`, `encoding/`, `inference/`, `LICENSE` | ~6.2 MiB | Verbatim upstream configuration, tokenizer, reference code & MIT license |

---

## Quick start (512 GB M5 Ultra Mac Studio)

Requires an Apple Silicon Mac with **512 GB unified memory**, **macOS 27+ (Metal 4.1)**, Python 3.12, Git, and Xcode command-line tools.

### 1. Clone and build the TensorFold `v1.3.3` Metal 4.1 recipe

```sh
git clone https://github.com/swbiggart/ds41-1017-studio.git
cd ds41-1017-studio
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'huggingface_hub==1.31.0' hf_xet
python -m mac build
```

### 2. Download this repacked checkpoint directly

```sh
HF_XET_HIGH_PERFORMANCE=1 hf download swbiggart/DeepSeek-V4.1-Flash-1017-TensorFold-Metal-Lossless --local-dir .local/ds41-repacked
```

### 3. Point `local.toml` at the downloaded directory and serve

Create `local.toml` in the `ds41-1017-studio` root:

```toml
[model]
model_dir = ".local/ds41-repacked"
engram_dir = ".local/ds41-repacked/engram"
assets_dir = ".local/ds41-repacked/assets"
```

Validate all manifests/checksums and start the OpenAI-compatible server (`http://127.0.0.1:8890/v1`, model ID `deepseek-v41-flash-1017-official`):

```sh
python -m mac validate
python -m mac up
python -m mac chat 'Write a Python generator that merges two sorted iterators.' --max-tokens 128
```

---

## Measured performance & parity (80-core M5 Ultra 512 GB)

Measured uncontended on an 80-core Apple M5 Ultra Mac Studio (512 GB unified memory, macOS Metal 4.1, `0 MB` swap) serving these exact bit-identical artifacts via [`swbiggart/ds41-1017-studio`](https://github.com/swbiggart/ds41-1017-studio) (`PROFILES.md` has full per-request JSON receipts):

| Workload | Measured | Notes |
|---|---:|---|
| Greedy decode (7 workloads, adaptive DSpark) | **97.06 tok/s** | Peaks at **133.1 tok/s** (`multiply-17-23`); **804/804** exact raw token IDs & **7/7** finish reasons vs serial (**1.84×**) |
| Greedy decode (7 workloads, `--no-drafts`) | **52.84 tok/s** | Serial control with identical target arithmetic |
| Filtered sampled (`T=0.8, k=20, p=0.95`) | **116.3 / 92.8 tok/s** | Code / chat; **2.19× / 1.76×** vs serial, **100% exact** token IDs |
| Pure nucleus (`T=0.8, k=0, p=0.95`) | **97.0 / 78.2 tok/s** | Code / chat; **2.81× / 2.28×** vs serial (`34.5 / 34.3 tok/s`), **100% exact** token IDs |
| 32,768-token Python prompt (cold TTFT / decode) | **8.78 s** (`3,733 tok/s`) / **119.2 tok/s** | **2.47× faster prefill** (`-12.91 s`) than stock TensorFold `v1.3.3` (`8.75 s` vs `21.67 s` serial matched pair) |
| 32,768-token prompt-cache hit | **0.0087 s** / **123.2 tok/s** | **100% exact** token IDs & finish reason vs cold run (`32,768` cached tokens) |
| 131,072-token Python prompt (cold TTFT / decode) | **33.44 s** (`3,920 tok/s`) / **97.0 tok/s** | Default 139,264-token context; **0 MB swap** |
| 131,072-token prompt-cache hit | **0.0143 s** / **98.6 tok/s** | **100% exact** token IDs & finish reason vs cold run (`131,072` cached tokens) |
| Concurrent drafted serving (`p=1 / 2 / 4 / 8`) | **120.9 / 106.0 / 121.9 / 127.2 tok/s** | Active-window throughput across 1–8 simultaneous clients |

---

## Cryptographic provenance & integrity verification

Every artifact in this repository was built deterministically from DeepSeek's pinned revision and cross-verified bit-for-bit against both the upstream Hub LFS SHA-256s and the locally benchmarked M5 Ultra build:

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| Upstream `deepseek-ai/DeepSeek-V4.1-Flash-1017` commit | — | `08c8657f02291e11b6f6b63e0e16014f8e748518` |
| Upstream `model.safetensors.index.json` | — | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| Repacked `model.safetensors.index.json` | — | `6bf6178ed4002972f97d410d01241bf123000c1464d01f576b698a700cf24200` |
| 45 aligned `model-*.safetensors` shards (total) | `307,539,247,104` | Per-shard SHA-256s in `repack-manifest.json` |
| `engram/engram-l1-r0of1.bin` | `101,377,632,448` | `5db321aff48c80a3f45d2d7176d42b44bcd256f8e802487b104db755fce980aa` |
| `engram/engram-l14-r0of1.bin` | `101,380,408,144` | `a7a62bd1626f140417c771f945a90676cb1810de10dfd38ca2b5cf537478ce46` |
| `assets/engram-token-map.bin` | `517,680` | `b9a7cf9d341f5daf8fc435dbc36153bf4b28c1b8075be20de30d0556fb910511` |

To verify all 45 shard alignments and SHA-256 digests locally after downloading:

```sh
python -c "from pathlib import Path; from mac import repack; repack.verify_repack(Path('.local/ds41-repacked'), check_sha256=True); print('All 45 shards verified OK')"
```

---

## License & Attribution

- **Model weights & upstream files:** Licensed under the [MIT License](LICENSE) — Copyright (c) 2023 DeepSeek. Original repository: [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
- **Serving recipe & repacker:** [`swbiggart/ds41-1017-studio`](https://github.com/swbiggart/ds41-1017-studio) (Apache-2.0), built on [`ashhart/TensorFold`](https://github.com/ashhart/TensorFold) `v1.3.3` (`1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc`).

### Upstream citation

```bibtex
@misc{deepseekai2026deepseekv41flash,
      title={DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression},
      author={DeepSeek-AI},
      year={2026},
}
```
