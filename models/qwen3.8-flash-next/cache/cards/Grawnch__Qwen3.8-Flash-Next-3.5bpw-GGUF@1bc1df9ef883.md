---
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- qwen3.8
- moe
- quantized
license: apache-2.0
---

# Qwen3.8-Flash-Next-3.5bpw-GGUF

Native **`llama.cpp` (`3.5 bpw`)** 2-shard GGUF quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` Total Parameters / `48` Hybrid GDN + Full-Attention Layers / `512` Routed + `1` Shared Expert / `4`-Stream mHC Hyper-Connections / Layer-1 N-Gram Per-Layer Embedding).

Uses 100% standard upstream `ggml` quantization formats (`Q3_K`, `Q4_0`, `Q2_0`, `Q6_K`, `Q8_0`, `BF16`, `F32`) for direct out-of-the-box execution in `llama.cpp` (`qwen4exp` architecture) with zero custom decoders or patches required.

## Model Family

| Repository | Format & Target Runtime | GPU VRAM | Host RAM (PLE) | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`Qwen3.8-Flash-Next-3.5bpw-GGUF`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF)** | 2-Shard Standard `llama.cpp` `GGUF` (`3.5 bpw`) | `55.05 GB` | `28.80 GB` (`mmap`) | 100% upstream `ggml` (`Q3_K`/`Q4_0`/`Q2_0`/`Q6_K`) drop-in for `llama.cpp` |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw)** | 2-Shard `WV` Non-Linear `GGUF` (`3.5 bpw`) | `55.12 GB` | `28.80 GB` (`mmap`) | Master lossless `WV` container (`ΔPPL +0.040` on `<think>`, `0.09757` KLD) |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw-SM120)** | Pre-swizzled `sm_120` `safetensors` + Host PLE | `66.70 GB` (`Mode 0`)<br>`45.25 GB` (`Mode 1`) | `28.80 GB` (`0 GB` VRAM) | **Up to `2,500.27 tok/s`** on 1× RTX PRO 6000; `< 6s` load; `80K` sparse `QSA` indexer |

---

## Shards & Memory Footprint

| File | Size | Placement | Contents |
| :--- | :---: | :---: | :--- |
| `Qwen3.8-Flash-Next-3.5bpw-00001-of-00002.gguf` | `55.05 GB` (`51.27 GiB`) | GPU VRAM | Token embeddings, all `48` GDN / Attention / MoE layers (`1,223` tensors, `3.51 bpw` effective), & output head |
| `Qwen3.8-Flash-Next-3.5bpw-00002-of-00002.gguf` | `28.80 GB` (`26.82 GiB`) | Host RAM (`mmap`) | Layer-1 N-Gram Per-Layer Embedding table (`Q4_0`, `1` tensor) |
| `benchmarks/qfn_stock_35bpw_eval_summary.json` | `32.4 KB` | Reference | Full 48-layer teacher-vs-quantized KLD/PPL/cosine evaluation logs |

## Precision Allocation

- **Output Head (`output.weight`) & Attention / SSM Linear Projections:** `Q6_K`
- **Shared Experts (`ffn_*_shexp.weight`) & Hyper-Connection Mixers:** `Q8_0`
- **Token Embeddings & High-Sensitivity Routed `down_exps` (`16/48` layers, incl. all Full-Attention anchors):** `Q4_0`
- **Routed `gate_exps` & `up_exps` (`48/48` layers):** `Q3_K`
- **Remaining Routed `down_exps` (`32/48` layers):** `Q2_0`
- **Norms, Router Gates (`ffn_gate_inp.weight`), Shared Gates, & SSM `A` / Bias Vectors:** `F32` / `BF16`

## Held-Out 48-Layer Logit & Perplexity Evaluation (`248,320` Vocab)

| Evaluation Suite | `BF16` PPL | `3.5bpw GGUF` PPL (`ΔPPL`) | Mean `KLD` | Median `KLD` | Mean `JSD` | `Top-1` Match | `Top-5` in `BF16` | Final `CosSim` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Self-Gen Math/Code` (`<think>`)** | `5.437` | **`5.521` (`+0.084`)** | **`0.10509`** | `0.04360` | `0.02348` | **`87.98%`** | **`98.83%`** | **`0.952932`** |
| **`External Held-Out` (`2048`)** | `11.142` | **`10.134` (`-1.007`)** | **`0.15463`** | `0.08019` | `0.03315` | **`84.22%`** | **`98.73%`** | **`0.949186`** |
| **`WikiText-2` (`4×2048`)** | `2.156` | **`2.402` (`+0.245`)** | **`0.32467`** | `0.02330` | `0.05923` | **`85.14%`** | **`97.29%`** | **`0.890650`** |
| **`WikiText-2` (`4×1024`)** | `2.320` | **`2.599` (`+0.280`)** | **`0.34798`** | `0.02854` | `0.06393` | **`83.77%`** | **`97.02%`** | **`0.888777`** |

## Usage (`llama.cpp`)

Point `llama-cli` or `llama-server` to Shard `00001-of-00002` (Shard `00002` is discovered automatically in the same directory and memory-mapped into host RAM):

```bash
./llama-cli \
  -m Qwen3.8-Flash-Next-3.5bpw-00001-of-00002.gguf \
  -ngl 99 \
  -c 32768 \
  --temp 0.6 --top-p 0.95 --top-k 20
```
