---
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- llama.cpp
- qwen3.8
- moe
- quantized
license: apache-2.0
---

# Qwen3.8-Flash-Next-3.5bpw-GGUF

Native **`llama.cpp` (`3.5 bpw`)** 2-shard GGUF quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` Total Parameters / `48` Hybrid GDN + Full-Attention Layers / `512` Routed + `1` Shared Expert / `4x` Hyper-Connections / Layer-1 N-Gram Per-Layer Embedding).

Uses 100% standard upstream `ggml` quantization formats (`Q3_K`, `Q4_0`, `Q2_0`, `Q6_K`, `Q8_0`, `BF16`, `F32`) for direct out-of-the-box execution in `llama.cpp` (`qwen4exp` architecture) with zero custom decoders or patches required.

## Shards & Memory Footprint

| File | Size | Placement | Contents |
| :--- | :---: | :---: | :--- |
| `Qwen3.8-Flash-Next-3.5bpw-00001-of-00002.gguf` | `55.05 GB` (`51.27 GiB`) | GPU VRAM | Token embeddings, all `48` GDN / Attention / MoE layers (`1,223` tensors, `3.51 bpw` effective), & output head |
| `Qwen3.8-Flash-Next-3.5bpw-00002-of-00002.gguf` | `28.80 GB` (`26.82 GiB`) | Host RAM (`mmap`) | Layer-1 N-Gram Per-Layer Embedding table (`Q4_0`, `1` tensor) |

## Precision Allocation

- **Output Head (`output.weight`) & Attention / SSM Linear Projections:** `Q6_K`
- **Shared Experts (`ffn_*_shexp.weight`) & Hyper-Connection Mixers:** `Q8_0`
- **Token Embeddings & High-Sensitivity Routed `down_exps` (`16/48` layers, incl. all Full-Attention anchors):** `Q4_0`
- **Routed `gate_exps` & `up_exps` (`48/48` layers):** `Q3_K`
- **Remaining Routed `down_exps` (`32/48` layers):** `Q2_0`
- **Norms, Router Gates (`ffn_gate_inp.weight`), Shared Gates, & SSM `A` / Bias Vectors:** `F32` / `BF16`

## Held-Out Logit & Perplexity Evaluation (`248,320` Vocab)

End-to-end `48`-layer teacher-vs-quantized logit evaluation over the full `248,320`-token vocabulary:

| Evaluation Suite | `BF16` PPL | `3.5bpw GGUF` PPL (`ΔPPL`) | Mean `KLD` | Median `KLD` | Mean `JSD` | `Top-1` Match | `Top-5` in `BF16` | Final `CosSim` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Self-Gen Math/Code` (`<think>`)** | `5.437` | **`5.521` (`+0.084`)** | **`0.10509`** | `0.04360` | `0.02348` | **`87.98%`** | **`98.83%`** | **`0.952932`** |
| **`External Held-Out` (`2048`)** | `11.142` | **`10.134` (`-1.007`)** | **`0.15463`** | `0.08019` | `0.03315` | **`84.22%`** | **`98.73%`** | **`0.949186`** |
| **`WikiText-2` (`4x2048`)** | `2.156` | **`2.402` (`+0.245`)** | **`0.32467`** | `0.02330` | `0.05923` | **`85.14%`** | **`97.29%`** | **`0.890650`** |
| **`WikiText-2` (`4x1024`)** | `2.320` | **`2.599` (`+0.280`)** | **`0.34798`** | `0.02854` | `0.06393` | **`83.77%`** | **`97.02%`** | **`0.888777`** |

## Usage (`llama.cpp`)

Point `llama-cli` or `llama-server` to Shard `00001-of-00002` (Shard `00002` is discovered automatically in the same directory and memory-mapped into host RAM):

```bash
./llama-cli \
  -m Qwen3.8-Flash-Next-3.5bpw-00001-of-00002.gguf \
  -ngl 99 \
  -c 32768 \
  --temp 0.6 --top-p 0.95 --top-k 20
```
