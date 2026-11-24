---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: transformers
pipeline_tag: text-generation
tags: [autoround, int4, int3, mixed-precision, moe, vllm, fp8, mtp]
---

# Qwen3.8-Flash-Next-3.43BPW-MTP

Mixed-precision **AutoRound** quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
for vLLM, designed for **64 GB GPUs needing extra KV-cache headroom** without sacrificing any MoE layers to 2-bit.

- **Non-PLE backbone (GPU):** **50.22 GiB** (**3.43 bits per weight** average across packed weights, scales, zero points, and BF16 parts) — **zero W2 layers**
- **Embedded MTP speculative-decoding head (GPU):** **+1.03 GiB** (`W3 g128 sym` experts; `W8 g128` QSA + `fc_*`) → **51.25 GiB total GPU weights** (**2.02 GiB smaller** than `3.55BPW-MTP`)
- **Per-Layer Embedding (PLE, CPU RAM):** Qwen's official FP8 E4M3 table + global scale (**~47.7 GiB**, offloaded to host memory by vLLM)

---

## Precision layout

| Component | Format |
|---|---|
| Routed experts, all 48 MoE layers (L0–L47) | W3A16, group 128, symmetric (zero W2) |
| GDN linear attention (`in_proj_qkv/z`) | W4A16, group 64, symmetric |
| GDN linear attention (`out_proj`) | W8A16, group 128, symmetric |
| QSA self-attention (`q/k/v/o_proj`, `indexer.index_qk_proj`) | W8A16, group 128, symmetric |
| Routed shared experts (`mlp.shared_expert`) | W8A16, group 128, symmetric |
| `mlp.gate`, `shared_expert_gate`, `in_proj_a/b`, hyper-connections, `ple.(key\|value)_proj`, embeddings, LM head, norms | BF16 |
| Per-layer embedding (PLE) | FP8 E4M3 + global scale (official Qwen tensors) |
| MTP head (`mtp.layers.0`, `+1.03 GiB`) | Experts W3A16 g128 sym; `self_attn` (incl. `index_qk_proj`) & `fc_*` W8A16 g128 sym; gate/shared/HC/norms BF16 |

AutoRound 0.16.1 (SignRound), **400 iterations/block**, **512 × 2,048-token** calibration samples (**1.05M tokens** — 2.67× larger than `3.55BPW-MTP`; balanced 5-domain chat / code / math / long-context / prose with Qwen ChatML template; `best-MSE=False`). Format `auto_round:auto_gptq`.

---

## Fidelity vs. BF16 (PLE Active)

Teacher-forced **full-vocabulary** `KL(P_bf16 ‖ Q_quant)` on a held-out set (`24 × 1,024` tokens; sources and splits disjoint from calibration). Both BF16 teacher and quantized candidate run with **PLE active** (SSD-streamed BF16 PLE teacher vs. official FP8 PLE candidate):

| Domain | Positions | Mean KL | Median KL | P99 KL | Top-1 agreement | PPL bf16 → quant |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 24,552 | **0.11355** | 0.00924 | 1.9539 | 91.13% | 3.507 → 3.540 (+0.96%) |
| chat | 10,230 | 0.09287 | 0.02619 | 1.0880 | 87.83% | 7.686 → 7.635 |
| code | 5,115 | 0.11181 | 0.00033 | 2.5067 | 95.60% | 1.842 → 1.815 |
| math | 5,115 | 0.06561 | 0.00258 | 1.2782 | 94.57% | 2.308 → 2.369 |
| prose | 4,092 | 0.22733 | 0.00792 | 3.3509 | 89.49% | 1.859 → 1.974 |

`P99.9 KL = 5.5255`, `max = 13.872`. Static vLLM INC dry-check passed (`GDN=4b g64, out=8b; QSA=8b g128; MoE=3b g128`).

---

## The Story of These Quants (`3.48BPW` → `3.55BPW-MTP` → `3.43BPW-MTP`)

Every checkpoint in this series was evaluated on the **identical** held-out `24 × 1,024` evalset with **real PLE active**:

1. **[`Qwen3.8-Flash-Next-3.48BPW`](https://huggingface.co/Beling86/Qwen3.8-Flash-Next-3.48BPW) (v1 baseline, `51.07 GiB` GPU, no MTP):**
   First zero-W2 build (`48× W3 g128` MoE, `GDN/QSA W8`, `shared_expert` BF16, `104×2048` = 213K calibration tokens). Established the `0.12488` PLE-active KL baseline (`90.92%` top-1).
2. **[`Qwen3.8-Flash-Next-3.55BPW-MTP`](https://huggingface.co/Beling86/Qwen3.8-Flash-Next-3.55BPW-MTP) (max-fidelity build, `51.94 GiB` backbone + `1.33 GiB` MTP = `53.27 GiB` GPU):**
   Quantizing `shared_expert` from BF16 → `W8 g128` freed `0.23 GiB` (while improving KL), and per-layer PLE-active sensitivity identified the top high-impact layers (`2× W4 g128` on `L46–47` + `11× W3 g64` on `L1, 3, 37–45`). Paired with `192×2048` (393K tokens) and `400` iterations, this cut KL by **−14.0%** to **`0.10734`** (`91.70%` top-1) and added a `W4 g128` MTP head.
3. **`Qwen3.8-Flash-Next-3.43BPW-MTP` (this checkpoint — compact zero-W2 build, `50.22 GiB` backbone + `1.03 GiB` MTP = `51.25 GiB` GPU):**
   PLE-active ablation showed that dropping `GDN in_proj_qkvz` to **`W4 g64`** (while keeping `GDN out_proj`, `QSA`, and `shared_expert` at `W8 g128`) plus a `W3 g128` MTP head reclaims **`2.02 GiB` of GPU VRAM** with far lower KL cost than introducing any W2 MoE layers. Scaling calibration **2.67×** to **`512 × 2,048` (1.05M tokens)** delivered **`0.11355` KL** — **−9.1% better than `3.48BPW`** despite being `0.85 GiB` smaller on backbone with MTP added, and only **`+0.0062` KL** above `3.55BPW-MTP` while **beating `3.55BPW-MTP` by −12.6% on `chat`**!

### Overall Evolution
| Checkpoint | Backbone (GPU) | MTP (GPU) | Total GPU | Key Layout & Calibration | Mean KL *(PLE active)* | P99 KL | Top-1 | PPL *(bf16 → quant)* |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| [`3.48BPW`](https://huggingface.co/Beling86/Qwen3.8-Flash-Next-3.48BPW) | `51.07 GiB` | — | `51.07 GiB` | 48× W3g128, GDN W8, SE BF16 · `104×2048` | `0.12488` | `2.0779` | `90.92%` | — |
| [`3.55BPW-MTP`](https://huggingface.co/Beling86/Qwen3.8-Flash-Next-3.55BPW-MTP) | `51.94 GiB` | `1.33 GiB` (`W4`) | `53.27 GiB` | 2× W4 + 11× W3g64 + 35× W3g128, GDN/QSA/SE W8 · `192×2048` | **`0.10734`** | **`1.7738`** | **`91.70%`** | `3.486 → 3.408` |
| **`3.43BPW-MTP` (this)** | **`50.22 GiB`** | **`1.03 GiB` (`W3`)** | **`51.25 GiB`** | **48× W3g128 (zero W2), GDN `qkvz` W4g64, out/QSA/SE W8 · `512×2048`** | **`0.11355`** | `1.9539` | `91.13%` | **`3.507 → 3.540`** |

### Domain Comparison (`3.55BPW-MTP` vs. `3.43BPW-MTP`)
| Domain | Positions | `3.55BPW-MTP` KL / P99 / Top-1 | **`3.43BPW-MTP` KL / P99 / Top-1** | Notes |
|---|---:|---:|---:|---|
| **chat** | 10,230 | `0.10621` / `1.442` / `88.69%` | **`0.09287` / `1.088`** / `87.83%` | **`3.43BPW-MTP` wins (−12.6% KL, −24.5% P99)** |
| **code** | 5,115 | **`0.08917` / `1.697`** / `95.60%` | `0.11181` / `2.507` / **`95.60%`** | Tied top-1; `3.55BPW-MTP` tighter tail |
| **math** | 5,115 | **`0.06035` / `1.191`** / `94.84%` | `0.06561` / `1.278` / `94.57%` | Near-tie (`+0.0053` KL) |
| **prose** | 4,092 | **`0.19161` / `2.675` / `90.42%`** | `0.22733` / `3.351` / `89.49%` | `3.55BPW-MTP` wins via `W4`/`W3g64` |

**Which variant should I pick?**
- Pick **`3.43BPW-MTP`** (this repo) for **64 GB GPUs** when you want **+2.02 GiB extra KV-cache headroom**, MTP speculative decoding, zero W2 layers, and the lowest `chat` KL.
- Pick **[`3.55BPW-MTP`](https://huggingface.co/Beling86/Qwen3.8-Flash-Next-3.55BPW-MTP)** if you have the extra 2 GiB VRAM and want maximum fidelity on `code` and `prose`.

---

## Serving (vLLM)

The checkpoint ships the official FP8 per-layer-embedding (PLE) shards with their global scale (`ple-*.safetensors`), `ple_embedding_dtype: float8_e4m3fn` in `config.json`, and uniform fused QSA/GDN quantization.

```bash
python vllm-patch.py              # run with the python of your vLLM environment (patches INC RoutedExperts bit-width lookup)
vllm serve Beling86/Qwen3.8-Flash-Next-3.43BPW-MTP --trust-remote-code --dtype bfloat16 --engram-config '{"cpu_offload": true, "use_thp": true}'
```

## Validation gates passed

Coverage, packed layout vs. config, zero points vs. `sym`, fused-group consistency (GDN `in_proj_qkvz` + QSA `qkv_proj`/`index_qk_proj` + MoE), size ≤ 50.3 GiB (non-PLE backbone), plan conformance, strict reload (0 missing/mismatched keys) for the KL evaluation, PLE shards and global scale, and static vLLM INC dry-check.
