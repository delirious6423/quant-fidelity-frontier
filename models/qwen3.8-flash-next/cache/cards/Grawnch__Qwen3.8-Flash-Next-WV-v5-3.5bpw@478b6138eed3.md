---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - quantization
  - wave-quant
  - wv-v5
  - moe
  - qwen3.8-flash-next
---

> [!WARNING]
> **Superseded:** A newer checkpoint is recommended: [Grawnch/Qwen3.8-Flash-Next-WV-v5-3.5bpw-GGUF](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-v5-3.5bpw-GGUF) — WV-v5-3.5bpw-GGUF (or WV-2.96bpw-SM120 for bare-metal Blackwell).
> Kept here for benchmark reproducibility and historical reference.

# Qwen3.8-Flash-Next-WV-v5-3.5bpw (Master 2-Shard Container)

Full 48-layer **`WV-v5` (`3.5 bpw`)** checkpoint of **`Qwen/Qwen3.8-Flash-Next`** (`180B` total / `512` routed experts, Top-`10` active + shared expert, `36` GDN + `12` sparse-QSA layers).

## Verified Quality Metrics

| Benchmark Suite | Metric | Score |
| :--- | :--- | :---: |
| **48-Layer Direct vs `BF16` (`External Held-Out 1x2048`)** | PPL (`BF16` -> `WV-v5`) / CosSim | **`11.422 -> 10.068` (`dPPL = -1.3539`)** / `0.9481` |
| **48-Layer Direct vs `BF16` (`Self-Gen Math/Code 1x2048`)** | Mean KLD / Top-1 / CosSim | **`0.11252`** / `89.30%` / `0.9515` |
| **48-Layer Direct vs `BF16` (`WT2_2048`)** | PPL (`BF16` -> `WV-v5`) / CosSim | **`2.164 -> 2.363`** / `0.9429` |
| **`Strata` v1 5-Domain Yardstick (`4,995` Positions)** | Reference PPL (`code` / `all`) | **`2.108` / `3.920`** |
| **Repacked `SM120-v20` `3.1 bpw` `128K` `MathArena/aime_2025`** | Pass@1 (`B=30, pp=0.0`, Vanilla) | **`30 / 30` (`100.00%`, `15/15 + 15/15`, `0` loops)** |

## Shards
- `Qwen3.8-Flash-Next-WV-v5-3.5bpw-00001-of-00002.gguf` (`55.31 GB`): 48-layer quantized MoE, `Q8_0` dense/shared-experts/head, `FP32` norms & GDN gates, `Q4_0` token embeddings.
- `Qwen3.8-Flash-Next-WV-v5-3.5bpw-00002-of-00002.gguf` (`30.51 GB`): Per-Layer Embeddings (`PLE`, zero-copy host `mmap`).
