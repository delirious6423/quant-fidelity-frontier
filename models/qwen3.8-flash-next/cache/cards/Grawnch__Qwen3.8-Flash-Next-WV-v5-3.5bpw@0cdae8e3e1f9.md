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

# Qwen3.8-Flash-Next-WV-v5-3.5bpw (Master 2-Shard Wave-V5 Container)

Full 48-layer **`WV-v5` (`3.5 bpw`)** master checkpoint of **`Qwen/Qwen3.8-Flash-Next`** (`180B` total / `512` routed experts, Top-`10` active + shared expert, `36` GDN + `12` sparse-QSA layers), built with the complete Sprint 3.4 `WV-v5` recipe:

- **Zero-Repeat `v2` Multi-Domain Calibration:** Genuine on-policy `<think>...</think>` math/logic/coding trajectories + 12-language polyglot code (`python, rust, go, cpp, java, javascript, typescript, csharp, ruby, php, bash, sql`) + WikiText-2 (`50%` prose / `50%` math+code Hessian balance).
- **Temporal-Quarter `Early+Late Minimax` on Pre-QSA `GDN-2` (`l % 4 == 2`):** Replaces monotonic `+U4` horizon ramp with `0.40 Q1 + 0.10 Q2 + 0.10 Q3 + 0.40 Q4` weighting on `L02, L06, ..., L46` to preserve early-context key recall feeding `QSA` layers.
- **Marchenko-Pastur `MP+DG-Flat` RMT Hessian Denoising (`r/D < 0.5`):** Analytically shrinks undersampled cold-expert empirical Hessian eigenvalues while preserving signal spikes (`lam > lam_+`).
- **True `Q8_0` `output.weight` (`lm_head`) w/ Pass D+ Gain & 9-Token Double Gram-Schmidt LDA Surgery:** Cuts head-only KLD vs `BF16` by **`-24.5%` to `-47.8%`** (`0.000197` Math/Code head KLD, `0.000279` WikiText-2 head KLD).

## Verified Quality Metrics

| Benchmark Suite | Metric | Score |
| :--- | :--- | :---: |
| **48-Layer Direct vs `BF16` (`External Held-Out 1x2048`)** | PPL (`BF16` -> `WV-v5`) / CosSim | **`11.422 -> 10.068` (`dPPL = -1.3539`)** / `0.9481` |
| **48-Layer Direct vs `BF16` (`Self-Gen Math/Code 1x2048`)** | Mean KLD / Top-1 / CosSim | **`0.11252`** / `89.30%` / `0.9515` |
| **48-Layer Direct vs `BF16` (`WT2_2048`)** | PPL (`BF16` -> `WV-v5`) / CosSim | **`2.164 -> 2.363`** / `0.9429` |
| **`Strata` v1 5-Domain Yardstick (`4,995` Positions)** | Reference PPL (`code` / `all`) | **`2.108` / `3.920`** |
| **Repacked `SM120-v20` `3.1 bpw` `128K` `MathArena/aime_2025`** | Pass@1 (`B=30, pp=0.0`, Vanilla) | **`30 / 30` (`100.00%`, `15/15 + 15/15`, `0` loops)** |

## Shards
- `Qwen3.8-Flash-Next-WV-v5-3.5bpw-00001-of-00002.gguf` (`55.31 GB`): 48-layer `WZV4` MoE, `Q8_0` dense spine/shared-experts/head, `FP32` norms & GDN gates, `Q4_0` token embeddings.
- `Qwen3.8-Flash-Next-WV-v5-3.5bpw-00002-of-00002.gguf` (`30.51 GB`): Per-Layer Embeddings (`PLE`, zero-copy host `mmap`).
