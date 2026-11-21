---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: transformers
pipeline_tag: text-generation
tags: [autoround, int4, int3, mixed-precision, moe, vllm, fp8]
---

# Qwen3.8-Flash-Next-3.55BPW-MTP

Mixed-precision **AutoRound** quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
for vLLM. Includes packed `W4 g128 sym` MTP speculative-decoding head (+1.34 GiB; `W8 g128` QSA + `fc_*`). Non-PLE backbone weights: **51.969 GiB**, **3.5451 bits per
weight** on average (packed weights, scales, zero points and BF16 parts). The 51.2 B-parameter per-layer embedding (PLE)
is Qwen's official FP8 table with its global scale (~47.7 GiB, offloaded to host memory by vLLM).

## Precision layout

| Component | Format |
|---|---|
| Routed experts, layers 46, 47 | W4A16, group 128, symmetric |
| Routed experts, 11 layers (1, 3, 37, 38, 39, 40, 41, 42, 43, 44, 45) | W3A16, group 64, symmetric |
| Routed experts, remaining 35 layers | W3A16, group 128, symmetric |
| GDN linear attention, QSA self-attention (incl. `indexer.index_qk_proj`) | W8A16, group 128, symmetric |
| Routed shared experts (`mlp.shared_expert`) | W8A16, group 128, symmetric |
| `mlp.gate`, `shared_expert_gate`, `in_proj_a/b`, hyper-connections, `ple.(key\|value)_proj`, embeddings, LM head, norms | BF16 |
| Per-layer embedding (PLE) | FP8 E4M3 + global scale (official Qwen tensors) |
| MTP head (`mtp.layers.0`) | Experts W4A16 g128 sym; `self_attn` (incl. `index_qk_proj`) & `fc_*` W8A16 g128 sym; gate/shared/HC/norms BF16 |

AutoRound 0.16.1 (SignRound), 400 iterations/block, 192 x 2048-token
calibration samples (`calib_qwen_v3_512x2048.jsonl`, Qwen ChatML template; best-MSE=False). Format `auto_round:auto_gptq`.

## Fidelity vs. BF16

Teacher-forced **full-vocabulary** KL(P_bf16 ‖ Q_quant) on a held-out set (24 × 1,024 tokens; sources and splits disjoint
from calibration). Both BF16 teacher and quantized candidate run with **PLE active** (SSD-streamed BF16 PLE vs. official FP8 PLE).

| Domain | Positions | Mean KL | Median KL | P99 KL | Top-1 agreement | PPL bf16 → quant |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 24552 | **0.10734** | 0.00854 | 1.7738 | 91.70% | 3.486 → 3.408 |
| chat | 10230 | 0.10621 | 0.02603 | 1.4421 | 88.69% | 7.668 → 6.898 |
| code | 5115 | 0.08917 | 0.00035 | 1.6966 | 95.60% | 1.810 → 1.830 |
| math | 5115 | 0.06035 | 0.00257 | 1.1911 | 94.84% | 2.317 → 2.388 |
| prose | 4092 | 0.19161 | 0.00619 | 2.6749 | 90.42% | 1.837 → 1.986 |

P99.9 KL 4.8578, max 14.022.

### Evolution & Journey (`3.48BPW` → `Qwen3.8-Flash-Next-3.55BPW-MTP`)

| Stage / Checkpoint | Config & Key Changes | Mean KL *(real PLE)* | P99 KL | Top-1 |
|---|---|---:|---:|---:|
| **`Qwen3.8-Flash-Next-3.48BPW`** (`run-20261114-213016`) | All 48 MoE `W3 g128`, `shared_expert` BF16, `104 × 2048`, `best-MSE=True` *(51.071 GiB; earlier 0.09578 card was measured without PLE)* | 0.12488 | 2.0779 | 90.92% |
| `Nov 18 3.55BPW-MTP` (`run-20261118-162556`) | Added `W4` `L46–47`, `W3 g64` `L1,3,37–45`, `W8` `shared_expert` + `index_qk_proj` + MTP (`+1.33 GiB`), but hit shared foreign `<role>` template & `best-MSE=True` | 0.15897 | 2.4166 | 90.26% |
| **Qwen3.8-Flash-Next-3.55BPW-MTP** (`run-20261120-091813`) | **Native Qwen ChatML 5-domain corpus** (`chat/code/math/long/prose`, SHA+template gated), `192 × 2048`, `BS=8`, `400` iters, `not_use_best_mse=True` | **0.10734** | **1.7738** | **91.70%** |

Key milestones from `3.48BPW` to `Qwen3.8-Flash-Next-3.55BPW-MTP`:
1. **Apples-to-apples PLE evaluation:** Replaced the no-PLE teacher with SSD-streamed BF16 PLE vs. official FP8 PLE (`0.12488` true baseline for `3.48BPW`).
2. **Per-layer sensitivity & fused-group alignment:** Reallocated budget under `≤ 52 GiB` backbone by promoting tail/sensitive layers (`L46–47` → `W4 g128`; `L1, 3, 37–45` → `W3 g64`), quantizing `shared_expert` to `W8 g128` (`-0.215 GiB`), locking `indexer.index_qk_proj` to `W8 g128` for vLLM QSA fusion, and packing `mtp.layers.0` (`W4 g128` experts / `W8` QSA).
3. **Corpus & optimization hygiene:** Eliminated the Nov 18 template mismatch via strict ChatML preflight, switched to `BS=8, SEQLEN=2048, not_use_best_mse=True`, and cut KL by **-14.0%** vs `3.48BPW` (`-32.5%` vs Nov 18), reaching **95.60% top-1 on code** (`0.08917` KL) and **94.84% on math** (`0.06035` KL).

## Serving (vLLM)

The checkpoint ships the official FP8 per-layer-embedding (PLE) shards with their global scale
(`ple-*.safetensors`), `ple_embedding_dtype: float8_e4m3fn` in `config.json`, and uniform `W8A16` QSA fusion (`q_proj`, `k_proj`, `v_proj`, `indexer.index_qk_proj`).

```bash
python vllm-patch.py              # run with the python of your vLLM environment (patches INC RoutedExperts bit-width lookup)
vllm serve Beling86/Qwen3.8-Flash-Next-3.55BPW-MTP --trust-remote-code --dtype bfloat16 --engram-config '{"cpu_offload": true, "use_thp": true}'
```

## Validation gates passed

Coverage, packed layout vs. config, zero points vs. `sym`, fused-group consistency (GDN + QSA `qkv_proj`/`index_qk_proj` + MoE), size ≤ 52.0 GiB
(non-PLE backbone), plan conformance, strict reload (no missing keys) for the KL evaluation, PLE shards and global scale.
