---
base_model: Qwen/Qwen3.8-Flash-Next
library_name: transformers
pipeline_tag: text-generation
tags: [autoround, int3, mixed-precision, moe, vllm, fp8]
---

# Qwen3.8-Flash-Next-3.48BPW

Mixed-precision **AutoRound** quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
for vLLM. Text-only (vision and MTP removed). Non-PLE weights: **51.07 GiB**, **3.48 bits per
weight** on average (packed weights, scales, zero points and BF16 parts). The 51.2 B-parameter per-layer embedding (PLE)
is Qwen's own FP8 table with its global scale (~47.7 GiB, offloaded to host memory by vLLM).

## Precision layout

| Component | Format |
|---|---|
| Routed experts, all 48 layers | W3A16, group 128, symmetric |
| GDN linear attention, QSA self-attention projections | W8A16, group 128, symmetric |
| Indexer, decay/gate controls, hyper-connections, shared experts, routers, embeddings, LM head, norms | BF16 |
| Per-layer embedding (PLE) | FP8 E4M3 + global scale (official Qwen tensors) |

AutoRound 0.16.1 (SignRound), 500 iterations/block, 236 x 2048-token
calibration samples (chat/code/reasoning/long-context/wikitext prose, production chat template; best-MSE=True). Format `auto_round:auto_gptq`.

## Fidelity vs. BF16

Teacher-forced **full-vocabulary** KL(P_bf16 ‖ Q_quant) on a held-out set (24 × 1,024 tokens; sources and splits disjoint
from calibration). Both models run **without the PLE injection** (the BF16 PLE table does not fit the quantization host),
so this measures the quantization of the transformer blocks only:

| Domain | Positions | Mean KL | Median KL | P99 KL | Top-1 agreement | PPL bf16 → quant |
|---|---:|---:|---:|---:|---:|---:|
| **all** | 24552 | **0.09578** | 0.01605 | 1.3431 | 90.53% | 5.339 → 5.541 |
| chat | 10230 | 0.09869 | 0.02694 | 1.2333 | 88.76% | 9.926 → 10.597 |
| code | 5115 | 0.07578 | 0.00088 | 1.3812 | 94.98% | 2.739 → 2.756 |
| math | 5115 | 0.06486 | 0.00440 | 1.1120 | 94.06% | 2.648 → 2.679 |
| prose | 4092 | 0.15218 | 0.05524 | 1.5814 | 85.02% | 6.265 → 6.505 |

P99.9 KL 4.2165, max 6.879.

### Progression Across `3.48BPW` Quantizations

| Version | Run | Config | Mean KL | P99 KL | P99.9 KL | Top-1 | Chat Top-1 | Code Top-1 | Math Top-1 | Prose Top-1 | PPL bf16 → quant |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 (Baseline) | `20261114-213016` | 104 samples, 250 iters, final-iter | 0.11572 | 1.5207 | 4.5617 | 88.78% | 85.61% | 94.43% | 92.98% | 84.38% | 5.431 → 5.677 |
| v2 | `run-20261116-132458` | 236 samples, 400 iters, best-MSE=True | 0.10026 | 1.4335 | 4.2587 | 90.47% | **89.32%** | 94.41% | 93.76% | 84.31% | 5.481 → 5.608 |
| **v3 (Final)** | `run-20261116-175428` | 236 samples, 500 iters, best-MSE=True | **0.09578** | **1.3431** | **4.2165** | **90.53%** | 88.76% | **94.98%** | **94.06%** | **85.02%** | **5.339 → 5.541** |
| **Δ (v1 → v3)** | — | — | **-17.2%** | **-11.7%** | **-7.6%** | **+1.75 pp** | **+3.15 pp** | **+0.55 pp** | **+1.08 pp** | **+0.64 pp** | — |

## Serving (vLLM)

The checkpoint ships the official FP8 per-layer-embedding (PLE) shards with their global scale
(`ple-*.safetensors`) and `ple_embedding_dtype: float8_e4m3fn` in `config.json`.

```bash
python vllm-patch.py              # run with the python of your vLLM environment (see below)
vllm serve Beling86/Qwen3.8-Flash-Next-3.48BPW --trust-remote-code --dtype bfloat16 --engram-config '{"cpu_offload": true, "use_thp": true}'
```

`vllm-patch.py` (in this repo; idempotent, keeps `.orig` backups, `--check` / `--revert`) fixes a vLLM bug we hit
while serving a sibling checkpoint with per-expert mixed bit-widths: vLLM's INC parser reads the per-layer bits of a
fused MoE layer only if the class name contains `fusedmoe`, but the class is now `RoutedExperts`, so every MoE layer
falls back to the global `bits: 4` and the 2/3-bit experts are allocated with 4-bit containers (CUDA OOM while the model
is constructed, before any weight is read). This checkpoint has the same layout (W2/W3 experts under a global
`bits: 4`), so expect the same failure on stock vLLM without the patch. The Ling-specific KDA patch in that script
does not apply to this model and reports "not applicable".

2/3-bit experts are meant to run on the Humming kernels, 8-bit projections on Marlin. If you serve this model, please
report what happens.

## Validation gates passed

Coverage, packed layout vs. config, zero points vs. `sym`, fused-group consistency, size ≤ 52.0 GiB
(non-PLE weights), plan conformance, strict reload (no missing keys) for the KL evaluation, PLE shards and global scale.
